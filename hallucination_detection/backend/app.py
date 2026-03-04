"""
Flask API for WeakRAG-Detect hallucination detection system.

Extends LegalInsight's Flask backend with new endpoints for weak labeling,
bootstrapping control, and result inspection.

Existing endpoints (from LegalInsight — kept):
    POST /analyze    – full contract analysis
    GET  /health     – health check

New endpoints for DSC 253 project:
    POST /label              – run all 3 LFs on a single example
    POST /bootstrap          – trigger one bootstrapping iteration
    GET  /results/<domain>   – return latest evaluation metrics
    GET  /taxonomy/<domain>  – return hallucination taxonomy JSON
"""

import json
import logging
import os
import sys
import time
from typing import Optional

# Allow imports from parent package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request
from flask_cors import CORS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# ---------------------------------------------------------------------------
_pipeline: Optional[object] = None
_entailment_lf: Optional[object] = None
_consistency_lf: Optional[object] = None
_reflection_lf: Optional[object] = None
_label_model_cache: dict = {}
_bootstrap_state: dict = {}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LEGALINSIGHT_BASELINE = 0.7465


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from labeling_functions.label_model import WeakLabelPipeline
        _pipeline = WeakLabelPipeline()
    return _pipeline


def _get_consistency_lf():
    global _consistency_lf
    if _consistency_lf is None:
        from labeling_functions.semantic_consistency_lf import SemanticConsistencyLF
        _consistency_lf = SemanticConsistencyLF()
    return _consistency_lf


def _get_entailment_lf():
    global _entailment_lf
    if _entailment_lf is None:
        from labeling_functions.entailment_lf import EntailmentLF
        _entailment_lf = EntailmentLF()
    return _entailment_lf


def _get_reflection_lf():
    global _reflection_lf
    if _reflection_lf is None:
        from labeling_functions.reflection_token_lf import ReflectionTokenLF
        _reflection_lf = ReflectionTokenLF()
    return _reflection_lf


# ---------------------------------------------------------------------------
# Existing endpoints (kept from LegalInsight)
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint.

    Returns
    -------
    JSON
        {"status": "ok", "service": "WeakRAG-Detect"}
    """
    return jsonify({"status": "ok", "service": "WeakRAG-Detect", "legalinsight_baseline": LEGALINSIGHT_BASELINE})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Full contract analysis (ported from LegalInsight).

    Body: {"question": str, "context": str, "answer": str}

    Returns
    -------
    JSON
        Combined analysis from all LFs + Snorkel label model.
    """
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    context = data.get("context", "")
    answer = data.get("answer", "")

    if not (question and context and answer):
        return jsonify({"error": "question, context, and answer are required."}), 400

    try:
        result = _run_all_lfs(question, context, answer)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Error in /analyze")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# New endpoints for DSC 253 project
# ---------------------------------------------------------------------------

@app.route("/label", methods=["POST"])
def label_example():
    """Run all 3 LFs on a single example and return individual votes.

    Body:
        {"question": str, "context": str, "answer": str}

    Returns
    -------
    JSON
        {
            "entailment_lf":    {"label": int, "score": float},
            "consistency_lf":   {"label": int, "consistency_score": float},
            "reflection_lf":    {"label": int, "grounding_score": float,
                                  "entity_overlap": float, "hedging_detected": bool},
            "snorkel_combined": {"label": int, "probability": float},
            "legalinsight_comparison": {
                "prior_consistency_score": 0.7465,
                "current_consistency_score": float,
                "improvement": float
            }
        }
    """
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    context = data.get("context", "")
    answer = data.get("answer", "")

    if not (question and context and answer):
        return jsonify({"error": "question, context, and answer are required."}), 400

    try:
        result = _run_all_lfs(question, context, answer)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Error in /label")
        return jsonify({"error": str(exc)}), 500


@app.route("/bootstrap", methods=["POST"])
def trigger_bootstrap():
    """Run one bootstrapping iteration and return updated metrics.

    Body:
        {"domain": str, "method": "self_training" | "co_training"}

    Returns
    -------
    JSON
        {"iteration": int, "train_size": int, "val_f1": float, "newly_labeled": int}
    """
    data = request.get_json(force=True, silent=True) or {}
    domain = data.get("domain", "legal")
    method = data.get("method", "self_training")

    if domain not in ("legal", "medical", "scientific"):
        return jsonify({"error": "domain must be legal, medical, or scientific"}), 400
    if method not in ("self_training", "co_training"):
        return jsonify({"error": "method must be self_training or co_training"}), 400

    # Load pre-computed bootstrapping state (populated by experiment scripts)
    state_path = os.path.join(RESULTS_DIR, domain, "bootstrap_state.json")
    if not os.path.exists(state_path):
        return jsonify({
            "error": f"No bootstrapping state found for domain '{domain}'. "
                     f"Run experiments/run_{domain}.py first."
        }), 404

    try:
        with open(state_path) as f:
            state = json.load(f)
        return jsonify({
            "domain": domain,
            "method": method,
            "iteration": state.get("iteration", 0),
            "train_size": state.get("train_size", 0),
            "val_f1": state.get("val_f1", 0.0),
            "newly_labeled": state.get("newly_labeled", 0),
        })
    except Exception as exc:
        logger.exception("Error in /bootstrap")
        return jsonify({"error": str(exc)}), 500


@app.route("/results/<domain>", methods=["GET"])
def get_results(domain: str):
    """Return latest evaluation metrics for a domain.

    Returns
    -------
    JSON
        Full comparison table across all methods, loaded from CSV log.
    """
    if domain not in ("legal", "medical", "scientific"):
        return jsonify({"error": "domain must be legal, medical, or scientific"}), 400

    log_path = os.path.join(RESULTS_DIR, domain, "evaluation_log.csv")
    if not os.path.exists(log_path):
        return jsonify({
            "domain": domain,
            "message": f"No results yet. Run experiments/run_{domain}.py first.",
            "results": [],
        })

    try:
        import csv
        rows = []
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return jsonify({"domain": domain, "results": rows})
    except Exception as exc:
        logger.exception("Error reading results for domain %s", domain)
        return jsonify({"error": str(exc)}), 500


@app.route("/taxonomy/<domain>", methods=["GET"])
def get_taxonomy(domain: str):
    """Return hallucination taxonomy JSON for a domain.

    Returns
    -------
    JSON
        Taxonomy built by HallucinationTaxonomyBuilder.
    """
    if domain not in ("legal", "medical", "scientific"):
        return jsonify({"error": "domain must be legal, medical, or scientific"}), 400

    taxonomy_path = os.path.join(RESULTS_DIR, domain, "hallucination_taxonomy.json")
    if not os.path.exists(taxonomy_path):
        return jsonify({
            "domain": domain,
            "message": f"No taxonomy yet. Run experiments/run_{domain}.py first.",
        })

    try:
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)
        return jsonify(taxonomy)
    except Exception as exc:
        logger.exception("Error reading taxonomy for domain %s", domain)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_all_lfs(question: str, context: str, answer: str) -> dict:
    """Run all three LFs and the Snorkel label model on a single example.

    Returns
    -------
    dict
        Combined result from all LFs and the label model.
    """
    import numpy as np

    # --- Entailment LF ---
    entailment_result = {"label": -1, "score": 0.0}
    try:
        elf = _get_entailment_lf()
        lbl, score = elf.label_with_scores(context, answer)
        entailment_result = {"label": int(lbl), "score": round(float(score), 4)}
    except Exception as exc:
        logger.warning("Entailment LF failed: %s", exc)

    # --- Consistency LF ---
    consistency_result = {"label": -1, "consistency_score": 0.0}
    current_consistency_score = 0.0
    try:
        clf = _get_consistency_lf()
        lbl, score = clf.label(question, context, answer)
        current_consistency_score = float(score)
        consistency_result = {"label": int(lbl), "consistency_score": round(current_consistency_score, 4)}
    except Exception as exc:
        logger.warning("Consistency LF failed: %s", exc)

    # --- Reflection Token LF ---
    reflection_result = {"label": -1, "grounding_score": 0.5, "entity_overlap": 0.0, "hedging_detected": False}
    try:
        rlf = _get_reflection_lf()
        result = rlf.label(context, answer)
        reflection_result = {
            "label": int(result.get("label", -1)),
            "grounding_score": round(float(result.get("grounding_score", 0.5)), 4),
            "entity_overlap": round(float(result.get("entity_overlap", 0.0)), 4),
            "hedging_detected": bool(result.get("hedging_detected", False)),
            "misattribution": bool(result.get("misattribution", False)),
        }
    except Exception as exc:
        logger.warning("Reflection Token LF failed: %s", exc)

    # --- Snorkel combined ---
    snorkel_result = {"label": -1, "probability": 0.5}
    try:
        example = {"question": question, "context": context, "answer": answer}
        L = np.array([[
            entailment_result["label"],
            consistency_result["label"],
            reflection_result["label"],
        ]])
        pipeline = _get_pipeline()
        model = pipeline.fit_label_model(L)
        soft = pipeline.get_probabilistic_labels(L, model)
        prob = float(soft[0])
        snorkel_result = {"label": int(prob > 0.5), "probability": round(prob, 4)}
    except Exception as exc:
        logger.warning("Snorkel label model failed: %s", exc)

    return {
        "question": question,
        "answer": answer,
        "entailment_lf": entailment_result,
        "consistency_lf": consistency_result,
        "reflection_lf": reflection_result,
        "snorkel_combined": snorkel_result,
        "legalinsight_comparison": {
            "prior_consistency_score": LEGALINSIGHT_BASELINE,
            "current_consistency_score": round(current_consistency_score, 4),
            "improvement": round(current_consistency_score - LEGALINSIGHT_BASELINE, 4),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    logger.info("Starting WeakRAG-Detect Flask API on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
