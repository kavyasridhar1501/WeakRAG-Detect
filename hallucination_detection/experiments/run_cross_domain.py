"""
Cross-domain transfer experiment.

Trains on the legal domain (LegalBench-RAG) with weak supervision and
bootstrapping, then evaluates zero-shot transfer to the medical domain
(PubMedQA) and scientific domain (SciQ).

This tests whether hallucination patterns learned from legal text generalize
to other domains without any domain-specific labeled data.
"""

import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "..", "results", "experiment_log.txt"),
            mode="a",
        ),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SEEDS = 30


def main():
    """Run cross-domain transfer evaluation."""
    print("=" * 70)
    print("CROSS-DOMAIN TRANSFER EXPERIMENT")
    print("Train: Legal (LegalBench-RAG)  →  Test: Medical + Scientific")
    print("=" * 70)

    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    from evaluation.metrics import HallucinationEvaluator
    evaluator = HallucinationEvaluator()

    # ------------------------------------------------------------------
    # Step 1: Build source model on Legal domain
    # ------------------------------------------------------------------
    logger.info("Step 1: Building source model on Legal domain …")
    from data.load_datasets import load_legalbench_rag, load_pubmedqa, load_sciq
    from data.load_datasets import get_train_test_split, build_seed_set
    from labeling_functions.label_model import WeakLabelPipeline
    from bootstrapping.self_training import SelfTrainer
    from models.hallucination_classifier import LogisticRegressionClassifier

    legal_data = load_legalbench_rag()
    if not legal_data:
        logger.warning("LegalBench-RAG unavailable — using synthetic legal data.")
        from experiments.run_legal import _make_synthetic_legal_data
        legal_data = _make_synthetic_legal_data(n=500)

    legal_gold_test, legal_pool = get_train_test_split(legal_data, test_size=200, seed=RANDOM_SEED)
    legal_seeds, legal_pool = build_seed_set(legal_pool, n_seeds=N_SEEDS, domain="legal")

    logger.info("Running weak labeling on legal domain …")
    pipeline = WeakLabelPipeline()
    L_legal, soft_labels_legal, _ = pipeline.run(legal_pool)

    logger.info("Self-training on legal domain …")
    trainer = SelfTrainer(base_classifier=LogisticRegressionClassifier(), max_iterations=5)
    legal_result = trainer.fit(legal_seeds, legal_pool, soft_labels_legal, legal_gold_test[:50])
    source_model = legal_result["final_model"]

    # Evaluate source on legal (sanity check)
    legal_preds = source_model.predict(legal_gold_test)
    legal_labels = [ex["label"] for ex in legal_gold_test]
    evaluator.evaluate(legal_preds.tolist(), legal_labels, "Source (Legal → Legal)", "legal")

    # ------------------------------------------------------------------
    # Step 2: Zero-shot transfer to Medical
    # ------------------------------------------------------------------
    logger.info("Step 2: Zero-shot transfer to Medical (PubMedQA) …")
    medical_data = load_pubmedqa()
    if not medical_data:
        logger.warning("PubMedQA unavailable — using synthetic medical data.")
        from experiments.run_medical import _make_synthetic_medical_data
        medical_data = _make_synthetic_medical_data(n=400)

    med_gold_test, _ = get_train_test_split(medical_data, test_size=min(200, len(medical_data) // 3))
    med_labels = [ex["label"] for ex in med_gold_test]

    # Zero-shot: use legal-trained model directly on medical data
    med_preds_zero = source_model.predict(med_gold_test)
    evaluator.evaluate(med_preds_zero.tolist(), med_labels,
                       "Legal→Medical (zero-shot)", "medical")

    # Few-shot adaptation: fine-tune on 30 medical seeds
    med_gold_test_ft, med_pool = get_train_test_split(medical_data, test_size=min(200, len(medical_data) // 3))
    med_seeds, med_pool = build_seed_set(med_pool, n_seeds=N_SEEDS, domain="medical_fewshot")
    if med_seeds:
        ft_clf = LogisticRegressionClassifier()
        ft_clf.fit(
            legal_result["labeled_pool"] + med_seeds,
            legal_result["labels"] + [ex["label"] for ex in med_seeds],
        )
        med_preds_ft = ft_clf.predict(med_gold_test_ft)
        med_labels_ft = [ex["label"] for ex in med_gold_test_ft]
        evaluator.evaluate(med_preds_ft.tolist(), med_labels_ft,
                           f"Legal→Medical (few-shot n={N_SEEDS})", "medical")

    # ------------------------------------------------------------------
    # Step 3: Zero-shot transfer to Scientific
    # ------------------------------------------------------------------
    logger.info("Step 3: Zero-shot transfer to Scientific (SciQ) …")
    sci_data = load_sciq()
    if not sci_data:
        logger.warning("SciQ unavailable — using synthetic scientific data.")
        sci_data = _make_synthetic_scientific_data(n=400)

    sci_gold_test, _ = get_train_test_split(sci_data, test_size=200)
    sci_labels = [ex["label"] for ex in sci_gold_test]

    sci_preds_zero = source_model.predict(sci_gold_test)
    evaluator.evaluate(sci_preds_zero.tolist(), sci_labels,
                       "Legal→Scientific (zero-shot)", "scientific")

    # Few-shot adaptation: fine-tune on 30 scientific seeds
    sci_gold_test_ft, sci_pool = get_train_test_split(sci_data, test_size=200)
    sci_seeds, _ = build_seed_set(sci_pool, n_seeds=N_SEEDS, domain="scientific_fewshot")
    if sci_seeds:
        ft_clf2 = LogisticRegressionClassifier()
        ft_clf2.fit(
            legal_result["labeled_pool"] + sci_seeds,
            legal_result["labels"] + [ex["label"] for ex in sci_seeds],
        )
        sci_preds_ft = ft_clf2.predict(sci_gold_test_ft)
        sci_labels_ft = [ex["label"] for ex in sci_gold_test_ft]
        evaluator.evaluate(sci_preds_ft.tolist(), sci_labels_ft,
                           f"Legal→Scientific (few-shot n={N_SEEDS})", "scientific")

    # ------------------------------------------------------------------
    # Step 4: Cross-domain pattern comparison
    # ------------------------------------------------------------------
    logger.info("Step 4: Cross-domain n-gram pattern comparison …")
    try:
        from pattern_mining.ngram_analysis import HallucinationPatternMiner
        miner = HallucinationPatternMiner()

        domain_results = {}
        for domain, test_set in [
            ("legal", legal_gold_test),
            ("medical", med_gold_test),
            ("scientific", sci_gold_test),
        ]:
            hallucinated = [ex for ex in test_set if ex.get("label") == 1]
            faithful = [ex for ex in test_set if ex.get("label") == 0]
            if hallucinated and faithful:
                result = miner.extract_ngrams(
                    [ex["answer"] for ex in hallucinated],
                    [ex["answer"] for ex in faithful],
                    domain=domain,
                )
                domain_results[domain] = result

        if len(domain_results) > 1:
            miner.compare_domains(domain_results)
    except Exception as exc:
        logger.warning("Cross-domain pattern mining skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 5: Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN TRANSFER SUMMARY")
    print("=" * 70)
    print("Comparing per-domain results …")
    for domain in ("legal", "medical", "scientific"):
        try:
            evaluator.compare_methods(domain)
        except Exception:
            pass

    print(f"\nResults saved to {RESULTS_DIR}")


def _make_synthetic_scientific_data(n: int = 400) -> list:
    """Generate synthetic SciQ-style data for smoke testing."""
    import random
    random.seed(42)
    templates = [
        ("What causes photosynthesis?", "Plants use sunlight, water and CO2 to produce glucose.", "Sunlight, water and CO2.", 0),
        ("What is the boiling point of water?", "Water boils at 100°C at standard pressure.", "Water boils at 200°C.", 1),
        ("How does gravity work?", "Gravity attracts objects with mass towards each other.", "Gravity repels objects with mass.", 1),
        ("What is DNA?", "DNA is a double-helix molecule encoding genetic information.", "DNA encodes genetic information.", 0),
    ]
    data = []
    for i in range(n):
        q, c, a, lbl = random.choice(templates)
        data.append({
            "id": f"scientific_{i}", "question": q, "context": c,
            "answer": a, "label": lbl, "domain": "scientific", "source": "synthetic",
        })
    return data


if __name__ == "__main__":
    main()
