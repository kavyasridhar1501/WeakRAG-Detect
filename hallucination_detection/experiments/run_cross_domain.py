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
    # Remove stale CSV logs so each domain's compare_methods only shows this
    # run's results — mirrors the cleanup done in run_legal.py (Step 4) and
    # run_medical.py before the first evaluator.evaluate call.
    for _domain in ("legal", "medical", "scientific"):
        _stale_log = os.path.join(RESULTS_DIR, _domain, "evaluation_log.csv")
        if os.path.exists(_stale_log):
            os.remove(_stale_log)
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
    """Generate synthetic SciQ-style data for smoke testing.

    Uses 8 question templates × 6 subject domains × 3 answer variants to
    avoid trivial memorisation.  Hallucinated answers introduce physically or
    chemically incorrect facts.
    """
    import random
    random.seed(42)

    # (question_template, context_template, faithful_answers, hallucinated_answers)
    # Placeholders: {subj}, {val}, {unit}, {mechanism}, {product}
    templates = [
        (
            "What is the {subj} of {product}?",
            "The {subj} of {product} is {val} {unit} under standard conditions.",
            [
                "The {subj} of {product} is {val} {unit}.",
                "{product} has a {subj} of {val} {unit} at standard conditions.",
                "Under standard conditions, {product}'s {subj} measures {val} {unit}.",
            ],
            [
                "The {subj} of {product} is {val}00 {unit}.",
                "{product} does not have a measurable {subj}.",
                "The {subj} of {product} varies wildly and cannot be specified.",
            ],
        ),
        (
            "How does {mechanism} work in {product}?",
            "{product} relies on {mechanism} to convert energy into usable forms.",
            [
                "{product} uses {mechanism} to convert energy.",
                "Energy conversion in {product} occurs through {mechanism}.",
                "{mechanism} is the primary process by which {product} functions.",
            ],
            [
                "{product} functions without any {mechanism}.",
                "{mechanism} is irrelevant to how {product} operates.",
                "{product} uses the inverse of {mechanism} for energy conversion.",
            ],
        ),
        (
            "What is produced when {product} undergoes {mechanism}?",
            "When {product} undergoes {mechanism}, it produces {val} moles of energy and water.",
            [
                "{mechanism} of {product} produces energy and water.",
                "The products of {mechanism} in {product} are energy and water.",
                "{product} yields energy and water through {mechanism}.",
            ],
            [
                "{mechanism} of {product} produces pure nitrogen gas.",
                "Only carbon dioxide is produced when {product} undergoes {mechanism}.",
                "{product} disintegrates completely during {mechanism} leaving no product.",
            ],
        ),
        (
            "What is the unit of {subj}?",
            "The SI unit of {subj} is the {unit}, defined as {val} base units.",
            [
                "The SI unit of {subj} is the {unit}.",
                "{subj} is measured in {unit} (SI standard).",
                "Scientists measure {subj} using the {unit} as the base unit.",
            ],
            [
                "The unit of {subj} is the kilogram regardless of quantity.",
                "{subj} has no standardised unit of measurement.",
                "{subj} is measured in candelas per square metre.",
            ],
        ),
        (
            "Why does {product} exhibit {mechanism}?",
            "{product} exhibits {mechanism} because of its molecular structure and {val} bonds.",
            [
                "{product} shows {mechanism} due to its molecular structure.",
                "The {val} bonds in {product} are responsible for {mechanism}.",
                "{mechanism} arises from the molecular structure of {product}.",
            ],
            [
                "{product} does not exhibit {mechanism} under any conditions.",
                "{mechanism} in {product} is caused by external magnetic fields.",
                "{product} exhibits {mechanism} only at absolute zero.",
            ],
        ),
        (
            "What temperature is required for {mechanism} in {product}?",
            "{mechanism} in {product} requires a temperature of {val} {unit}.",
            [
                "{mechanism} requires {val} {unit} in {product}.",
                "A temperature of {val} {unit} initiates {mechanism} in {product}.",
                "{product} undergoes {mechanism} at {val} {unit}.",
            ],
            [
                "{mechanism} in {product} occurs at room temperature spontaneously.",
                "{product} requires {val}000 {unit} for {mechanism} to start.",
                "Temperature is irrelevant to {mechanism} in {product}.",
            ],
        ),
        (
            "How is {subj} measured in {product}?",
            "{subj} in {product} is measured using {mechanism} with {unit} precision.",
            [
                "{mechanism} measures {subj} in {product} with {unit} precision.",
                "Scientists use {mechanism} to determine {subj} in {product}.",
                "{subj} of {product} is quantified via {mechanism}.",
            ],
            [
                "{subj} in {product} cannot be measured by any known method.",
                "Measuring {subj} in {product} requires no specialised equipment.",
                "{subj} in {product} is inferred from colour alone.",
            ],
        ),
        (
            "What effect does {mechanism} have on {product}?",
            "Applying {mechanism} to {product} increases its {subj} by {val} {unit}.",
            [
                "{mechanism} increases {product}'s {subj} by {val} {unit}.",
                "The effect of {mechanism} on {product} is a {val} {unit} rise in {subj}.",
                "{product} gains {val} {unit} of {subj} through {mechanism}.",
            ],
            [
                "{mechanism} has no measurable effect on {product}.",
                "{mechanism} destroys {product} instantaneously.",
                "Applying {mechanism} reduces {product}'s {subj} to zero.",
            ],
        ),
    ]

    subjects = ["boiling point", "melting point", "density", "refractive index", "conductivity", "viscosity"]
    products = ["water", "ethanol", "iron", "sodium chloride", "glucose", "nitrogen gas"]
    vals = [100, 78, 1000, 1.33, 58, 0.018]
    units = ["°C", "°C", "kg/m³", "dimensionless", "g/mol", "Pa·s"]
    mechanisms = ["combustion", "electrolysis", "photosynthesis", "oxidation", "sublimation", "diffusion"]

    data = []
    idx = 0
    for subj, product, val, unit, mechanism in zip(subjects, products, vals, units, mechanisms):
        for q_t, c_t, faith_ans, hall_ans in templates:
            ctx = c_t.format(subj=subj, product=product, val=val, unit=unit, mechanism=mechanism)
            q = q_t.format(subj=subj, product=product, mechanism=mechanism)
            for ans_t in faith_ans:
                ans = ans_t.format(subj=subj, product=product, val=val, unit=unit, mechanism=mechanism)
                data.append({
                    "id": f"scientific_{idx}", "question": q, "context": ctx,
                    "answer": ans, "label": 0, "domain": "scientific", "source": "synthetic",
                })
                idx += 1
            for ans_t in hall_ans:
                ans = ans_t.format(subj=subj, product=product, val=val, unit=unit, mechanism=mechanism)
                data.append({
                    "id": f"scientific_{idx}", "question": q, "context": ctx,
                    "answer": ans, "label": 1, "domain": "scientific", "source": "synthetic",
                })
                idx += 1

    random.shuffle(data)
    while len(data) < n:
        data.extend(data[: n - len(data)])
    data = data[:n]
    for i, ex in enumerate(data):
        ex["id"] = f"scientific_{i}"
    return data


if __name__ == "__main__":
    main()
