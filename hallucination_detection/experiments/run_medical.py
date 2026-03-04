"""
Medical domain experiment — PubMedQA.

Tests weak-supervision + bootstrapping on the medical domain.
Transfer test: model trained with weak labels from PubMedQA.
"""

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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "medical")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SEEDS = 30


def main():
    """Run the full medical domain experiment pipeline."""
    print("=" * 60)
    print("MEDICAL DOMAIN EXPERIMENT (PubMedQA)")
    print("=" * 60)

    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading PubMedQA …")
    from data.load_datasets import load_pubmedqa, get_train_test_split, build_seed_set
    data = load_pubmedqa()

    if not data:
        logger.warning("PubMedQA unavailable. Generating synthetic data …")
        data = _make_synthetic_medical_data(n=400)

    gold_test, unlabeled_pool = get_train_test_split(data, test_size=min(200, len(data) // 3), seed=RANDOM_SEED)
    seed_data, unlabeled_pool = build_seed_set(
        unlabeled_pool, n_seeds=N_SEEDS, domain="medical", strategy="stratified"
    )
    gold_labels = [ex["label"] for ex in gold_test]
    logger.info("Gold test: %d | Seed data: %d | Unlabeled pool: %d",
                len(gold_test), len(seed_data), len(unlabeled_pool))

    # ------------------------------------------------------------------
    # Step 2: Weak labeling pipeline
    # ------------------------------------------------------------------
    logger.info("Step 2: Running weak labeling pipeline …")
    from labeling_functions.label_model import WeakLabelPipeline
    pipeline = WeakLabelPipeline()
    L, soft_labels, label_model = pipeline.run(unlabeled_pool, save_path=RESULTS_DIR)
    pipeline.analyze_lf_coverage(L)

    from evaluation.metrics import HallucinationEvaluator
    evaluator = HallucinationEvaluator()
    from models.hallucination_classifier import LogisticRegressionClassifier

    # ------------------------------------------------------------------
    # Step 3: Baseline — seeds only
    # ------------------------------------------------------------------
    lr_clf = LogisticRegressionClassifier()
    lr_clf.fit(seed_data, [d["label"] for d in seed_data])
    seed_preds = lr_clf.predict(gold_test)
    evaluator.evaluate(seed_preds.tolist(), gold_labels, f"Seeds Only (n={N_SEEDS})", "medical")

    # ------------------------------------------------------------------
    # Step 4: Seeds + Snorkel
    # ------------------------------------------------------------------
    high_conf_mask = (soft_labels > 0.85) | (soft_labels < 0.15)
    high_conf_indices = np.where(high_conf_mask)[0]
    snorkel_train = seed_data + [unlabeled_pool[i] for i in high_conf_indices]
    snorkel_train_labels = [d["label"] for d in seed_data] + [
        int(soft_labels[i] > 0.5) for i in high_conf_indices
    ]
    lr_clf2 = LogisticRegressionClassifier()
    lr_clf2.fit(snorkel_train, snorkel_train_labels)
    snorkel_preds = lr_clf2.predict(gold_test)
    evaluator.evaluate(snorkel_preds.tolist(), gold_labels, "Seeds + Snorkel", "medical")

    # ------------------------------------------------------------------
    # Step 5: Self-training
    # ------------------------------------------------------------------
    from bootstrapping.self_training import SelfTrainer
    self_trainer = SelfTrainer(base_classifier=LogisticRegressionClassifier(), max_iterations=5)
    st_result = self_trainer.fit(seed_data, unlabeled_pool, soft_labels, gold_test[:50])
    self_preds = st_result["final_model"].predict(gold_test)
    evaluator.evaluate(self_preds.tolist(), gold_labels, "Self-Training", "medical")

    # ------------------------------------------------------------------
    # Step 6: Co-training
    # ------------------------------------------------------------------
    from bootstrapping.co_training import CoTrainer
    co_trainer = CoTrainer(max_iterations=5)
    co_result = co_trainer.fit(seed_data, unlabeled_pool, soft_labels, gold_test[:50])
    co_preds = co_result["final_model"].predict(gold_test)
    evaluator.evaluate(co_preds.tolist(), gold_labels, "Co-Training", "medical")

    # ------------------------------------------------------------------
    # Step 7: Pattern mining
    # ------------------------------------------------------------------
    try:
        from pattern_mining.ngram_analysis import HallucinationPatternMiner
        from pattern_mining.entity_clustering import HallucinationTaxonomyBuilder
        hallucinated = [ex for ex in gold_test if ex.get("label") == 1]
        faithful = [ex for ex in gold_test if ex.get("label") == 0]
        if hallucinated and faithful:
            miner = HallucinationPatternMiner()
            miner.extract_ngrams([ex["answer"] for ex in hallucinated],
                                  [ex["answer"] for ex in faithful], domain="medical")
            taxonomy_builder = HallucinationTaxonomyBuilder()
            entities = taxonomy_builder.extract_hallucinated_entities(hallucinated)
            if entities:
                clustered = taxonomy_builder.cluster_entities(entities)
                taxonomy_builder.build_taxonomy("medical", clustered)
    except Exception as exc:
        logger.warning("Pattern mining skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 8: Label efficiency curve
    # ------------------------------------------------------------------
    try:
        from evaluation.label_efficiency import plot_label_efficiency_curve
        plot_label_efficiency_curve("medical", gold_test, unlabeled_pool, data, save_dir=RESULTS_DIR)
    except Exception as exc:
        logger.warning("Label efficiency curve skipped: %s", exc)

    evaluator.compare_methods("medical")
    print(f"\nResults saved to {RESULTS_DIR}")


def _make_synthetic_medical_data(n: int = 400) -> list:
    """Generate synthetic PubMedQA-style data for smoke testing."""
    import random
    random.seed(42)
    templates = [
        ("Does drug A reduce blood pressure?", "Study shows drug A reduces blood pressure by 10mmHg.", "Yes, drug A reduces blood pressure.", 0),
        ("Is there evidence for treatment B?", "No significant effect of treatment B was observed.", "I believe treatment B is effective.", 1),
        ("What is the efficacy of vaccine C?", "Vaccine C showed 92% efficacy in trials.", "Vaccine C is 92% effective.", 0),
    ]
    data = []
    for i in range(n):
        q, c, a, lbl = random.choice(templates)
        data.append({
            "id": f"medical_{i}", "question": q, "context": c,
            "answer": a, "label": lbl, "domain": "medical", "source": "synthetic",
        })
    return data


if __name__ == "__main__":
    main()
