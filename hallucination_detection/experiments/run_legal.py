"""
Legal domain experiment — LegalBench-RAG.

Builds on the LegalInsight baseline (74.65% consistency score) and evaluates
the full weak-supervision + bootstrapping pipeline on the legal domain.
"""

import json
import logging
import os
import sys

# Must be set before any tokenizer/torch import to prevent MPS fork segfault on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

# Allow package-level imports
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "legal")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SEEDS = 30
LEGALINSIGHT_BASELINE = 0.7465


def main():
    """Run the full legal domain experiment pipeline."""
    print("=" * 60)
    print("LEGAL DOMAIN EXPERIMENT (LegalBench-RAG)")
    print("Building on LegalInsight baseline (74.65% consistency)")
    print("=" * 60)

    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading LegalBench-RAG …")
    from data.load_datasets import load_legalbench_rag, get_train_test_split, build_seed_set
    data = load_legalbench_rag()

    if not data:
        logger.warning("LegalBench-RAG unavailable. Generating synthetic data for smoke test …")
        data = _make_synthetic_legal_data(n=500)

    gold_test, unlabeled_pool = get_train_test_split(data, test_size=200, seed=RANDOM_SEED)
    seed_data, unlabeled_pool = build_seed_set(
        unlabeled_pool, n_seeds=N_SEEDS, domain="legal", strategy="stratified"
    )
    gold_labels = [ex["label"] for ex in gold_test]

    # Reserve a small held-out val split from seed_data for bootstrapping callbacks.
    # Using gold_test for validation leaks test-time information into training.
    import math
    n_val = max(10, math.ceil(0.2 * len(seed_data)))
    val_data = seed_data[:n_val]
    seed_data = seed_data[n_val:]
    logger.info("Gold test: %d | Seed data: %d | Val: %d | Unlabeled pool: %d",
                len(gold_test), len(seed_data), len(val_data), len(unlabeled_pool))

    # ------------------------------------------------------------------
    # Step 2: Build FAISS retriever (port from LegalInsight)
    # ------------------------------------------------------------------
    logger.info("Step 2: Building FAISS retriever …")
    try:
        from data.retrieval import FAISSRetriever
        retriever = FAISSRetriever.build_or_load(
            documents=[d["context"] for d in data],
            ids=[d["id"] for d in data],
            domain="legal",
        )
        sample_results = retriever.retrieve("termination clause", k=3)
        logger.info("Retriever test: %d results for 'termination clause'", len(sample_results))
    except Exception as exc:
        logger.warning("FAISS retriever skipped (missing deps): %s", exc)

    # ------------------------------------------------------------------
    # Step 3: Weak labeling pipeline
    # ------------------------------------------------------------------
    logger.info("Step 3: Running weak labeling pipeline …")
    from labeling_functions.label_model import WeakLabelPipeline
    pipeline = WeakLabelPipeline()
    L, soft_labels, label_model = pipeline.run(
        unlabeled_pool, save_path=RESULTS_DIR
    )
    pipeline.analyze_lf_coverage(L)

    # ------------------------------------------------------------------
    # Step 4: Evaluator
    # ------------------------------------------------------------------
    from evaluation.metrics import HallucinationEvaluator
    evaluator = HallucinationEvaluator()

    # ------------------------------------------------------------------
    # Step 5: Baseline — seeds only
    # ------------------------------------------------------------------
    logger.info("Step 5: Baseline — seeds only (n=%d) …", N_SEEDS)
    from models.hallucination_classifier import LogisticRegressionClassifier
    lr_clf = LogisticRegressionClassifier()
    lr_clf.fit(seed_data, [d["label"] for d in seed_data])
    seed_preds = lr_clf.predict(gold_test)
    evaluator.evaluate(seed_preds.tolist(), gold_labels, f"Seeds Only (n={N_SEEDS})", "legal")

    # ------------------------------------------------------------------
    # Step 6: Seeds + Snorkel (no bootstrapping)
    # ------------------------------------------------------------------
    logger.info("Step 6: Seeds + Snorkel …")
    high_conf_mask = (soft_labels > 0.85) | (soft_labels < 0.15)
    high_conf_indices = np.where(high_conf_mask)[0]
    snorkel_train = seed_data + [unlabeled_pool[i] for i in high_conf_indices]
    snorkel_train_labels = [d["label"] for d in seed_data] + [
        int(soft_labels[i] > 0.5) for i in high_conf_indices
    ]
    lr_clf2 = LogisticRegressionClassifier()
    lr_clf2.fit(snorkel_train, snorkel_train_labels)
    snorkel_preds = lr_clf2.predict(gold_test)
    evaluator.evaluate(snorkel_preds.tolist(), gold_labels, "Seeds + Snorkel", "legal")

    # ------------------------------------------------------------------
    # Step 7: Self-training bootstrapping
    # ------------------------------------------------------------------
    logger.info("Step 7: Self-training bootstrapping …")
    from bootstrapping.self_training import SelfTrainer
    self_trainer = SelfTrainer(
        base_classifier=LogisticRegressionClassifier(),
        confidence_threshold=0.85,
        max_iterations=5,
    )
    st_result = self_trainer.fit(seed_data, unlabeled_pool, soft_labels, val_data)
    self_preds = st_result["final_model"].predict(gold_test)
    evaluator.evaluate(self_preds.tolist(), gold_labels, "Self-Training", "legal")

    # ------------------------------------------------------------------
    # Step 8: Co-training bootstrapping
    # ------------------------------------------------------------------
    logger.info("Step 8: Co-training bootstrapping …")
    from bootstrapping.co_training import CoTrainer
    co_trainer = CoTrainer(confidence_threshold=0.85, max_iterations=5)
    co_result = co_trainer.fit(seed_data, unlabeled_pool, soft_labels, val_data)
    co_preds = co_result["final_model"].predict(gold_test)
    evaluator.evaluate(co_preds.tolist(), gold_labels, "Co-Training", "legal")

    # ------------------------------------------------------------------
    # Step 9: DistilBERT on bootstrapped data
    # ------------------------------------------------------------------
    logger.info("Step 9: DistilBERT fine-tuning …")
    try:
        from models.hallucination_classifier import DistilBERTClassifier
        distilbert = DistilBERTClassifier()
        distilbert.train(
            st_result["labeled_pool"],
            st_result["labels"],
            val_data=val_data,
            val_labels=[d["label"] for d in val_data],
            save_dir=os.path.join(RESULTS_DIR, "distilbert_ckpt"),
        )
        distilbert_preds = distilbert.predict(gold_test)
        evaluator.evaluate(distilbert_preds.tolist(), gold_labels,
                           "Self-Training + DistilBERT", "legal")
    except Exception as exc:
        logger.warning("DistilBERT training skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 10: Fully supervised baseline (oracle upper bound)
    # Train on unlabeled_pool with true labels + seeds; test on gold_test.
    # ------------------------------------------------------------------
    logger.info("Step 10: Fully supervised baseline …")
    sup_train = seed_data + unlabeled_pool
    sup_train_labels = [d["label"] for d in seed_data] + [d["label"] for d in unlabeled_pool]
    sup_clf = LogisticRegressionClassifier()
    sup_clf.fit(sup_train, sup_train_labels)
    sup_preds = sup_clf.predict(gold_test)
    evaluator.evaluate(sup_preds.tolist(), gold_labels, "Fully Supervised", "legal")

    # ------------------------------------------------------------------
    # Step 11: Pattern mining
    # ------------------------------------------------------------------
    logger.info("Step 11: Pattern mining …")
    try:
        from pattern_mining.ngram_analysis import HallucinationPatternMiner
        from pattern_mining.entity_clustering import HallucinationTaxonomyBuilder

        hallucinated_examples = [ex for ex in gold_test if ex.get("label") == 1]
        faithful_examples = [ex for ex in gold_test if ex.get("label") == 0]
        hallucinated_answers = [ex["answer"] for ex in hallucinated_examples]
        faithful_answers = [ex["answer"] for ex in faithful_examples]

        if hallucinated_answers and faithful_answers:
            miner = HallucinationPatternMiner()
            miner.extract_ngrams(hallucinated_answers, faithful_answers, domain="legal")

            taxonomy_builder = HallucinationTaxonomyBuilder()
            entities = taxonomy_builder.extract_hallucinated_entities(hallucinated_examples)
            if entities:
                clustered = taxonomy_builder.cluster_entities(entities)
                taxonomy_builder.build_taxonomy("legal", clustered)
    except Exception as exc:
        logger.warning("Pattern mining skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 12: Label efficiency curve (opt-in — re-runs full NLI pipeline
    # 7 times; set RUN_LABEL_EFFICIENCY=1 in env to enable)
    # ------------------------------------------------------------------
    import os as _os
    if _os.environ.get("RUN_LABEL_EFFICIENCY", "0") == "1":
        logger.info("Step 12: Label efficiency curve …")
        try:
            from evaluation.label_efficiency import plot_label_efficiency_curve
            plot_label_efficiency_curve(
                "legal", gold_test, unlabeled_pool, data,
                seed_sizes=[5, 10, 20, 30, 50, 100, 200],
                save_dir=RESULTS_DIR,
            )
        except Exception as exc:
            logger.warning("Label efficiency curve skipped: %s", exc)
    else:
        logger.info("Step 12: Label efficiency curve skipped (set RUN_LABEL_EFFICIENCY=1 to run).")

    # ------------------------------------------------------------------
    # Step 13: Consistency score comparison to LegalInsight
    # ------------------------------------------------------------------
    logger.info("Step 13: Consistency score comparison …")
    try:
        from labeling_functions.semantic_consistency_lf import SemanticConsistencyLF
        clf_lf = SemanticConsistencyLF()
        sample = gold_test[:50]
        consistency_results = clf_lf.label_batch(sample)
        scores = [r[1] for r in consistency_results]
        report = clf_lf.report_vs_baseline(scores)
        new_consistency = report.get("mean_consistency", 0.0)
    except Exception as exc:
        logger.warning("Consistency comparison failed: %s", exc)
        new_consistency = 0.0

    consistency_summary = {
        "legalinsight_baseline": LEGALINSIGHT_BASELINE,
        "new_system_consistency": round(new_consistency, 4),
        "improvement": round(new_consistency - LEGALINSIGHT_BASELINE, 4),
    }
    summary_path = os.path.join(RESULTS_DIR, "consistency_summary.json")
    with open(summary_path, "w") as f:
        json.dump(consistency_summary, f, indent=2)
    logger.info("Saved consistency summary to %s", summary_path)

    print(f"\n{'=' * 60}")
    print(f"LegalInsight baseline consistency : {LEGALINSIGHT_BASELINE:.2%}")
    print(f"New system consistency            : {new_consistency:.2%}")
    print(f"Improvement                       : {new_consistency - LEGALINSIGHT_BASELINE:+.2%}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Final comparison table
    # ------------------------------------------------------------------
    evaluator.compare_methods("legal")
    print(f"\nResults saved to {RESULTS_DIR}")


def _make_synthetic_legal_data(n: int = 500) -> list:
    """Generate synthetic LegalBench-style data for smoke testing.

    Hallucinated examples contain factually wrong information (wrong parties,
    amounts, or dates) rather than just hedging prefixes, so that the NLI and
    entity-overlap labeling functions can actually detect them.
    """
    import random
    random.seed(42)
    templates = [
        ("Who are the parties?", "This agreement is between {a} and {b}.", "{a} and {b}."),
        ("When does the contract terminate?", "The contract terminates on {date}.", "It terminates on {date}."),
        ("What is the payment amount?", "Payment of ${amount} is due within {days} days.", "${amount} due in {days} days."),
    ]
    parties = [("Acme Corp", "Beta Ltd"), ("Delta Inc", "Gamma LLC"), ("Alpha Co", "Omega Ltd")]
    wrong_dates = ["Mar 1, 2023", "Jun 15, 2024", "Sep 30, 2022"]
    wrong_amounts = ["1000", "25000", "750"]
    wrong_days = ["90", "7", "45"]
    data = []
    for i in range(n):
        tmpl_idx = i % len(templates)
        q_tmpl, c_tmpl, a_tmpl = templates[tmpl_idx]
        pa, pb = parties[i % len(parties)]
        ctx = c_tmpl.format(a=pa, b=pb, date="Dec 31, 2025", amount="5000", days="30")
        label = 0 if i % 3 != 0 else 1
        if label == 0:
            # Faithful: answer matches context exactly
            ans = a_tmpl.format(a=pa, b=pb, date="Dec 31, 2025", amount="5000", days="30")
        else:
            # Hallucinated: use wrong facts that contradict the context
            wrong_pa, wrong_pb = parties[(i // len(parties) + 1) % len(parties)]
            ans = a_tmpl.format(
                a=wrong_pa,
                b=wrong_pb,
                date=wrong_dates[i % len(wrong_dates)],
                amount=wrong_amounts[i % len(wrong_amounts)],
                days=wrong_days[i % len(wrong_days)],
            )
        data.append({
            "id": f"legal_{i}", "question": q_tmpl, "context": ctx,
            "answer": ans, "label": label, "domain": "legal", "source": "synthetic",
        })
    return data


if __name__ == "__main__":
    main()
