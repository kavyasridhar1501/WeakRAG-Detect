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
    # Remove stale CSV log so compare_methods only shows this run's results.
    _stale_log = os.path.join(RESULTS_DIR, "evaluation_log.csv")
    if os.path.exists(_stale_log):
        os.remove(_stale_log)
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
    # Step 6b: DistilBERT fine-tuning on self-training labeled pool
    # ------------------------------------------------------------------
    logger.info("Step 6b: DistilBERT fine-tuning …")
    try:
        from models.hallucination_classifier import DistilBERTClassifier
        distilbert = DistilBERTClassifier()
        distilbert.train(
            st_result["labeled_pool"],
            st_result["labels"],
            val_data=gold_test[:50],
            val_labels=[d["label"] for d in gold_test[:50]],
            save_dir=os.path.join(RESULTS_DIR, "distilbert_ckpt"),
        )
        distilbert_preds = distilbert.predict(gold_test)
        evaluator.evaluate(distilbert_preds.tolist(), gold_labels,
                           "Self-Training + DistilBERT", "medical")
    except Exception as exc:
        logger.warning("DistilBERT training skipped: %s", exc)

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
    # Step 8: Label efficiency curve (opt-in — set RUN_LABEL_EFFICIENCY=1)
    # ------------------------------------------------------------------
    import os as _os
    if _os.environ.get("RUN_LABEL_EFFICIENCY", "0") == "1":
        try:
            from evaluation.label_efficiency import plot_label_efficiency_curve
            plot_label_efficiency_curve("medical", gold_test, unlabeled_pool, data, save_dir=RESULTS_DIR)
        except Exception as exc:
            logger.warning("Label efficiency curve skipped: %s", exc)
    else:
        logger.info("Label efficiency curve skipped (set RUN_LABEL_EFFICIENCY=1 to run).")

    evaluator.compare_methods("medical")
    print(f"\nResults saved to {RESULTS_DIR}")


def _make_synthetic_medical_data(n: int = 400) -> list:
    """Generate synthetic PubMedQA-style data for smoke testing.

    Uses 8 question templates × 6 drug/condition variants × 3 answer variants
    to avoid trivial memorisation by logistic regression baselines.
    Hallucinated answers introduce wrong numbers, wrong mechanisms, or wrong
    outcomes that are not supported by the provided context.
    """
    import random
    random.seed(42)

    # (question_template, context_template, faithful_answers, hallucinated_answers)
    # Placeholders: {drug}, {cond}, {pct}, {dose}, {weeks}, {side}
    templates = [
        (
            "Does {drug} reduce {cond}?",
            "A randomised trial showed {drug} reduced {cond} by {pct}% after {weeks} weeks.",
            [
                "Yes, {drug} reduces {cond} by {pct}%.",
                "{drug} significantly reduced {cond} ({pct}% reduction in {weeks} weeks).",
                "The trial confirmed {drug} is effective against {cond}.",
            ],
            [
                "{drug} had no effect on {cond} according to the study.",
                "Studies suggest {drug} worsens {cond} in most patients.",
                "The trial found {drug} caused a {pct}% increase in {cond}.",
            ],
        ),
        (
            "What dose of {drug} is recommended for {cond}?",
            "Guidelines recommend {dose}mg of {drug} daily for treatment of {cond}.",
            [
                "The recommended dose of {drug} for {cond} is {dose}mg daily.",
                "{dose}mg per day of {drug} is the standard recommendation for {cond}.",
                "Clinical guidelines specify {dose}mg/day {drug} for {cond}.",
            ],
            [
                "The recommended dose is {pct}mg of {drug} twice daily.",
                "Patients with {cond} should take {drug} at {dose}00mg monthly.",
                "No established dosing guideline exists for {drug} in {cond}.",
            ],
        ),
        (
            "How long does {drug} treatment last for {cond}?",
            "Standard treatment duration for {cond} with {drug} is {weeks} weeks.",
            [
                "Treatment with {drug} for {cond} lasts {weeks} weeks.",
                "The standard course of {drug} for {cond} is {weeks} weeks.",
                "{drug} is administered for {weeks} weeks in {cond} patients.",
            ],
            [
                "{drug} is typically administered indefinitely for {cond}.",
                "Treatment duration is {pct} months regardless of response.",
                "Patients take {drug} for 2 days and then discontinue.",
            ],
        ),
        (
            "What are the side effects of {drug}?",
            "Common side effects of {drug} include {side} and mild headache.",
            [
                "{drug} commonly causes {side} and mild headache.",
                "The main side effects of {drug} are {side} and headache.",
                "Patients on {drug} may experience {side} or mild headaches.",
            ],
            [
                "{drug} has no known side effects in clinical trials.",
                "The primary side effect of {drug} is severe liver failure.",
                "{drug} causes {side} exclusively in elderly patients.",
            ],
        ),
        (
            "Is {drug} effective for treating {cond}?",
            "Meta-analysis of 12 trials confirms {drug} efficacy for {cond} (p<0.01).",
            [
                "Yes, meta-analysis confirms {drug} is effective for {cond}.",
                "{drug} shows statistically significant efficacy against {cond}.",
                "Evidence from 12 trials supports {drug} use in {cond}.",
            ],
            [
                "Evidence for {drug} in {cond} is inconclusive at this time.",
                "{drug} was shown to be inferior to placebo in {cond} treatment.",
                "Only one small study supports {drug} for {cond}.",
            ],
        ),
        (
            "What mechanism explains {drug} action in {cond}?",
            "{drug} treats {cond} by inhibiting the pro-inflammatory pathway.",
            [
                "{drug} works by inhibiting the pro-inflammatory pathway in {cond}.",
                "The mechanism involves pro-inflammatory pathway inhibition by {drug}.",
                "{drug} suppresses {cond} through anti-inflammatory inhibition.",
            ],
            [
                "{drug} treats {cond} by stimulating serotonin receptors.",
                "The drug activates T-cell proliferation to address {cond}.",
                "{drug} reduces {cond} via calcium channel blockade.",
            ],
        ),
        (
            "Can {drug} be combined with other treatments for {cond}?",
            "Combination of {drug} with standard therapy improved outcomes in {cond} by {pct}%.",
            [
                "{drug} combined with standard therapy improved {cond} outcomes by {pct}%.",
                "Adding {drug} to standard care boosted efficacy by {pct}% in {cond}.",
                "Combination therapy including {drug} outperforms monotherapy in {cond}.",
            ],
            [
                "{drug} must not be combined with any other {cond} treatment.",
                "Combination with {drug} reduces efficacy of standard {cond} therapy.",
                "No studies have assessed {drug} in combination for {cond}.",
            ],
        ),
        (
            "What is the relapse rate after {drug} therapy for {cond}?",
            "Relapse rate after {drug} therapy for {cond} is {pct}% at {weeks} weeks.",
            [
                "The relapse rate is {pct}% at {weeks} weeks after {drug} therapy.",
                "After {drug} treatment, {pct}% of {cond} patients relapse by week {weeks}.",
                "{drug} therapy achieves {pct}% sustained remission at {weeks} weeks.",
            ],
            [
                "Patients treated with {drug} for {cond} never relapse.",
                "Relapse is universal ({pct}0%) within days of stopping {drug}.",
                "{drug} provides permanent cure for {cond} with zero relapse.",
            ],
        ),
    ]

    drugs = ["metformin", "atorvastatin", "lisinopril", "amoxicillin", "ibuprofen", "sertraline"]
    conditions = ["hypertension", "type-2 diabetes", "chronic pain", "depression", "bacterial infection", "hyperlipidaemia"]
    pcts = [12, 24, 38, 47, 61, 73]
    doses = [10, 25, 50, 100, 200, 500]
    weeks_list = [4, 8, 12, 16, 24, 52]
    sides = ["nausea", "dizziness", "fatigue", "dry mouth", "insomnia", "rash"]

    data = []
    idx = 0
    for drug, cond, pct, dose, weeks, side in zip(drugs, conditions, pcts, doses, weeks_list, sides):
        for t_idx, (q_t, c_t, faith_ans, hall_ans) in enumerate(templates):
            ctx = c_t.format(drug=drug, cond=cond, pct=pct, dose=dose, weeks=weeks, side=side)
            q = q_t.format(drug=drug, cond=cond)
            for ans_t in faith_ans:
                ans = ans_t.format(drug=drug, cond=cond, pct=pct, dose=dose, weeks=weeks, side=side)
                data.append({
                    "id": f"medical_{idx}", "question": q, "context": ctx,
                    "answer": ans, "label": 0, "domain": "medical", "source": "synthetic",
                })
                idx += 1
            for ans_t in hall_ans:
                ans = ans_t.format(drug=drug, cond=cond, pct=pct, dose=dose, weeks=weeks, side=side)
                data.append({
                    "id": f"medical_{idx}", "question": q, "context": ctx,
                    "answer": ans, "label": 1, "domain": "medical", "source": "synthetic",
                })
                idx += 1

    random.shuffle(data)
    # Trim or pad to requested n
    while len(data) < n:
        data.extend(data[: n - len(data)])
    data = data[:n]
    # Re-index IDs after shuffle/trim
    for i, ex in enumerate(data):
        ex["id"] = f"medical_{i}"
    return data


if __name__ == "__main__":
    main()
