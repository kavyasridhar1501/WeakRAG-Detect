"""
Label efficiency curve generation.

Measures how system performance (F1) scales with the number of seed labels,
demonstrating the benefit of weak supervision and bootstrapping vs. fully
supervised training.

This is the key result figure for the DSC 253 paper.
"""

import logging
import os
import random
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — label efficiency plots unavailable.")

try:
    from sklearn.metrics import f1_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def plot_label_efficiency_curve(
    domain: str,
    gold_test: List[dict],
    unlabeled_pool: List[dict],
    all_domain_data: List[dict],
    seed_sizes: Optional[List[int]] = None,
    save_dir: Optional[str] = None,
) -> dict:
    """Plot label efficiency curve for a given domain.

    For each seed size:
      1. Sample a seed set of that size from *unlabeled_pool*.
      2. Run weak labeling + self-training.
      3. Evaluate the final model on *gold_test*.
      4. Record F1.

    Also computes a fully supervised baseline (train on all gold_test).

    Parameters
    ----------
    domain : str
    gold_test : List[dict]
        200 gold-labeled examples held out for evaluation.
    unlabeled_pool : List[dict]
        Unlabeled examples for Snorkel + bootstrapping.
    all_domain_data : List[dict]
        All domain data (for fully supervised baseline).
    seed_sizes : List[int] | None
        Seed sizes to evaluate (default: [5, 10, 20, 30, 50, 100, 200]).
    save_dir : str | None
        Output directory (default: results/{domain}/).

    Returns
    -------
    dict
        {"seed_sizes": [...], "f1_scores": [...], "supervised_f1": float}
    """
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required.")

    seed_sizes = seed_sizes or [5, 10, 20, 30, 50, 100, 200]
    save_dir = save_dir or os.path.join(RESULTS_DIR, domain)
    os.makedirs(save_dir, exist_ok=True)

    gold_labels = [ex["label"] for ex in gold_test]

    # Lazy imports to avoid circular import at module load
    from hallucination_detection.labeling_functions.label_model import WeakLabelPipeline
    from hallucination_detection.bootstrapping.self_training import SelfTrainer
    from hallucination_detection.models.hallucination_classifier import LogisticRegressionClassifier

    f1_scores: List[float] = []

    for n_seeds in seed_sizes:
        logger.info("[%s] Label efficiency: n_seeds=%d …", domain, n_seeds)
        try:
            # Sample seed set
            labeled_pool = [ex for ex in unlabeled_pool if ex.get("label") is not None]
            random.seed(42)
            seeds = random.sample(labeled_pool, min(n_seeds, len(labeled_pool)))
            seed_ids = {ex["id"] for ex in seeds}
            remaining = [ex for ex in unlabeled_pool if ex["id"] not in seed_ids]

            # Weak labeling
            pipeline = WeakLabelPipeline()
            L, soft_labels, label_model = pipeline.run(remaining)

            # Self-training
            trainer = SelfTrainer(base_classifier=LogisticRegressionClassifier(), max_iterations=3)
            result = trainer.fit(seeds, remaining, soft_labels, gold_test[:50])
            clf = result["final_model"]

            # Evaluate
            preds = clf.predict(gold_test)
            f1 = float(f1_score(gold_labels, preds[:len(gold_labels)], zero_division=0))
            f1_scores.append(f1)
            logger.info("  n_seeds=%d → F1=%.4f", n_seeds, f1)

        except Exception as exc:
            logger.warning("Label efficiency failed for n_seeds=%d: %s", n_seeds, exc)
            f1_scores.append(0.0)

    # Fully supervised baseline (LR trained on all gold test labels)
    supervised_f1 = 0.0
    try:
        sup_clf = LogisticRegressionClassifier()
        sup_clf.fit(gold_test, gold_labels)
        sup_preds = sup_clf.predict(gold_test)
        supervised_f1 = float(f1_score(gold_labels, sup_preds, zero_division=0))
        logger.info("Fully supervised baseline F1: %.4f", supervised_f1)
    except Exception as exc:
        logger.warning("Supervised baseline failed: %s", exc)

    # Plot
    if _MPL_AVAILABLE:
        _plot_curve(domain, seed_sizes, f1_scores, supervised_f1, save_dir)

    # Save numeric results
    results = {
        "domain": domain,
        "seed_sizes": seed_sizes,
        "f1_scores": f1_scores,
        "supervised_f1": supervised_f1,
    }
    results_path = os.path.join(save_dir, "label_efficiency.json")
    import json
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved label efficiency results to %s", results_path)

    return results


def _plot_curve(
    domain: str,
    seed_sizes: List[int],
    f1_scores: List[float],
    supervised_f1: float,
    save_dir: str,
) -> None:
    """Render and save the label efficiency curve plot."""
    try:
        fig, ax = plt.subplots(figsize=(9, 6))

        ax.plot(seed_sizes, f1_scores, "o-", color="#1f77b4", linewidth=2,
                markersize=7, label="WeakRAG-Detect (self-training)")
        ax.axhline(supervised_f1, color="#d62728", linestyle="--", linewidth=1.5,
                   label=f"Fully supervised baseline (F1={supervised_f1:.3f})")

        # Shade the gap between bootstrapped and supervised
        ax.fill_between(
            seed_sizes,
            f1_scores,
            [supervised_f1] * len(seed_sizes),
            alpha=0.15,
            color="#d62728",
            label="Gap to fully supervised",
        )

        ax.set_xlabel("Number of Seed Labels", fontsize=13)
        ax.set_ylabel("F1 Score (hallucination detection)", fontsize=13)
        ax.set_title(f"Label Efficiency Curve — {domain.capitalize()}", fontsize=14)
        ax.legend(fontsize=11)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(save_dir, "label_efficiency_curve.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info("Saved label efficiency curve to %s", out_path)
    except Exception as exc:
        logger.warning("Failed to save label efficiency plot: %s", exc)
