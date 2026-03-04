"""
Evaluation metrics for hallucination detection.

Computes Precision, Recall, F1, Accuracy, and AUC-ROC.
Logs results to both console and CSV for comparison across methods.
"""

import csv
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LEGALINSIGHT_BASELINE: float = 0.7465  # 74.65% from prior system

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed — HallucinationEvaluator unavailable.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


class HallucinationEvaluator:
    """Compute and log hallucination detection metrics.

    Metrics computed:
        - Precision
        - Recall
        - F1 (binary, positive class = hallucinated)
        - Accuracy
        - AUC-ROC (if probabilities provided)
    """

    def evaluate(
        self,
        predictions: List[int],
        gold_labels: List[int],
        method_name: str,
        domain: str,
        probabilities: Optional[List[float]] = None,
    ) -> dict:
        """Evaluate predictions against gold labels and log results.

        Parameters
        ----------
        predictions : List[int]
            Hard predicted labels (0=faithful, 1=hallucinated).
        gold_labels : List[int]
            Gold-standard labels.
        method_name : str
            Descriptive name for the method (used in comparison table).
        domain : str
            Domain name (used for the output directory).
        probabilities : List[float] | None
            P(hallucinated) for AUC-ROC computation.

        Returns
        -------
        dict
            {method, domain, precision, recall, f1, accuracy, auc_roc}
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required.")

        preds = np.array(predictions)
        golds = np.array(gold_labels)

        # Align lengths
        n = min(len(preds), len(golds))
        preds, golds = preds[:n], golds[:n]

        precision = float(precision_score(golds, preds, zero_division=0))
        recall = float(recall_score(golds, preds, zero_division=0))
        f1 = float(f1_score(golds, preds, zero_division=0))
        accuracy = float(accuracy_score(golds, preds))

        auc_roc = None
        if probabilities is not None:
            try:
                auc_roc = float(roc_auc_score(golds, probabilities[:n]))
            except Exception as exc:
                logger.warning("AUC-ROC computation failed: %s", exc)

        metrics = {
            "method": method_name,
            "domain": domain,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "auc_roc": round(auc_roc, 4) if auc_roc is not None else "N/A",
            "n_examples": n,
        }

        self._print_metrics(metrics)
        self._append_to_log(metrics, domain)

        return metrics

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def compare_methods(self, domain: str) -> None:
        """Load all logged results and print a comparison table.

        Parameters
        ----------
        domain : str
        """
        log_path = self._log_path(domain)
        if not os.path.exists(log_path):
            logger.warning("No evaluation log found at %s", log_path)
            return

        rows = []
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print("No evaluation results found.")
            return

        # Print table
        header = f"\n{'Method':<35} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}"
        print("\n" + "=" * 75)
        print(f"{'Evaluation Results — ' + domain.capitalize():^75}")
        print("=" * 75)
        print(header)
        print("-" * 75)
        for row in rows:
            print(
                f"{row.get('method', ''):<35} "
                f"{row.get('precision', ''):>10} "
                f"{row.get('recall', ''):>8} "
                f"{row.get('f1', ''):>8} "
                f"{row.get('accuracy', ''):>10}"
            )
        print("=" * 75)

    # ------------------------------------------------------------------
    # Consistency score comparison (vs LegalInsight baseline)
    # ------------------------------------------------------------------

    def consistency_score_comparison(
        self,
        domain_results: Dict[str, List[float]],
    ) -> None:
        """Compare system consistency scores to LegalInsight baseline (74.65%).

        Parameters
        ----------
        domain_results : dict
            {domain: [consistency_scores]}
        """
        print("\n" + "=" * 60)
        print("Consistency Score Comparison vs LegalInsight (74.65%)")
        print("=" * 60)

        for domain, scores in domain_results.items():
            if not scores:
                continue
            mean_score = float(np.mean(scores))
            delta = mean_score - LEGALINSIGHT_BASELINE
            print(
                f"  {domain:<15} mean={mean_score:.4f}  "
                f"delta={delta:+.4f} vs baseline={LEGALINSIGHT_BASELINE:.4f}"
            )

        if _MPL_AVAILABLE and domain_results:
            self._plot_consistency_comparison(domain_results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_path(self, domain: str) -> str:
        out_dir = os.path.join(RESULTS_DIR, domain)
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, "evaluation_log.csv")

    def _append_to_log(self, metrics: dict, domain: str) -> None:
        log_path = self._log_path(domain)
        file_exists = os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def _print_metrics(self, metrics: dict) -> None:
        print(
            f"\n[{metrics['domain'].upper()}] {metrics['method']}\n"
            f"  Precision={metrics['precision']:.4f}  "
            f"Recall={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"Accuracy={metrics['accuracy']:.4f}  "
            f"AUC-ROC={metrics['auc_roc']}"
        )

    def _plot_consistency_comparison(self, domain_results: Dict[str, List[float]]) -> None:
        try:
            domains = list(domain_results.keys())
            means = [np.mean(domain_results[d]) for d in domains]

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#2ca02c" if m > LEGALINSIGHT_BASELINE else "#d62728" for m in means]
            ax.bar(domains, means, color=colors, alpha=0.8)
            ax.axhline(LEGALINSIGHT_BASELINE, color="navy", linestyle="--",
                       label=f"LegalInsight baseline ({LEGALINSIGHT_BASELINE:.2%})")
            ax.set_ylabel("Mean Consistency Score")
            ax.set_title("Consistency Score vs LegalInsight Baseline")
            ax.legend()
            ax.set_ylim(0, 1)

            out_path = os.path.join(RESULTS_DIR, "consistency_comparison.png")
            os.makedirs(RESULTS_DIR, exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            logger.info("Saved consistency comparison plot to %s", out_path)
        except Exception as exc:
            logger.warning("Failed to save consistency plot: %s", exc)
