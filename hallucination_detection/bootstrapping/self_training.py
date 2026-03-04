"""
Iterative self-training bootstrapper.

Starts from seeds + Snorkel pseudo-labels, then expands the training set
using high-confidence model predictions on the unlabeled pool.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SelfTrainer:
    """Iterative self-training that expands labeled data from a small seed set.

    Algorithm
    ---------
    1. Iteration 0: train on seeds + Snorkel high-confidence pseudo-labels.
    2. For each subsequent iteration:
       a. Predict on remaining unlabeled pool.
       b. Select examples where P(hallucinated) > threshold OR
          P(hallucinated) < (1 - threshold).
       c. Add selected examples to the training set with hard labels.
       d. Retrain the classifier.
       e. Evaluate on val_data, log F1.
       f. Stop if F1 plateaus or no new high-confidence examples are found.

    Parameters
    ----------
    base_classifier
        Classifier with fit(examples, labels), predict(examples),
        and predict_proba(examples) methods.
    confidence_threshold : float
        Probability threshold for selecting high-confidence examples.
    max_iterations : int
        Maximum number of self-training iterations.
    min_new_examples : int
        Stop early if fewer than this many examples are added in one iteration.
    """

    def __init__(
        self,
        base_classifier,
        confidence_threshold: float = 0.85,
        max_iterations: int = 5,
        min_new_examples: int = 10,
    ):
        """Initialise SelfTrainer."""
        self.base_classifier = base_classifier
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.min_new_examples = min_new_examples

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        seed_data: List[dict],
        unlabeled_data: List[dict],
        soft_labels: np.ndarray,
        val_data: List[dict],
    ) -> dict:
        """Run iterative self-training.

        Parameters
        ----------
        seed_data : List[dict]
            Manually labeled seed examples (gold labels in ["label"]).
        unlabeled_data : List[dict]
            Unlabeled pool — soft_labels[i] corresponds to unlabeled_data[i].
        soft_labels : np.ndarray
            P(hallucinated) from Snorkel, shape (len(unlabeled_data),).
        val_data : List[dict]
            Validation examples with gold labels for early stopping.

        Returns
        -------
        dict with keys:
            "final_model"      – trained classifier,
            "training_history" – list of per-iteration dicts,
            "labeled_pool"     – final labeled training examples,
            "labels"           – corresponding integer labels.
        """
        from sklearn.metrics import f1_score  # local import to avoid hard dep at module level

        history = []
        remaining_unlabeled = list(unlabeled_data)
        remaining_soft = soft_labels.copy()

        # --- Iteration 0: seeds + high-confidence Snorkel labels ---
        labeled_pool, pool_labels = self._init_from_seeds_and_snorkel(
            seed_data, unlabeled_data, soft_labels
        )
        used_ids = {ex["id"] for ex in labeled_pool}
        remaining_unlabeled = [ex for ex in unlabeled_data if ex["id"] not in used_ids]
        remaining_soft = np.array([
            soft_labels[i]
            for i, ex in enumerate(unlabeled_data)
            if ex["id"] not in used_ids
        ])

        logger.info("Iteration 0: training on %d examples (seeds + Snorkel).", len(labeled_pool))
        clf = self._clone_classifier()
        clf.fit(labeled_pool, pool_labels)

        val_f1 = self._eval_f1(clf, val_data)
        history.append({"iteration": 0, "train_size": len(labeled_pool), "val_f1": val_f1})
        logger.info("  Val F1 = %.4f", val_f1)

        # --- Iterations 1..max_iterations ---
        prev_f1 = val_f1
        for iteration in range(1, self.max_iterations + 1):
            if not remaining_unlabeled:
                logger.info("No remaining unlabeled data. Stopping.")
                break

            # Predict on unlabeled pool
            probs = clf.predict_proba(remaining_unlabeled)
            high_conf_idx = self._select_high_confidence(probs, self.confidence_threshold)

            if len(high_conf_idx) < self.min_new_examples:
                logger.info(
                    "Iteration %d: only %d high-confidence examples found (min=%d). Stopping.",
                    iteration, len(high_conf_idx), self.min_new_examples,
                )
                break

            # Add to labeled pool
            new_examples = [remaining_unlabeled[i] for i in high_conf_idx]
            new_labels = (probs[high_conf_idx] > 0.5).astype(int).tolist()
            labeled_pool = labeled_pool + new_examples
            pool_labels = pool_labels + new_labels

            # Remove from unlabeled
            keep_mask = np.ones(len(remaining_unlabeled), dtype=bool)
            keep_mask[high_conf_idx] = False
            remaining_unlabeled = [ex for ex, keep in zip(remaining_unlabeled, keep_mask) if keep]
            remaining_soft = remaining_soft[keep_mask]

            # Retrain
            logger.info("Iteration %d: +%d examples → train_size=%d.", iteration, len(new_examples), len(labeled_pool))
            clf = self._clone_classifier()
            clf.fit(labeled_pool, pool_labels)

            val_f1 = self._eval_f1(clf, val_data)
            history.append({
                "iteration": iteration,
                "train_size": len(labeled_pool),
                "val_f1": val_f1,
                "newly_labeled": len(new_examples),
            })
            logger.info("  Val F1 = %.4f (Δ = %+.4f)", val_f1, val_f1 - prev_f1)

            # Plateau detection
            if abs(val_f1 - prev_f1) < 0.001:
                logger.info("F1 plateau detected. Stopping self-training.")
                break
            prev_f1 = val_f1

        return {
            "final_model": clf,
            "training_history": history,
            "labeled_pool": labeled_pool,
            "labels": pool_labels,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_from_seeds_and_snorkel(
        self,
        seed_data: List[dict],
        unlabeled_data: List[dict],
        soft_labels: np.ndarray,
    ):
        """Bootstrap labeled pool from seeds and high-confidence Snorkel labels."""
        seed_labels = [ex["label"] for ex in seed_data]
        labeled_pool = list(seed_data)
        pool_labels = list(seed_labels)

        for i, (ex, p) in enumerate(zip(unlabeled_data, soft_labels)):
            if p > self.confidence_threshold or p < (1 - self.confidence_threshold):
                labeled_pool.append(ex)
                pool_labels.append(int(p > 0.5))

        return labeled_pool, pool_labels

    def _select_high_confidence(
        self, probs: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Return indices of examples with P > threshold OR P < (1 - threshold)."""
        return np.where((probs > threshold) | (probs < (1 - threshold)))[0]

    def _eval_f1(self, clf, val_data: List[dict]) -> float:
        """Compute F1 on validation data."""
        from sklearn.metrics import f1_score
        if not val_data:
            return 0.0
        try:
            preds = clf.predict(val_data)
            golds = [ex["label"] for ex in val_data if ex.get("label") is not None]
            if not golds:
                return 0.0
            return float(f1_score(golds, preds[: len(golds)], zero_division=0))
        except Exception as exc:
            logger.warning("F1 evaluation failed: %s", exc)
            return 0.0

    def _clone_classifier(self):
        """Return a fresh instance of the base classifier."""
        try:
            import copy
            return copy.deepcopy(self.base_classifier)
        except Exception:
            return self.base_classifier.__class__()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("SelfTrainer loaded successfully.")
