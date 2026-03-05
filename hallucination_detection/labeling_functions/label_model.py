"""
Snorkel-based weak label pipeline.

Combines all three labeling functions using Snorkel's generative label model
to learn LF accuracies and correlations, then outputs probabilistic training
labels for unlabeled data.
"""

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Snorkel import
# ---------------------------------------------------------------------------
try:
    from snorkel.labeling import LabelModel as SnorkelLabelModel  # type: ignore
    _SNORKEL_AVAILABLE = True
except ImportError:
    _SNORKEL_AVAILABLE = False
    logger.warning(
        "snorkel-metal not installed.  WeakLabelPipeline will fall back "
        "to majority vote instead of the generative label model."
    )

from .entailment_lf import EntailmentLF
from .semantic_consistency_lf import SemanticConsistencyLF
from .reflection_token_lf import ReflectionTokenLF

ABSTAIN = -1
FAITHFUL = 0
HALLUCINATED = 1


class WeakLabelPipeline:
    """Combines all 3 LFs using Snorkel's generative label model.

    The label model learns each LF's accuracy and inter-LF correlations
    from the unlabeled data and outputs probabilistic training labels.

    Falls back to majority voting when snorkel-metal is not installed.
    """

    LF_NAMES = ["entailment_lf", "consistency_lf", "reflection_lf"]

    def __init__(self):
        """Instantiate all three labeling functions."""
        logger.info("Initialising WeakLabelPipeline …")
        self.entailment_lf = EntailmentLF()
        # Lower thresholds for LLM-free answer-context alignment mode.
        # Default 0.85/0.70 was designed for LLM self-consistency where
        # repeated generations of the same answer have cosine ~0.90+.
        # In proxy mode (answer vs context sentence) MiniLM-L6 cosine
        # typically ranges 0.40–0.70, so the thresholds must be shifted down.
        self.consistency_lf = SemanticConsistencyLF(
            threshold_faithful=0.65,
            threshold_hallucinated=0.45,
        )
        self.reflection_lf = ReflectionTokenLF()

    # ------------------------------------------------------------------
    # Label matrix construction
    # ------------------------------------------------------------------

    def build_label_matrix(self, examples: List[dict]) -> np.ndarray:
        """Apply all 3 LFs to every example and return the label matrix.

        Parameters
        ----------
        examples : List[dict]
            Each dict must have keys: "question", "context", "answer".

        Returns
        -------
        np.ndarray
            Shape (N, 3), dtype int.  Values in {-1, 0, 1}.
            Column order: entailment_lf, consistency_lf, reflection_lf.
        """
        n = len(examples)
        L = np.full((n, 3), ABSTAIN, dtype=int)

        logger.info("Building label matrix for %d examples …", n)

        # --- LF 0: Entailment ---
        try:
            pairs = [(ex.get("context", ""), ex.get("answer", "")) for ex in examples]
            entailment_labels = self.entailment_lf.label_batch(pairs, batch_size=8)
            for i, lbl in enumerate(entailment_labels):
                L[i, 0] = lbl
            logger.info("EntailmentLF done.")
        except Exception as exc:
            logger.warning("EntailmentLF failed for batch: %s", exc)

        # --- LF 1: Semantic Consistency ---
        try:
            consistency_results = self.consistency_lf.label_batch(examples)
            for i, (lbl, _score) in enumerate(consistency_results):
                L[i, 1] = lbl
            logger.info("SemanticConsistencyLF done.")
        except Exception as exc:
            logger.warning("SemanticConsistencyLF failed for batch: %s", exc)

        # --- LF 2: Reflection Token ---
        try:
            reflection_results = self.reflection_lf.label_batch(examples)
            for i, result in enumerate(reflection_results):
                L[i, 2] = result.get("label", ABSTAIN)
            logger.info("ReflectionTokenLF done.")
        except Exception as exc:
            logger.warning("ReflectionTokenLF failed for batch: %s", exc)

        return L

    # ------------------------------------------------------------------
    # LF analysis
    # ------------------------------------------------------------------

    def analyze_lf_coverage(self, L: np.ndarray) -> None:
        """Print per-LF coverage, pairwise overlap, and conflict statistics.

        Parameters
        ----------
        L : np.ndarray
            Label matrix of shape (N, 3).
        """
        n = L.shape[0]
        print("\n" + "=" * 70)
        print(f"{'LF Analysis':^70}")
        print("=" * 70)
        print(f"{'LF':<25} {'Coverage':>10} {'Pos%':>8} {'Neg%':>8} {'Abstain%':>10}")
        print("-" * 70)

        for j, name in enumerate(self.LF_NAMES):
            col = L[:, j]
            labeled = col != ABSTAIN
            coverage = labeled.sum() / n
            pos_rate = (col == HALLUCINATED).sum() / max(labeled.sum(), 1)
            neg_rate = (col == FAITHFUL).sum() / max(labeled.sum(), 1)
            print(
                f"{name:<25} {coverage:>10.2%} {pos_rate:>8.2%} {neg_rate:>8.2%} "
                f"{1 - coverage:>10.2%}"
            )

        print("\nPairwise Conflicts:")
        print("-" * 50)
        for i in range(3):
            for j in range(i + 1, 3):
                both_labeled = (L[:, i] != ABSTAIN) & (L[:, j] != ABSTAIN)
                conflicts = (L[both_labeled, i] != L[both_labeled, j]).sum()
                overlap_n = both_labeled.sum()
                conflict_rate = conflicts / max(overlap_n, 1)
                print(
                    f"  {self.LF_NAMES[i]} vs {self.LF_NAMES[j]}: "
                    f"overlap={overlap_n}, conflicts={conflicts} ({conflict_rate:.2%})"
                )
        print("=" * 70)

    # ------------------------------------------------------------------
    # Label model fitting
    # ------------------------------------------------------------------

    def fit_label_model(self, L: np.ndarray):
        """Fit a Snorkel generative label model on the label matrix.

        Falls back to a naive majority-vote model when snorkel-metal is
        not installed.

        Parameters
        ----------
        L : np.ndarray
            Label matrix shape (N, 3).

        Returns
        -------
        SnorkelLabelModel | _MajorityVoteModel
        """
        if _SNORKEL_AVAILABLE:
            logger.info("Fitting Snorkel LabelModel (n_epochs=100) …")
            model = SnorkelLabelModel(cardinality=2, verbose=False)
            model.fit(L_train=L, n_epochs=100, lr=0.001, seed=42)
            logger.info("Snorkel LabelModel training complete.")
            return model
        else:
            logger.warning("Using majority-vote fallback (install snorkel-metal for full model).")
            return _MajorityVoteModel()

    # ------------------------------------------------------------------
    # Probabilistic labels
    # ------------------------------------------------------------------

    def get_probabilistic_labels(self, L: np.ndarray, label_model) -> np.ndarray:
        """Get soft labels P(hallucinated) for each example.

        Parameters
        ----------
        L : np.ndarray
            Label matrix shape (N, 3).
        label_model
            Fitted label model.

        Returns
        -------
        np.ndarray
            Shape (N,), float in [0, 1].
            0 = confident faithful, 1 = confident hallucinated.
        """
        if _SNORKEL_AVAILABLE and isinstance(label_model, SnorkelLabelModel):
            probs = label_model.predict_proba(L)  # shape (N, 2)
            return probs[:, HALLUCINATED]  # P(hallucinated)
        else:
            return label_model.predict_proba(L)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        examples: List[dict],
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, object]:
        """Run the full weak labeling pipeline.

        Steps:
            1. Build label matrix L
            2. Analyze LF coverage
            3. Fit generative label model
            4. Get probabilistic labels

        Parameters
        ----------
        examples : List[dict]
        save_path : str | None
            Directory to save L matrix and soft labels.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, object]
            (L, soft_labels, label_model)
        """
        L = self.build_label_matrix(examples)
        self.analyze_lf_coverage(L)
        label_model = self.fit_label_model(L)
        soft_labels = self.get_probabilistic_labels(L, label_model)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, "label_matrix.npy"), L)
            np.save(os.path.join(save_path, "soft_labels.npy"), soft_labels)
            logger.info("Saved L and soft_labels to %s", save_path)

        return L, soft_labels, label_model


# ---------------------------------------------------------------------------
# Majority-vote fallback (used when snorkel-metal not installed)
# ---------------------------------------------------------------------------

class _MajorityVoteModel:
    """Naive majority-vote label model used as a Snorkel fallback."""

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """Return P(hallucinated) based on majority vote of non-abstaining LFs.

        Parameters
        ----------
        L : np.ndarray
            Label matrix (N, n_lfs).

        Returns
        -------
        np.ndarray
            Shape (N,), float in [0, 1].
        """
        n = L.shape[0]
        soft = np.zeros(n, dtype=float)
        for i in range(n):
            row = L[i]
            labeled = row[row != ABSTAIN]
            if len(labeled) == 0:
                soft[i] = 0.5  # uncertain
            else:
                soft[i] = float((labeled == HALLUCINATED).mean())
        return soft


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    examples = [
        {
            "question": "Who signed the contract?",
            "context": "Acme Corp and Beta Ltd signed the agreement on Jan 1, 2023.",
            "answer": "Acme Corp and Beta Ltd signed the agreement on Jan 1, 2023.",
            "label": 0,
        },
        {
            "question": "Who signed the contract?",
            "context": "Acme Corp and Beta Ltd signed the agreement on Jan 1, 2023.",
            "answer": "I believe Gamma Inc may have signed this, but I'm not sure.",
            "label": 1,
        },
    ]

    pipeline = WeakLabelPipeline()
    L, soft_labels, model = pipeline.run(examples, save_path="/tmp/smoke_test/")
    print("\nLabel matrix:\n", L)
    print("Soft labels (P(hallucinated)):", soft_labels)
