"""
Majority-vote ensemble of multiple classifiers.

Trains N classifiers (default 3) on bootstrapped data with different random
seeds and combines their predictions via majority vote.
"""

import copy
import logging
from typing import Callable, List

import numpy as np

logger = logging.getLogger(__name__)


class MajorityVoteEnsemble:
    """Train multiple classifiers with different seeds; predict by majority vote.

    Parameters
    ----------
    base_classifier_fn : Callable[[], classifier]
        Zero-argument factory that returns a fresh classifier instance.
        The classifier must implement fit(examples, labels),
        predict(examples), and predict_proba(examples).
    n_classifiers : int
        Number of ensemble members (default 3).
    seeds : List[int]
        One random seed per classifier.  Length must equal n_classifiers.
    """

    def __init__(
        self,
        base_classifier_fn: Callable,
        n_classifiers: int = 3,
        seeds: List[int] = None,
    ):
        """Initialise MajorityVoteEnsemble."""
        self.base_classifier_fn = base_classifier_fn
        self.n_classifiers = n_classifiers
        self.seeds = seeds if seeds is not None else [42, 123, 456]
        if len(self.seeds) != n_classifiers:
            raise ValueError(
                f"len(seeds)={len(self.seeds)} must equal n_classifiers={n_classifiers}."
            )
        self.classifiers = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_data: List[dict], labels: List[int]) -> "MajorityVoteEnsemble":
        """Train each classifier with a different random seed.

        Parameters
        ----------
        train_data : List[dict]
        labels : List[int]

        Returns
        -------
        self
        """
        self.classifiers = []
        for seed in self.seeds:
            clf = self.base_classifier_fn()
            # Inject seed if the classifier supports it
            if hasattr(clf, "random_state"):
                clf.random_state = seed
            elif hasattr(clf, "set_params"):
                try:
                    clf.set_params(random_state=seed)
                except Exception:
                    pass

            # Shuffle training data with this seed
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(train_data))
            shuffled_data = [train_data[i] for i in idx]
            shuffled_labels = [labels[i] for i in idx]

            logger.info("Fitting ensemble member (seed=%d) on %d examples …", seed, len(shuffled_data))
            clf.fit(shuffled_data, shuffled_labels)
            self.classifiers.append(clf)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, examples: List[dict]) -> np.ndarray:
        """Return majority-vote hard labels.

        Parameters
        ----------
        examples : List[dict]

        Returns
        -------
        np.ndarray
            Shape (N,), integer labels.
        """
        if not self.classifiers:
            raise RuntimeError("Call fit() before predict().")

        votes = np.zeros((len(examples), self.n_classifiers), dtype=int)
        for j, clf in enumerate(self.classifiers):
            votes[:, j] = clf.predict(examples)

        # Majority vote (ties go to HALLUCINATED = 1)
        majority = (votes.mean(axis=1) >= 0.5).astype(int)
        return majority

    def predict_proba(self, examples: List[dict]) -> np.ndarray:
        """Return mean P(hallucinated) across all classifiers.

        Parameters
        ----------
        examples : List[dict]

        Returns
        -------
        np.ndarray
            Shape (N,), floats in [0, 1].
        """
        if not self.classifiers:
            raise RuntimeError("Call fit() before predict_proba().")

        probs = np.zeros((len(examples), self.n_classifiers), dtype=float)
        for j, clf in enumerate(self.classifiers):
            probs[:, j] = clf.predict_proba(examples)

        return probs.mean(axis=1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"MajorityVoteEnsemble(n_classifiers={self.n_classifiers}, "
            f"seeds={self.seeds})"
        )
