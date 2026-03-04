"""
Two-view co-training bootstrapper.

View 1: TF-IDF features on answer text (surface / lexical features)
View 2: Sentence embeddings of [question + context + answer] (semantic features)

Each view's classifier labels high-confidence examples for the other view,
allowing the two classifiers to teach each other from unlabeled data.
"""

import logging
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed – CoTrainer view 2 degraded.")


class CoTrainer:
    """Two-view co-training bootstrapper.

    View 1 (surface): TF-IDF on answer text, max_features=5000.
    View 2 (semantic): sentence-embedding of
        ``"{question} [SEP] {context[:200]} [SEP] {answer}"``.

    Each view's classifier labels high-confidence unlabeled examples for
    the OTHER view, allowing them to teach each other iteratively.

    The final ensemble predicts by majority vote of both classifiers.

    Parameters
    ----------
    confidence_threshold : float
        Probability threshold for selecting high-confidence examples.
    max_iterations : int
        Maximum number of co-training iterations.
    embedding_model : str
        Sentence-transformer model name for view 2.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        max_iterations: int = 5,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialise CoTrainer."""
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.embedding_model_name = embedding_model

        # View 1: TF-IDF
        self.tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        self.clf1 = LogisticRegression(max_iter=1000, random_state=42, C=1.0)

        # View 2: Sentence embeddings
        if _ST_AVAILABLE:
            logger.info("Loading sentence transformer for co-training view 2: %s", embedding_model)
            self.encoder = SentenceTransformer(embedding_model)
        else:
            self.encoder = None
        self.clf2 = LogisticRegression(max_iter=1000, random_state=42, C=1.0)

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    def _build_view1_features(self, examples: List[dict], fit: bool = False) -> np.ndarray:
        """TF-IDF features on answer text.

        Parameters
        ----------
        examples : List[dict]
        fit : bool
            If True, fit the TF-IDF vectorizer (only on seed data).

        Returns
        -------
        np.ndarray
            Shape (N, max_features).
        """
        texts = [ex.get("answer", "") for ex in examples]
        if fit:
            return self.tfidf.fit_transform(texts).toarray()
        return self.tfidf.transform(texts).toarray()

    def _build_view2_features(self, examples: List[dict]) -> np.ndarray:
        """Sentence embeddings of the concatenated question + context + answer.

        Parameters
        ----------
        examples : List[dict]

        Returns
        -------
        np.ndarray
            Shape (N, embedding_dim).
        """
        texts = [
            f"{ex.get('question', '')} [SEP] {ex.get('context', '')[:200]} [SEP] {ex.get('answer', '')}"
            for ex in examples
        ]
        if self.encoder is not None:
            return self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:
            # Fallback: character n-gram TF-IDF
            fallback_tfidf = TfidfVectorizer(max_features=500, analyzer="char_wb", ngram_range=(3, 4))
            return fallback_tfidf.fit_transform(texts).toarray()

    # ------------------------------------------------------------------
    # Main co-training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        seed_data: List[dict],
        unlabeled_data: List[dict],
        soft_labels: np.ndarray,
        val_data: List[dict],
    ) -> dict:
        """Run iterative co-training.

        Parameters
        ----------
        seed_data : List[dict]
            Manually labeled seed examples.
        unlabeled_data : List[dict]
        soft_labels : np.ndarray
            P(hallucinated) per unlabeled example from Snorkel.
        val_data : List[dict]
            Validation examples with gold labels.

        Returns
        -------
        dict with keys:
            "final_model"      – ensemble (this CoTrainer instance),
            "training_history" – list of per-iteration dicts,
            "labeled_pool"     – combined training examples after all iterations,
            "labels"           – corresponding integer labels.
        """
        history = []

        # Separate seeds + high-confidence Snorkel examples for initialisation
        init_data, init_labels = self._init_pool(seed_data, unlabeled_data, soft_labels)
        used_ids = {ex["id"] for ex in init_data}

        remaining = [ex for ex in unlabeled_data if ex["id"] not in used_ids]

        # Build initial features
        X1_init = self._build_view1_features(init_data, fit=True)
        X2_init = self._build_view2_features(init_data)
        y_init = np.array(init_labels)

        # Initial fit
        self.clf1.fit(X1_init, y_init)
        self.clf2.fit(X2_init, y_init)

        pool1 = list(init_data)
        labels1 = list(init_labels)
        pool2 = list(init_data)
        labels2 = list(init_labels)

        val_f1 = self._eval_ensemble(val_data)
        history.append({"iteration": 0, "train_size": len(init_data), "val_f1": val_f1})
        logger.info("Co-Training Iteration 0: train_size=%d, Val F1=%.4f", len(init_data), val_f1)

        prev_f1 = val_f1
        for iteration in range(1, self.max_iterations + 1):
            if not remaining:
                break

            X1_rem = self._build_view1_features(remaining)
            X2_rem = self._build_view2_features(remaining)

            p1 = self.clf1.predict_proba(X1_rem)[:, 1]  # P(hallucinated) from clf1
            p2 = self.clf2.predict_proba(X2_rem)[:, 1]  # P(hallucinated) from clf2

            # clf1 labels high-confidence examples for clf2
            idx_for_clf2 = np.where((p1 > self.confidence_threshold) | (p1 < (1 - self.confidence_threshold)))[0]
            # clf2 labels high-confidence examples for clf1
            idx_for_clf1 = np.where((p2 > self.confidence_threshold) | (p2 < (1 - self.confidence_threshold)))[0]

            newly_labeled = set()

            for i in idx_for_clf2:
                ex = remaining[i]
                pool2.append(ex)
                labels2.append(int(p1[i] > 0.5))
                newly_labeled.add(i)

            for i in idx_for_clf1:
                ex = remaining[i]
                pool1.append(ex)
                labels1.append(int(p2[i] > 0.5))
                newly_labeled.add(i)

            if len(newly_labeled) == 0:
                logger.info("Co-Training: no new high-confidence examples. Stopping.")
                break

            # Retrain both classifiers
            X1_pool = self._build_view1_features(pool1)
            self.clf1.fit(X1_pool, np.array(labels1))

            X2_pool = self._build_view2_features(pool2)
            self.clf2.fit(X2_pool, np.array(labels2))

            # Remove labeled examples from the unlabeled pool
            keep_mask = np.ones(len(remaining), dtype=bool)
            keep_mask[list(newly_labeled)] = False
            remaining = [ex for ex, keep in zip(remaining, keep_mask) if keep]

            val_f1 = self._eval_ensemble(val_data)
            history.append({
                "iteration": iteration,
                "train_size": max(len(pool1), len(pool2)),
                "val_f1": val_f1,
                "newly_labeled": len(newly_labeled),
            })
            logger.info(
                "Co-Training Iteration %d: +%d examples, Val F1=%.4f (Δ=%+.4f)",
                iteration, len(newly_labeled), val_f1, val_f1 - prev_f1,
            )

            if abs(val_f1 - prev_f1) < 0.001:
                logger.info("F1 plateau. Stopping co-training.")
                break
            prev_f1 = val_f1

        # Merge pools for downstream use
        merged_data, merged_labels = self._merge_pools(
            pool1, labels1, pool2, labels2
        )

        return {
            "final_model": self,  # ensemble predictions via predict / predict_proba
            "training_history": history,
            "labeled_pool": merged_data,
            "labels": merged_labels,
        }

    # ------------------------------------------------------------------
    # Ensemble inference
    # ------------------------------------------------------------------

    def predict(self, examples: List[dict]) -> np.ndarray:
        """Return hard labels from majority vote of both classifiers."""
        probs = self.predict_proba(examples)
        return (probs > 0.5).astype(int)

    def predict_proba(self, examples: List[dict]) -> np.ndarray:
        """Return mean P(hallucinated) from both classifiers."""
        X1 = self._build_view1_features(examples)
        X2 = self._build_view2_features(examples)
        p1 = self.clf1.predict_proba(X1)[:, 1]
        p2 = self.clf2.predict_proba(X2)[:, 1]
        return (p1 + p2) / 2.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_pool(self, seed_data, unlabeled_data, soft_labels):
        """Combine seeds and high-confidence Snorkel examples."""
        pool = list(seed_data)
        labels = [ex["label"] for ex in seed_data]
        for ex, p in zip(unlabeled_data, soft_labels):
            if p > self.confidence_threshold or p < (1 - self.confidence_threshold):
                pool.append(ex)
                labels.append(int(p > 0.5))
        return pool, labels

    def _eval_ensemble(self, val_data: List[dict]) -> float:
        """Compute F1 of the ensemble on val_data."""
        if not val_data:
            return 0.0
        try:
            golds = [ex["label"] for ex in val_data if ex.get("label") is not None]
            preds = self.predict(val_data)[: len(golds)]
            return float(f1_score(golds, preds, zero_division=0))
        except Exception as exc:
            logger.warning("Ensemble F1 evaluation failed: %s", exc)
            return 0.0

    def _merge_pools(self, pool1, labels1, pool2, labels2):
        """Merge the two pools, deduplicating by example ID."""
        seen = {}
        for ex, lbl in zip(pool1, labels1):
            seen[ex["id"]] = (ex, lbl)
        for ex, lbl in zip(pool2, labels2):
            if ex["id"] not in seen:
                seen[ex["id"]] = (ex, lbl)
        merged_data = [v[0] for v in seen.values()]
        merged_labels = [v[1] for v in seen.values()]
        return merged_data, merged_labels
