"""
Hallucination classifiers.

DistilBERTClassifier
    Fine-tunes distilbert-base-uncased on bootstrapped labeled data.
    Input: [CLS] question [SEP] context (truncated) [SEP] answer [SEP]
    Output: binary classification (0=faithful, 1=hallucinated).

LogisticRegressionClassifier
    Simpler baseline: TF-IDF on answer text + sentence embeddings →
    LogisticRegression.  Used as base_classifier in SelfTrainer and CoTrainer.
"""

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sklearn.metrics import f1_score
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch / transformers not installed — DistilBERTClassifier unavailable.")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed — LogisticRegressionClassifier unavailable.")

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ===========================================================================
# Internal Dataset class for DistilBERT fine-tuning
# ===========================================================================

class _HallucinationDataset(Dataset):
    """PyTorch Dataset for fine-tuning DistilBERT."""

    def __init__(self, encodings: dict, labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ===========================================================================
# DistilBERT classifier
# ===========================================================================

class DistilBERTClassifier:
    """Fine-tuned DistilBERT binary classifier for hallucination detection.

    Input format:
        ``{question} [SEP] {context[:300]} [SEP] {answer}``

    The classification head produces logits for [FAITHFUL, HALLUCINATED].

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum token length (default 512).
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
    ):
        """Initialise DistilBERTClassifier."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch and transformers are required.  "
                "Run: pip install torch transformers"
            )

        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("DistilBERTClassifier device: %s", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_input(self, example: dict) -> str:
        """Format example as a single string for tokenization."""
        q = example.get("question", "")
        c = example.get("context", "")[:300]
        a = example.get("answer", "")
        return f"{q} [SEP] {c} [SEP] {a}"

    def _tokenize(self, examples: List[dict]):
        """Tokenize a list of examples."""
        texts = [self._prepare_input(ex) for ex in examples]
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: List[dict],
        labels: List[int],
        val_data: List[dict],
        val_labels: List[int],
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        save_dir: Optional[str] = None,
    ) -> List[dict]:
        """Fine-tune DistilBERT on training data.

        Parameters
        ----------
        train_data : List[dict]
        labels : List[int]
        val_data : List[dict]
        val_labels : List[int]
        epochs : int
        batch_size : int
        lr : float
        save_dir : str | None
            If provided, save the best model checkpoint here.

        Returns
        -------
        List[dict]
            Per-epoch training history: [{"epoch": e, "loss": l, "val_f1": f}].
        """
        from torch.optim import AdamW
        from tqdm import tqdm

        train_enc = self._tokenize(train_data)
        train_dataset = _HallucinationDataset(train_enc, labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        best_f1 = -1.0
        history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(train_loader), 1)
            val_f1 = self._eval_f1(val_data, val_labels)
            history.append({"epoch": epoch, "loss": avg_loss, "val_f1": val_f1})
            logger.info("Epoch %d/%d — loss=%.4f, val_f1=%.4f", epoch, epochs, avg_loss, val_f1)

            if val_f1 > best_f1 and save_dir:
                best_f1 = val_f1
                self.save(save_dir)
                logger.info("  ↑ New best model saved to %s (F1=%.4f)", save_dir, best_f1)

        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, examples: List[dict]) -> np.ndarray:
        """Return hard labels (0 or 1) for each example."""
        probs = self.predict_proba(examples)
        return (probs > 0.5).astype(int)

    @torch.no_grad()
    def predict_proba(self, examples: List[dict], batch_size: int = 32) -> np.ndarray:
        """Return P(hallucinated) for each example.

        Parameters
        ----------
        examples : List[dict]
        batch_size : int

        Returns
        -------
        np.ndarray
            Shape (N,), floats in [0, 1].
        """
        self.model.eval()
        all_probs = []

        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i : i + batch_size]
            enc = self._tokenize(batch_examples)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # P(hallucinated)
            all_probs.append(probs)

        return np.concatenate(all_probs) if all_probs else np.array([])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model and tokenizer to *path*."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("DistilBERTClassifier saved to %s", path)

    def load(self, path: str) -> None:
        """Load model and tokenizer from *path*."""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        logger.info("DistilBERTClassifier loaded from %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_f1(self, val_data: List[dict], val_labels: List[int]) -> float:
        """Compute F1 on validation data."""
        try:
            preds = self.predict(val_data)
            return float(f1_score(val_labels[: len(preds)], preds, zero_division=0))
        except Exception as exc:
            logger.warning("DistilBERT eval F1 failed: %s", exc)
            return 0.0


# ===========================================================================
# Logistic Regression baseline classifier
# ===========================================================================

class LogisticRegressionClassifier:
    """TF-IDF + sentence embeddings → LogisticRegression classifier.

    Used as the base_classifier in SelfTrainer and CoTrainer for fast
    iteration before fine-tuning DistilBERT.

    Features:
        - TF-IDF on answer text (max_features=5000)
        - Sentence-transformer embedding of
          ``"{question} [SEP] {context[:200]} [SEP] {answer}"``
        - Concatenated and fed to LogisticRegression.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        random_state: int = 42,
    ):
        """Initialise LogisticRegressionClassifier."""
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required: pip install scikit-learn")

        self.random_state = random_state
        self.tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        self.clf = LogisticRegression(
            max_iter=1000, random_state=random_state, C=1.0,
            class_weight="balanced",  # handles label skew from weak supervision
        )
        self._tfidf_fitted = False

        if _ST_AVAILABLE:
            logger.info("Loading sentence transformer: %s", embedding_model)
            self.encoder = SentenceTransformer(embedding_model)
        else:
            self.encoder = None
            logger.warning("sentence-transformers unavailable; using TF-IDF only.")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _features(self, examples: List[dict], fit_tfidf: bool = False) -> np.ndarray:
        """Build feature matrix by concatenating TF-IDF and embedding features."""
        answer_texts = [ex.get("answer", "") for ex in examples]

        if fit_tfidf:
            tfidf_feats = self.tfidf.fit_transform(answer_texts).toarray()
            self._tfidf_fitted = True
        else:
            if not self._tfidf_fitted:
                logger.warning("TF-IDF not fitted — fitting on current data as fallback.")
                tfidf_feats = self.tfidf.fit_transform(answer_texts).toarray()
                self._tfidf_fitted = True
            else:
                tfidf_feats = self.tfidf.transform(answer_texts).toarray()

        if self.encoder is not None:
            combined_texts = [
                f"{ex.get('question', '')} [SEP] {ex.get('context', '')[:200]} [SEP] {ex.get('answer', '')}"
                for ex in examples
            ]
            emb_feats = self.encoder.encode(
                combined_texts, convert_to_numpy=True, show_progress_bar=False
            )
            return np.hstack([tfidf_feats, emb_feats])

        return tfidf_feats

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------

    def fit(self, examples: List[dict], labels: List[int]) -> "LogisticRegressionClassifier":
        """Fit TF-IDF and logistic regression on examples.

        Parameters
        ----------
        examples : List[dict]
        labels : List[int]

        Returns
        -------
        self
        """
        X = self._features(examples, fit_tfidf=True)
        y = np.array(labels)
        self.clf.fit(X, y)
        return self

    def predict(self, examples: List[dict]) -> np.ndarray:
        """Return hard labels.

        Parameters
        ----------
        examples : List[dict]

        Returns
        -------
        np.ndarray
            Shape (N,).
        """
        X = self._features(examples)
        return self.clf.predict(X)

    def predict_proba(self, examples: List[dict]) -> np.ndarray:
        """Return P(hallucinated).

        Parameters
        ----------
        examples : List[dict]

        Returns
        -------
        np.ndarray
            Shape (N,), floats in [0, 1].
        """
        X = self._features(examples)
        return self.clf.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    examples = [
        {"question": "Who are the parties?", "context": "Acme Corp and Beta Ltd.",
         "answer": "Acme Corp and Beta Ltd.", "label": 0},
        {"question": "Who are the parties?", "context": "Acme Corp and Beta Ltd.",
         "answer": "I believe Gamma Inc might be involved.", "label": 1},
    ] * 5  # duplicate for enough examples

    labels = [ex["label"] for ex in examples]

    try:
        lr_clf = LogisticRegressionClassifier()
        lr_clf.fit(examples[:8], labels[:8])
        preds = lr_clf.predict(examples[8:])
        probs = lr_clf.predict_proba(examples[8:])
        print("LR predictions:", preds)
        print("LR probabilities:", probs.round(3))
    except Exception as e:
        print(f"LR smoke test skipped: {e}")
