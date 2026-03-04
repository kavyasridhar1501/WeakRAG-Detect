"""
NLI-based hallucination labeling function using DeBERTa cross-encoder.

Checks whether the retrieved context ENTAILS the generated answer.
If the context *contradicts* the answer, the answer is flagged as hallucinated.

Model: cross-encoder/nli-deberta-v3-small (HuggingFace)
"""

import logging
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed – EntailmentLF unavailable.")


class EntailmentLF:
    """NLI-based labeling function.

    Checks if the retrieved context ENTAILS the generated answer.
    If the context contradicts the answer, the answer is likely hallucinated.

    Label constants (compatible with Snorkel):
        HALLUCINATED =  1
        FAITHFUL     =  0
        ABSTAIN      = -1
    """

    HALLUCINATED: int = 1
    FAITHFUL: int = 0
    ABSTAIN: int = -1

    # Label order is model-specific; we resolve it at init time from id2label
    _LABEL_ORDER: dict = {}

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
        threshold: float = 0.5,
    ):
        """Load the cross-encoder NLI model.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier.
        threshold : float
            Confidence threshold for making a hard decision (default 0.5).
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Install transformers: pip install transformers")

        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading NLI model '%s' on %s …", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Resolve label-to-index mapping from model config
        id2label = self.model.config.id2label  # e.g. {0: "entailment", 1: "neutral", 2: "contradiction"}
        self._label_to_idx = {v.lower(): k for k, v in id2label.items()}
        logger.info("NLI label mapping: %s", self._label_to_idx)

    # ------------------------------------------------------------------
    # Single-example labeling
    # ------------------------------------------------------------------

    def label(self, context: str, answer: str) -> int:
        """Label a single (context, answer) pair.

        Parameters
        ----------
        context : str
            Retrieved document text (premise).
        answer : str
            Generated answer to evaluate (hypothesis).

        Returns
        -------
        int
            HALLUCINATED (1), FAITHFUL (0), or ABSTAIN (-1).
        """
        try:
            scores = self._run_nli(context, answer)
            contradiction_idx = self._label_to_idx.get("contradiction", 0)
            entailment_idx = self._label_to_idx.get("entailment", 2)

            contradiction_score = scores[contradiction_idx]
            entailment_score = scores[entailment_idx]

            if contradiction_score > self.threshold:
                return self.HALLUCINATED
            if entailment_score > self.threshold:
                return self.FAITHFUL
            return self.ABSTAIN
        except Exception as exc:
            logger.warning("EntailmentLF.label failed: %s", exc)
            return self.ABSTAIN

    def label_with_scores(self, context: str, answer: str) -> Tuple[int, float]:
        """Label and return the maximum confidence score.

        Returns
        -------
        Tuple[int, float]
            (label, confidence_score)
        """
        try:
            scores = self._run_nli(context, answer)
            contradiction_idx = self._label_to_idx.get("contradiction", 0)
            entailment_idx = self._label_to_idx.get("entailment", 2)

            contradiction_score = scores[contradiction_idx]
            entailment_score = scores[entailment_idx]

            if contradiction_score > self.threshold:
                return self.HALLUCINATED, float(contradiction_score)
            if entailment_score > self.threshold:
                return self.FAITHFUL, float(entailment_score)
            return self.ABSTAIN, float(max(scores))
        except Exception as exc:
            logger.warning("EntailmentLF.label_with_scores failed: %s", exc)
            return self.ABSTAIN, 0.0

    # ------------------------------------------------------------------
    # Batch labeling
    # ------------------------------------------------------------------

    def label_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32,
    ) -> List[int]:
        """Batch-label (context, answer) pairs.

        Parameters
        ----------
        pairs : List[Tuple[str, str]]
            Each tuple is (context, answer).
        batch_size : int
            Number of pairs to process per forward pass.

        Returns
        -------
        List[int]
            Label per pair: HALLUCINATED, FAITHFUL, or ABSTAIN.
        """
        labels: List[int] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            try:
                batch_labels = self._run_nli_batch(batch)
                labels.extend(batch_labels)
            except Exception as exc:
                logger.warning("Batch %d failed: %s — falling back to per-item", i, exc)
                for ctx, ans in batch:
                    labels.append(self.label(ctx, ans))
        return labels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_nli(self, premise: str, hypothesis: str) -> List[float]:
        """Run a single NLI inference and return softmax probabilities."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        return probs

    @torch.no_grad()
    def _run_nli_batch(self, pairs: List[Tuple[str, str]]) -> List[int]:
        """Run NLI on a batch of (premise, hypothesis) pairs."""
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]

        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu()

        contradiction_idx = self._label_to_idx.get("contradiction", 0)
        entailment_idx = self._label_to_idx.get("entailment", 2)

        batch_labels: List[int] = []
        for row in probs:
            c_score = row[contradiction_idx].item()
            e_score = row[entailment_idx].item()
            if c_score > self.threshold:
                batch_labels.append(self.HALLUCINATED)
            elif e_score > self.threshold:
                batch_labels.append(self.FAITHFUL)
            else:
                batch_labels.append(self.ABSTAIN)
        return batch_labels


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    lf = EntailmentLF()

    ctx = "The contract was signed between Acme Corp and Beta Ltd on January 1, 2023."
    faithful_ans = "Acme Corp and Beta Ltd signed the contract on January 1, 2023."
    hallucinated_ans = "The contract was signed between Acme Corp and Gamma Inc on March 5, 2024."

    print("Faithful answer label:", lf.label(ctx, faithful_ans))      # expect 0
    print("Hallucinated answer label:", lf.label(ctx, hallucinated_ans))  # expect 1

    labels = lf.label_batch([(ctx, faithful_ans), (ctx, hallucinated_ans)])
    print("Batch labels:", labels)
