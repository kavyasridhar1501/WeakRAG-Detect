"""
Reflection token labeling function — inspired by Self-RAG.

Simulates Self-RAG's [Retrieve], [ISREL], [ISSUP], [ISUSE] reflection tokens
using heuristics, since we do not have a deployed Self-RAG model.

Three sub-signals:
  1. Hedging language detection      → simulates low [ISSUP] (not supported)
  2. Named entity overlap            → simulates [ISREL] (relevance)
  3. Numeric consistency check       → simulates [ISSUP] (factual grounding)

Special heuristic for ContractNLI misattribution:
    In ContractNLI data, answers are often technically present in the
    context but attributed to the WRONG party or clause.  A dedicated
    misattribution detector checks whether parties mentioned in the
    answer differ from those in the context.

Reference: Asai et al. 2023, "Self-RAG: Learning to Retrieve, Generate,
and Critique through Self-Reflection" (arXiv 2310.11511).
"""

import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import spacy  # type: ignore
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    logger.warning("spaCy not installed – ReflectionTokenLF will use regex fallback.")


def _load_spacy():
    """Load spaCy model, downloading en_core_web_sm if necessary."""
    if not _SPACY_AVAILABLE:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading en_core_web_sm …")
        os.system("python -m spacy download en_core_web_sm")
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Could not load en_core_web_sm; using regex fallback.")
            return None


class ReflectionTokenLF:
    """Simulates Self-RAG reflection token signals using heuristics.

    Sub-scores:
        entity_overlap       – named entity overlap between context and answer.
        numeric_consistency  – numeric value overlap.
        hedging              – presence of hedging language.

    Combined grounding score:
        grounding = 0.5 * entity_overlap + 0.3 * numeric_consistency
                    - 0.2 * hedging_score

    Thresholds:
        grounding > 0.7  → FAITHFUL
        grounding < 0.3  → HALLUCINATED
        else             → ABSTAIN

    Label constants (Snorkel-compatible):
        HALLUCINATED =  1
        FAITHFUL     =  0
        ABSTAIN      = -1
    """

    HALLUCINATED: int = 1
    FAITHFUL: int = 0
    ABSTAIN: int = -1

    # Hedging phrases that signal ungrounded generation (low [ISSUP])
    HEDGING_PHRASES: List[str] = [
        "i think",
        "i believe",
        "possibly",
        "i'm not sure",
        "might be",
        "could be",
        "based on my knowledge",
        "i assume",
        "probably",
        "it seems",
        "i recall",
        "to my knowledge",
        "not certain",
        "i'm unsure",
        "i am not sure",
    ]

    # Common party-role words used in ContractNLI misattribution detection
    _PARTY_ROLES = re.compile(
        r"\b(licensor|licensee|buyer|seller|vendor|client|customer|"
        r"employer|employee|contractor|subcontractor|provider|recipient|"
        r"disclosing party|receiving party|party a|party b)\b",
        re.IGNORECASE,
    )

    def __init__(self):
        """Load spaCy NER model (with automatic download fallback)."""
        self.nlp = _load_spacy()

    # ------------------------------------------------------------------
    # Sub-score helpers
    # ------------------------------------------------------------------

    def _extract_entities_spacy(self, text: str) -> set:
        """Extract named entity strings using spaCy NER."""
        if self.nlp is None:
            return set()
        try:
            doc = self.nlp(text[:1000])  # truncate for speed
            return {ent.text.lower().strip() for ent in doc.ents if ent.text.strip()}
        except Exception as exc:
            logger.debug("spaCy NER failed: %s", exc)
            return set()

    def _extract_entities_regex(self, text: str) -> set:
        """Regex fallback: extract capitalized multi-word phrases as entities."""
        # Match sequences of capitalized words (naive NER substitute)
        matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
        return {m.lower() for m in matches}

    def _entity_overlap_score(self, context: str, answer: str) -> float:
        """Compute named entity overlap between context and answer.

        entity_overlap = |ents_in_answer ∩ ents_in_context| / max(|ents_in_answer|, 1)

        Returns
        -------
        float
            Value in [0, 1].
        """
        if self.nlp:
            ctx_ents = self._extract_entities_spacy(context)
            ans_ents = self._extract_entities_spacy(answer)
        else:
            ctx_ents = self._extract_entities_regex(context)
            ans_ents = self._extract_entities_regex(answer)

        if not ans_ents:
            return 0.5  # neutral when answer has no entities

        overlap = ans_ents & ctx_ents
        return len(overlap) / max(len(ans_ents), 1)

    def _numeric_consistency_score(self, context: str, answer: str) -> float:
        """Compute numeric value overlap between context and answer.

        Extracts all numbers (integers, decimals, percentages) from both texts.
        numeric_overlap = |nums_in_answer ∩ nums_in_context| / max(|nums_in_answer|, 1)

        Returns
        -------
        float
            Value in [0, 1].
        """
        # Pattern matches: integers, decimals, percentages, currency amounts
        number_pattern = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
        ctx_nums = set(number_pattern.findall(context))
        ans_nums = set(number_pattern.findall(answer))

        if not ans_nums:
            return 0.5  # neutral when answer has no numbers

        overlap = ans_nums & ctx_nums
        return len(overlap) / max(len(ans_nums), 1)

    def _hedging_score(self, answer: str) -> float:
        """Count hedging phrases in the answer.

        Returns
        -------
        float
            min(count / 2, 1.0) so 2+ hedges → score of 1.0.
        """
        answer_lower = answer.lower()
        count = sum(1 for phrase in self.HEDGING_PHRASES if phrase in answer_lower)
        return min(count / 2.0, 1.0)

    # ------------------------------------------------------------------
    # Misattribution detector (ContractNLI-specific)
    # ------------------------------------------------------------------

    def _misattribution_score(self, context: str, answer: str) -> float:
        """Detect ContractNLI-style party misattribution.

        In ContractNLI data, answers are often technically present in the
        context but attributed to the WRONG party or clause.  This checks
        whether the party roles mentioned in the answer appear in the context
        in the same combination.

        Returns
        -------
        float
            1.0 if a misattribution is detected, 0.0 otherwise.
        """
        ctx_parties = set(m.lower() for m in self._PARTY_ROLES.findall(context))
        ans_parties = set(m.lower() for m in self._PARTY_ROLES.findall(answer))

        if not ans_parties:
            return 0.0  # no party roles in answer → not applicable

        # If answer mentions parties NOT in the context, flag as misattribution
        unsupported_parties = ans_parties - ctx_parties
        if unsupported_parties:
            logger.debug("Possible misattribution — answer parties not in context: %s", unsupported_parties)
            return 1.0
        return 0.0

    # ------------------------------------------------------------------
    # Single-example labeling
    # ------------------------------------------------------------------

    def label(self, context: str, answer: str) -> dict:
        """Label a single (context, answer) pair with interpretable sub-scores.

        Parameters
        ----------
        context : str
            Retrieved document text.
        answer : str
            Generated answer to evaluate.

        Returns
        -------
        dict with keys:
            label              : int (HALLUCINATED / FAITHFUL / ABSTAIN)
            grounding_score    : float
            entity_overlap     : float
            numeric_consistency: float
            hedging_detected   : bool
            misattribution     : bool  (ContractNLI-specific)
        """
        try:
            entity_score = self._entity_overlap_score(context, answer)
            numeric_score = self._numeric_consistency_score(context, answer)
            hedging_score = self._hedging_score(answer)
            misattribution = self._misattribution_score(context, answer) > 0.5

            # Core grounding score
            grounding_score = (
                0.5 * entity_score
                + 0.3 * numeric_score
                - 0.2 * hedging_score
            )
            # Penalise misattribution
            if misattribution:
                grounding_score -= 0.2
            grounding_score = max(0.0, min(1.0, grounding_score))

            # Thresholds
            if grounding_score > 0.7:
                int_label = self.FAITHFUL
            elif grounding_score < 0.3:
                int_label = self.HALLUCINATED
            else:
                int_label = self.ABSTAIN

            return {
                "label": int_label,
                "grounding_score": round(grounding_score, 4),
                "entity_overlap": round(entity_score, 4),
                "numeric_consistency": round(numeric_score, 4),
                "hedging_detected": hedging_score > 0,
                "misattribution": misattribution,
            }
        except Exception as exc:
            logger.warning("ReflectionTokenLF.label failed: %s", exc)
            return {
                "label": self.ABSTAIN,
                "grounding_score": 0.5,
                "entity_overlap": 0.5,
                "numeric_consistency": 0.5,
                "hedging_detected": False,
                "misattribution": False,
            }

    # ------------------------------------------------------------------
    # Batch labeling
    # ------------------------------------------------------------------

    def label_batch(self, examples: List[dict]) -> List[dict]:
        """Batch-label a list of examples.

        Parameters
        ----------
        examples : List[dict]
            Each dict must have keys "context" and "answer".

        Returns
        -------
        List[dict]
            Result dict per example (same structure as label()).
        """
        results = []
        for ex in examples:
            result = self.label(
                ex.get("context", ""),
                ex.get("answer", ""),
            )
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    lf = ReflectionTokenLF()

    faithful_ex = {
        "context": (
            "This Agreement is entered into between Acme Corp (Licensor) and "
            "Beta Ltd (Licensee).  Payment of $5,000 is due within 30 days."
        ),
        "answer": "Acme Corp is the Licensor and Beta Ltd is the Licensee.  Payment of $5,000 is due within 30 days.",
    }
    hallu_ex = {
        "context": (
            "This Agreement is entered into between Acme Corp (Licensor) and "
            "Beta Ltd (Licensee).  Payment of $5,000 is due within 30 days."
        ),
        "answer": "I believe the Licensee is Acme Corp, and payment might be around $10,000.",
    }
    misattr_ex = {
        "context": (
            "The Disclosing Party shall provide confidential information to "
            "the Receiving Party within 14 days of signing."
        ),
        "answer": "The Receiving Party shall provide confidential information to the Disclosing Party.",
    }

    for name, ex in [("Faithful", faithful_ex), ("Hallucinated", hallu_ex), ("Misattribution", misattr_ex)]:
        result = lf.label(ex["context"], ex["answer"])
        print(f"\n{name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
