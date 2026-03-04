"""
Dataset loaders for WeakRAG-Detect hallucination detection system.

Normalizes LegalBench-RAG, PubMedQA, and SciQ to a common schema:
{
  "id": str,
  "question": str,
  "context": str,
  "answer": str,
  "label": int,   # 1=hallucinated, 0=faithful, None=unknown
  "domain": str,
  "source": str
}
"""

import json
import logging
import os
import random
from typing import List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS_DIR = os.path.join(os.path.dirname(__file__), "seeds")
os.makedirs(SEEDS_DIR, exist_ok=True)


def _normalize_record(
    id_: str,
    question: str,
    context: str,
    answer: str,
    label: Optional[int],
    domain: str,
    source: str,
) -> dict:
    """Return a record conforming to the common schema."""
    return {
        "id": str(id_),
        "question": str(question).strip(),
        "context": str(context).strip(),
        "answer": str(answer).strip(),
        "label": label,
        "domain": domain,
        "source": source,
    }


# ---------------------------------------------------------------------------
# LegalBench-RAG
# ---------------------------------------------------------------------------

def load_legalbench_rag() -> List[dict]:
    """Load LegalBench-RAG (6,858 QA pairs) and return normalized records.

    Sources: CUAD, ContractNLI, MAUD, PrivacyQA
    Query types: termination, party ID, payment terms, liability,
    confidentiality, governing law, indemnification, duration,
    warranty, dispute resolution.

    Returns
    -------
    List[dict]
        Normalized records with domain="legal".
    """
    logger.info("Loading LegalBench-RAG …")
    records: List[dict] = []

    # Primary HuggingFace identifier; fall back to alternative on failure.
    hf_candidates = [
        ("nguyen-brat/legalbench-rag", None),
        ("rcadene/legalbench", None),
    ]

    raw = None
    for hf_id, split in hf_candidates:
        try:
            raw = load_dataset(hf_id, split=split, trust_remote_code=True)
            logger.info("Loaded LegalBench-RAG from %s", hf_id)
            break
        except Exception as exc:
            logger.warning("Failed to load %s: %s", hf_id, exc)

    if raw is None:
        logger.warning(
            "Could not load LegalBench-RAG from HuggingFace. "
            "Returning empty list — place raw files in data/raw/legalbench_rag/ "
            "or install the correct dataset identifier."
        )
        return records

    # Handle DatasetDict vs Dataset
    if hasattr(raw, "items"):
        splits = list(raw.values())
        combined = splits[0]
        for s in splits[1:]:
            combined = combined.concatenate(s) if hasattr(combined, "concatenate") else combined
        raw = combined

    source_map = {
        "cuad": "CUAD",
        "contract_nli": "ContractNLI",
        "contractnli": "ContractNLI",
        "maud": "MAUD",
        "privacyqa": "PrivacyQA",
    }

    for i, row in enumerate(tqdm(raw, desc="LegalBench-RAG")):
        try:
            # Flexible field extraction
            question = (
                row.get("question") or row.get("query") or row.get("input") or ""
            )
            context = (
                row.get("context")
                or row.get("passage")
                or row.get("document")
                or row.get("text")
                or ""
            )
            answer = (
                row.get("answer")
                or row.get("output")
                or row.get("response")
                or ""
            )
            src_raw = str(row.get("source", row.get("dataset", ""))).lower()
            source = next((v for k, v in source_map.items() if k in src_raw), "LegalBench-RAG")

            records.append(
                _normalize_record(
                    id_=f"legal_{i}",
                    question=question,
                    context=context,
                    answer=answer,
                    label=None,  # LegalBench-RAG has no gold hallucination labels
                    domain="legal",
                    source=source,
                )
            )
        except Exception as exc:
            logger.debug("Skipping row %d: %s", i, exc)

    logger.info("Loaded %d LegalBench-RAG records", len(records))
    return records


# ---------------------------------------------------------------------------
# PubMedQA
# ---------------------------------------------------------------------------

def load_pubmedqa() -> List[dict]:
    """Load PubMedQA (pqa_labeled split, ~1 000 examples) and return normalized records.

    Label mapping:
        yes   -> 0 (faithful)
        no    -> 1 (hallucinated)
        maybe -> 1 (potentially hallucinated)

    Returns
    -------
    List[dict]
        Normalized records with domain="medical".
    """
    logger.info("Loading PubMedQA …")
    records: List[dict] = []

    try:
        raw = load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)
    except Exception as exc:
        logger.warning("Failed to load PubMedQA: %s", exc)
        return records

    decision_to_label = {"yes": 0, "no": 1, "maybe": 1}

    split = raw.get("train", raw[list(raw.keys())[0]])
    for i, row in enumerate(tqdm(split, desc="PubMedQA")):
        try:
            question = row.get("question", "")
            # long_answer is the curated paragraph; fall back to abstract sentences
            context_parts = row.get("context", {})
            if isinstance(context_parts, dict):
                contexts = context_parts.get("contexts", []) or context_parts.get("sentences", [])
                context = " ".join(contexts) if isinstance(contexts, list) else str(context_parts)
            else:
                context = str(context_parts)

            answer = row.get("long_answer", row.get("final_decision", ""))
            decision = str(row.get("final_decision", "maybe")).lower().strip()
            label = decision_to_label.get(decision, 1)

            records.append(
                _normalize_record(
                    id_=f"medical_{i}",
                    question=question,
                    context=context,
                    answer=answer,
                    label=label,
                    domain="medical",
                    source="PubMedQA",
                )
            )
        except Exception as exc:
            logger.debug("Skipping PubMedQA row %d: %s", i, exc)

    logger.info("Loaded %d PubMedQA records", len(records))
    return records


# ---------------------------------------------------------------------------
# SciQ
# ---------------------------------------------------------------------------

def load_sciq() -> List[dict]:
    """Load SciQ (13 679 examples) and return normalized records.

    Strategy:
        Non-hallucinated pair: (question, support, correct_answer) -> label=0
        Hallucinated pair:     (question, support, distractor_1)   -> label=1

    Returns
    -------
    List[dict]
        Normalized records with domain="scientific".
    """
    logger.info("Loading SciQ …")
    records: List[dict] = []

    try:
        raw = load_dataset("allenai/sciq", trust_remote_code=True)
    except Exception as exc:
        logger.warning("Failed to load SciQ: %s", exc)
        return records

    idx = 0
    for split_name in ["train", "validation", "test"]:
        split = raw.get(split_name)
        if split is None:
            continue
        for row in tqdm(split, desc=f"SciQ/{split_name}"):
            try:
                question = row.get("question", "")
                support = row.get("support", "")
                correct = row.get("correct_answer", "")
                distractor = row.get("distractor1", row.get("distractor_1", ""))

                # Faithful pair
                if correct:
                    records.append(
                        _normalize_record(
                            id_=f"scientific_{idx}",
                            question=question,
                            context=support,
                            answer=correct,
                            label=0,
                            domain="scientific",
                            source="SciQ",
                        )
                    )
                    idx += 1

                # Hallucinated pair
                if distractor:
                    records.append(
                        _normalize_record(
                            id_=f"scientific_{idx}",
                            question=question,
                            context=support,
                            answer=distractor,
                            label=1,
                            domain="scientific",
                            source="SciQ",
                        )
                    )
                    idx += 1
            except Exception as exc:
                logger.debug("Skipping SciQ row: %s", exc)

    logger.info("Loaded %d SciQ records", len(records))
    return records


# ---------------------------------------------------------------------------
# Seed set construction
# ---------------------------------------------------------------------------

def build_seed_set(
    domain_data: List[dict],
    n_seeds: int = 30,
    strategy: str = "stratified",
    domain: str = "unknown",
) -> Tuple[List[dict], List[dict]]:
    """Select seed examples and return (seeds, remaining_unlabeled).

    Parameters
    ----------
    domain_data : List[dict]
        Full domain dataset (labeled or partially labeled).
    n_seeds : int
        Number of seed examples to select.
    strategy : str
        "stratified" for balanced pos/neg, "random" for random sample.
    domain : str
        Domain name used for the output filename.

    Returns
    -------
    Tuple[List[dict], List[dict]]
        (seed_data, remaining_unlabeled)
    """
    labeled = [d for d in domain_data if d.get("label") is not None]
    unlabeled = [d for d in domain_data if d.get("label") is None]

    if len(labeled) < n_seeds:
        logger.warning(
            "Only %d labeled examples available; requested %d seeds. "
            "Using all labeled examples as seeds.",
            len(labeled),
            n_seeds,
        )
        seeds = labeled
        remaining = unlabeled
    elif strategy == "stratified":
        pos = [d for d in labeled if d["label"] == 1]
        neg = [d for d in labeled if d["label"] == 0]
        random.seed(42)
        n_each = n_seeds // 2
        seeds = random.sample(pos, min(n_each, len(pos))) + random.sample(neg, min(n_each, len(neg)))
        # Fill remainder if one class is small
        if len(seeds) < n_seeds:
            leftover = [d for d in labeled if d not in seeds]
            seeds += random.sample(leftover, min(n_seeds - len(seeds), len(leftover)))
        seeds = seeds[:n_seeds]
        random.shuffle(seeds)
        seed_ids = {d["id"] for d in seeds}
        remaining = [d for d in labeled if d["id"] not in seed_ids] + unlabeled
    else:  # random
        random.seed(42)
        seeds = random.sample(labeled, n_seeds)
        seed_ids = {d["id"] for d in seeds}
        remaining = [d for d in labeled if d["id"] not in seed_ids] + unlabeled

    # Persist seeds to disk
    out_path = os.path.join(SEEDS_DIR, f"{domain}_seeds.json")
    with open(out_path, "w") as f:
        json.dump(seeds, f, indent=2)
    logger.info("Saved %d seeds to %s", len(seeds), out_path)

    return seeds, remaining


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def get_train_test_split(
    domain_data: List[dict],
    test_size: int = 200,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Reserve *test_size* gold-labeled examples for final evaluation.

    Parameters
    ----------
    domain_data : List[dict]
        Full domain dataset.
    test_size : int
        Number of gold examples for the held-out test set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[List[dict], List[dict]]
        (gold_test, unlabeled_pool)
    """
    labeled = [d for d in domain_data if d.get("label") is not None]
    unlabeled = [d for d in domain_data if d.get("label") is None]

    random.seed(seed)
    random.shuffle(labeled)

    gold_test = labeled[:test_size]
    remaining_labeled = labeled[test_size:]
    unlabeled_pool = remaining_labeled + unlabeled

    logger.info(
        "Split: %d gold test | %d unlabeled pool",
        len(gold_test),
        len(unlabeled_pool),
    )
    return gold_test, unlabeled_pool


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=== Smoke test: load_datasets.py ===")

    # Quick SciQ test (always available)
    sciq = load_sciq()
    print(f"SciQ: {len(sciq)} records. First record:")
    print(json.dumps(sciq[0], indent=2))

    gold, pool = get_train_test_split(sciq, test_size=200)
    seeds, remaining = build_seed_set(pool, n_seeds=10, domain="scientific")
    print(f"Gold test: {len(gold)}, Pool: {len(pool)}, Seeds: {len(seeds)}")
