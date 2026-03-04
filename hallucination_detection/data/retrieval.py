"""
FAISS-based semantic retriever ported and simplified from LegalInsight.

Uses BGE-M3 (BAAI/bge-m3) for document and query encoding,
matching the retrieval model used in the prior LegalInsight system.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Lazy imports to avoid hard failures when GPU libs are absent
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed. FAISSRetriever will be unavailable.")

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed.")

INDEX_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(INDEX_DIR, exist_ok=True)


class FAISSRetriever:
    """Semantic document retriever using BGE-M3 + FAISS IndexFlatIP.

    Ported from LegalInsight's backend retrieval component and simplified
    for standalone use across all three domains.

    Parameters
    ----------
    model_name : str
        HuggingFace sentence-transformer model.  Defaults to BAAI/bge-m3,
        the same model used in LegalInsight.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """Initialize the retriever with the given embedding model."""
        if not (_FAISS_AVAILABLE and _ST_AVAILABLE):
            raise RuntimeError(
                "faiss-cpu and sentence-transformers are required. "
                "Run: pip install faiss-cpu sentence-transformers"
            )

        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        self.index: Optional[faiss.IndexFlatIP] = None
        self._ids: List[str] = []
        self._texts: List[str] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(
        self,
        documents: List[str],
        ids: List[str],
        domain: str = "default",
        batch_size: int = 64,
        save: bool = True,
    ) -> None:
        """Encode *documents* and build a FAISS IndexFlatIP.

        Parameters
        ----------
        documents : List[str]
            Raw text of each document/passage to index.
        ids : List[str]
            Unique identifier for each document (parallel list).
        domain : str
            Used as a prefix for the saved index file.
        batch_size : int
            Encoding batch size.
        save : bool
            If True, persist the index to *data/processed/{domain}_faiss.index*.
        """
        if len(documents) != len(ids):
            raise ValueError("documents and ids must have the same length.")

        logger.info("Encoding %d documents with %s …", len(documents), self.model_name)
        embeddings = self._encode(documents, batch_size=batch_size)

        # Normalize to unit vectors so inner-product = cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self._ids = list(ids)
        self._texts = list(documents)

        logger.info("Indexed %d documents (dim=%d)", self.index.ntotal, dim)

        if save:
            index_path = os.path.join(INDEX_DIR, f"{domain}_faiss.index")
            id_path = os.path.join(INDEX_DIR, f"{domain}_ids.npy")
            faiss.write_index(self.index, index_path)
            np.save(id_path, np.array(self._ids, dtype=object))
            logger.info("Saved FAISS index to %s", index_path)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Return the top-*k* documents most similar to *query*.

        Parameters
        ----------
        query : str
            Query string.
        k : int
            Number of results to return.

        Returns
        -------
        List[dict]
            Each entry: {"id": str, "text": str, "score": float}
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call index_documents() or load_index() first.")

        vec = self._encode([query])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        scores, indices = self.index.search(vec.astype(np.float32), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "id": self._ids[idx],
                "text": self._texts[idx],
                "score": float(score),
            })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_index(self, domain: str) -> None:
        """Load a previously saved FAISS index from disk.

        Parameters
        ----------
        domain : str
            Domain prefix used when the index was saved.
        """
        index_path = os.path.join(INDEX_DIR, f"{domain}_faiss.index")
        id_path = os.path.join(INDEX_DIR, f"{domain}_ids.npy")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self.index = faiss.read_index(index_path)
        self._ids = list(np.load(id_path, allow_pickle=True))
        logger.info(
            "Loaded FAISS index with %d vectors from %s",
            self.index.ntotal,
            index_path,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts into dense vectors using the loaded sentence transformer."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.astype(np.float32)

    @classmethod
    def build_or_load(
        cls,
        documents: List[str],
        ids: List[str],
        domain: str = "default",
        model_name: str = "BAAI/bge-m3",
    ) -> "FAISSRetriever":
        """Build index if not cached on disk, otherwise load from disk.

        Parameters
        ----------
        documents : List[str]
        ids : List[str]
        domain : str
        model_name : str

        Returns
        -------
        FAISSRetriever
        """
        retriever = cls(model_name=model_name)
        index_path = os.path.join(INDEX_DIR, f"{domain}_faiss.index")
        if os.path.exists(index_path):
            logger.info("Loading cached FAISS index for domain '%s'", domain)
            retriever.load_index(domain)
            retriever._texts = list(documents)  # keep texts for result display
        else:
            retriever.index_documents(documents, ids, domain=domain)
        return retriever


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = [
        "The contract shall terminate upon 30 days written notice.",
        "Payment is due within 15 days of invoice receipt.",
        "Liability is limited to the total contract value.",
    ]
    ids = ["doc_0", "doc_1", "doc_2"]

    try:
        retriever = FAISSRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
        retriever.index_documents(docs, ids, domain="smoke_test")
        results = retriever.retrieve("When does the agreement end?", k=2)
        print("Smoke test results:")
        for r in results:
            print(f"  [{r['score']:.4f}] {r['id']}: {r['text'][:80]}")
    except Exception as e:
        print(f"Smoke test skipped (dependencies not installed): {e}")
