"""
Entity clustering for hallucination taxonomy construction.

Extracts named entities from hallucinated answers, clusters them with KMeans,
and generates a JSON taxonomy describing hallucination types per domain.

Key insight from LegalInsight: ContractNLI answers are often technically
present in the context but attributed to the wrong party/clause.  A dedicated
"misattribution" cluster is seeded for the legal domain.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def _load_spacy():
    if not _SPACY_AVAILABLE:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            return None


class HallucinationTaxonomyBuilder:
    """Extract, cluster, and taxonomize entities from hallucinated answers.

    Parameters
    ----------
    embedding_model : str
        Sentence-transformer model for encoding entity strings.
    """

    # Cluster label suggestions (used as hint names; manually inspectable)
    _CLUSTER_HINTS = {
        0: "fabricated_citations",
        1: "wrong_party_names",
        2: "incorrect_dates_numbers",
        3: "misattributed_obligations",
        4: "invented_legal_terms",
        5: "misattribution",  # LegalInsight ContractNLI-specific
    }

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialise HallucinationTaxonomyBuilder."""
        self.nlp = _load_spacy()
        if _ST_AVAILABLE:
            self.encoder = SentenceTransformer(embedding_model)
        else:
            self.encoder = None
            logger.warning("sentence-transformers unavailable — clustering may be degraded.")

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def extract_hallucinated_entities(
        self, hallucinated_examples: List[dict]
    ) -> List[dict]:
        """Extract named entities from hallucinated answers.

        Parameters
        ----------
        hallucinated_examples : List[dict]
            Each dict should have "answer" and optionally "context".

        Returns
        -------
        List[dict]
            [{"entity_text": str, "entity_type": str, "answer_context": str}]
        """
        entities: List[dict] = []

        for ex in hallucinated_examples:
            answer = ex.get("answer", "")
            if not answer.strip():
                continue

            if self.nlp is not None:
                try:
                    doc = self.nlp(answer[:500])
                    for ent in doc.ents:
                        entities.append({
                            "entity_text": ent.text.strip(),
                            "entity_type": ent.label_,
                            "answer_context": answer[:200],
                            "example_id": ex.get("id", ""),
                        })
                except Exception as exc:
                    logger.debug("spaCy NER failed for example: %s", exc)
            else:
                # Regex fallback: extract capitalized phrases
                import re
                matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", answer)
                for m in matches:
                    entities.append({
                        "entity_text": m,
                        "entity_type": "UNKNOWN",
                        "answer_context": answer[:200],
                        "example_id": ex.get("id", ""),
                    })

        logger.info("Extracted %d entities from %d hallucinated examples.", len(entities), len(hallucinated_examples))
        return entities

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_entities(
        self,
        entities: List[dict],
        n_clusters: int = 6,
        seed: int = 42,
    ) -> Dict[int, List[dict]]:
        """Cluster entity strings using KMeans on sentence embeddings.

        Parameters
        ----------
        entities : List[dict]
            Output of extract_hallucinated_entities().
        n_clusters : int
            Number of clusters (default 6).
        seed : int
            Random seed.

        Returns
        -------
        Dict[int, List[dict]]
            {cluster_id: [entity_dict, ...]}, with top-5 representatives per cluster.
        """
        if not entities:
            logger.warning("No entities to cluster.")
            return {}

        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for clustering.")

        entity_texts = [e["entity_text"] for e in entities]

        # Encode
        if self.encoder is not None:
            embeddings = self.encoder.encode(entity_texts, convert_to_numpy=True, show_progress_bar=True)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(max_features=200, analyzer="char_wb", ngram_range=(2, 3))
            embeddings = tfidf.fit_transform(entity_texts).toarray()

        n_clusters = min(n_clusters, len(entities))
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Group by cluster
        clustered: Dict[int, List[dict]] = {i: [] for i in range(n_clusters)}
        for entity, label in zip(entities, cluster_labels):
            entity_with_cluster = {**entity, "cluster_id": int(label)}
            clustered[int(label)].append(entity_with_cluster)

        # Show top-5 representatives per cluster
        for cid, members in clustered.items():
            hint = self._CLUSTER_HINTS.get(cid, f"cluster_{cid}")
            top5 = [m["entity_text"] for m in members[:5]]
            logger.info("Cluster %d (%s): %d entities — top 5: %s", cid, hint, len(members), top5)

        return clustered

    # ------------------------------------------------------------------
    # Taxonomy construction
    # ------------------------------------------------------------------

    def build_taxonomy(
        self,
        domain: str,
        clustered_entities: Dict[int, List[dict]],
    ) -> dict:
        """Build a JSON taxonomy of hallucination types.

        Parameters
        ----------
        domain : str
        clustered_entities : Dict[int, List[dict]]
            Output of cluster_entities().

        Returns
        -------
        dict
            Taxonomy JSON, also saved to results/{domain}/hallucination_taxonomy.json.
        """
        hallucination_types = []
        for cid, members in clustered_entities.items():
            label = self._CLUSTER_HINTS.get(cid, f"cluster_{cid}")
            examples = list({m["entity_text"] for m in members})[:10]

            # Special handling for misattribution (ContractNLI insight)
            if label == "misattribution" or (
                domain == "legal" and cid == max(clustered_entities.keys())
            ):
                label = "misattribution"
                description = (
                    "Answer text is technically present in the context but "
                    "attributed to the wrong party or clause — a common error "
                    "in ContractNLI data (LegalInsight key insight)."
                )
            else:
                description = f"Cluster {cid}: entities grouped by semantic similarity."

            hallucination_types.append({
                "cluster_id": cid,
                "label": label,
                "description": description,
                "examples": examples,
                "count": len(members),
            })

        taxonomy = {
            "domain": domain,
            "total_entities": sum(len(v) for v in clustered_entities.values()),
            "n_clusters": len(clustered_entities),
            "hallucination_types": sorted(hallucination_types, key=lambda x: -x["count"]),
        }

        out_dir = os.path.join(RESULTS_DIR, domain)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "hallucination_taxonomy.json")
        with open(path, "w") as f:
            json.dump(taxonomy, f, indent=2)
        logger.info("Saved hallucination taxonomy to %s", path)

        return taxonomy
