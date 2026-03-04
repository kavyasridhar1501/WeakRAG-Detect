"""
N-gram PMI analysis for hallucination pattern mining.

Identifies lexical patterns that discriminate hallucinated from faithful
answers by computing Pointwise Mutual Information (PMI) for n-grams.
"""

import csv
import logging
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

try:
    from sklearn.feature_extraction.text import CountVectorizer
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed — HallucinationPatternMiner unavailable.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


class HallucinationPatternMiner:
    """Mine discriminative n-gram patterns from hallucinated vs. faithful answers.

    Uses Pointwise Mutual Information (PMI) to rank n-grams:
        PMI(ngram | hallucinated) = log P(ngram | hallucinated) / P(ngram | faithful)

    Higher PMI → n-gram is disproportionately common in hallucinated answers.
    """

    def extract_ngrams(
        self,
        hallucinated_answers: List[str],
        faithful_answers: List[str],
        domain: str,
        n_range: Tuple[int, int] = (2, 3),
        top_k: int = 20,
    ) -> List[dict]:
        """Extract top-k discriminative n-grams by PMI.

        Parameters
        ----------
        hallucinated_answers : List[str]
        faithful_answers : List[str]
        domain : str
            Used for the output directory.
        n_range : Tuple[int, int]
            Min and max n-gram sizes.
        top_k : int
            Number of top n-grams to return.

        Returns
        -------
        List[dict]
            Sorted by PMI: [{"ngram": str, "pmi": float, "hall_freq": int, "faith_freq": int}]
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required.")

        if not hallucinated_answers or not faithful_answers:
            logger.warning("Empty answer lists provided to extract_ngrams.")
            return []

        vectorizer = CountVectorizer(
            ngram_range=n_range,
            stop_words="english",
            min_df=1,
        )

        # Fit on combined corpus, then transform separately
        all_texts = hallucinated_answers + faithful_answers
        vectorizer.fit(all_texts)
        vocab = vectorizer.get_feature_names_out()

        hall_counts = vectorizer.transform(hallucinated_answers).toarray().sum(axis=0)
        faith_counts = vectorizer.transform(faithful_answers).toarray().sum(axis=0)

        hall_total = max(hall_counts.sum(), 1)
        faith_total = max(faith_counts.sum(), 1)

        results = []
        for i, ngram in enumerate(vocab):
            p_hall = (hall_counts[i] + 1) / hall_total  # add-1 smoothing
            p_faith = (faith_counts[i] + 1) / faith_total
            pmi = float(np.log(p_hall / p_faith))
            results.append({
                "ngram": ngram,
                "pmi": pmi,
                "hall_freq": int(hall_counts[i]),
                "faith_freq": int(faith_counts[i]),
            })

        results.sort(key=lambda x: x["pmi"], reverse=True)
        top_results = results[:top_k]

        # Save results and plot
        out_dir = os.path.join(RESULTS_DIR, domain)
        os.makedirs(out_dir, exist_ok=True)
        self._save_results(top_results, os.path.join(out_dir, "ngram_pmi.csv"))
        self._plot_pmi(top_results, domain, os.path.join(out_dir, "ngram_pmi.png"))

        logger.info("Extracted %d top n-grams for domain '%s'.", len(top_results), domain)
        return top_results

    def compare_domains(self, domain_results: Dict[str, List[dict]]) -> None:
        """Compare n-gram patterns across multiple domains.

        Identifies n-grams shared across domains (universal hallucination
        markers) and those unique to each domain.

        Parameters
        ----------
        domain_results : dict
            {domain: [{"ngram": str, "pmi": float, ...}]}
        """
        out_dir = RESULTS_DIR
        os.makedirs(out_dir, exist_ok=True)

        # Build sets of top n-grams per domain
        domain_ngrams: Dict[str, set] = {
            d: {item["ngram"] for item in items}
            for d, items in domain_results.items()
        }
        domains = list(domain_results.keys())

        rows = []
        all_ngrams = set().union(*domain_ngrams.values())
        for ngram in all_ngrams:
            present_in = [d for d in domains if ngram in domain_ngrams[d]]
            row = {"ngram": ngram, "domains": "|".join(present_in), "n_domains": len(present_in)}
            # Attach PMI values per domain
            for d in domains:
                match = next((x for x in domain_results.get(d, []) if x["ngram"] == ngram), None)
                row[f"{d}_pmi"] = match["pmi"] if match else None
            rows.append(row)

        rows.sort(key=lambda x: -x["n_domains"])

        csv_path = os.path.join(out_dir, "cross_domain_patterns.csv")
        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Saved cross-domain pattern comparison to %s", csv_path)

        # Print summary
        shared = [r for r in rows if r["n_domains"] > 1]
        print(f"\nCross-domain shared patterns: {len(shared)} n-grams appear in 2+ domains")
        for r in shared[:10]:
            print(f"  '{r['ngram']}' — domains: {r['domains']}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_results(self, results: List[dict], path: str) -> None:
        if not results:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    def _plot_pmi(self, results: List[dict], domain: str, path: str) -> None:
        if not _MPL_AVAILABLE or not results:
            return
        try:
            ngrams = [r["ngram"] for r in results[:15]]
            pmis = [r["pmi"] for r in results[:15]]

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["#d62728" if p > 0 else "#1f77b4" for p in pmis]
            ax.barh(range(len(ngrams)), pmis, color=colors)
            ax.set_yticks(range(len(ngrams)))
            ax.set_yticklabels(ngrams, fontsize=10)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("PMI (hallucinated vs faithful)")
            ax.set_title(f"Top Hallucination N-gram Patterns — {domain.capitalize()}")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            logger.info("Saved PMI plot to %s", path)
        except Exception as exc:
            logger.warning("Failed to save PMI plot: %s", exc)
