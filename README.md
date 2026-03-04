# WeakRAG-Detect

**Weakly Supervised Hallucination Detection for RAG Outputs**
DSC 253 (Advanced Text Mining) — Course Project

Extends [LegalInsight](https://github.com/kavyasridhar1501/LegalInsight-SelfRAG-HallucinationDetection) (Self-RAG + EigenScore, 74.65% consistency baseline on LegalBench-RAG) with weak supervision, iterative bootstrapping, and multi-domain generalization.

---

## Project Goal

Detect hallucinations across 3 domains using only **20–50 seed labeled examples** per domain:

| Domain      | Dataset         | Examples |
|-------------|-----------------|----------|
| Legal       | LegalBench-RAG  | 6,858    |
| Medical     | PubMedQA        | 1,000    |
| Scientific  | SciQ            | 13,679   |

**Target:** reach 80–90 % of fully supervised performance with just 20–50 seeds.

---

## Architecture

```
Seed Examples (30)
       │
       ▼
┌─────────────────────────────────┐
│   Weak Labeling (Snorkel)       │
│  ├─ EntailmentLF (DeBERTa NLI)  │
│  ├─ SemanticConsistencyLF       │  ← upgraded from LegalInsight
│  └─ ReflectionTokenLF (Self-RAG)│
└─────────────────────────────────┘
       │ soft labels
       ▼
┌─────────────────────────────────┐
│   Bootstrapping                 │
│  ├─ Self-Training               │
│  ├─ Co-Training (2-view)        │
│  └─ Majority Vote Ensemble      │
└─────────────────────────────────┘
       │ labeled pool
       ▼
┌─────────────────────────────────┐
│   DistilBERT Classifier         │
│   (fine-tuned, 512 tokens)      │
└─────────────────────────────────┘
       │
       ▼
  Evaluation + Pattern Mining
```

---

## Directory Structure

```
hallucination_detection/
├── data/
│   ├── raw/                    # downloaded datasets
│   ├── seeds/                  # seed examples (20-50 per domain)
│   ├── processed/              # tokenized/embedded data
│   ├── load_datasets.py        # dataset loaders (LegalBench, PubMedQA, SciQ)
│   └── retrieval.py            # FAISS retriever (ported from LegalInsight)
├── labeling_functions/
│   ├── entailment_lf.py        # DeBERTa NLI-based LF
│   ├── semantic_consistency_lf.py  # upgraded from LegalInsight EigenScore
│   ├── reflection_token_lf.py  # Self-RAG-inspired heuristics
│   └── label_model.py          # Snorkel generative label model
├── bootstrapping/
│   ├── self_training.py        # iterative self-training
│   ├── co_training.py          # two-view co-training
│   └── majority_vote.py        # majority vote ensemble
├── models/
│   └── hallucination_classifier.py  # DistilBERT + LogisticRegression
├── evaluation/
│   ├── metrics.py              # Precision/Recall/F1/AUC-ROC
│   └── label_efficiency.py     # label efficiency curve (key result figure)
├── pattern_mining/
│   ├── ngram_analysis.py       # PMI n-gram mining
│   └── entity_clustering.py    # entity taxonomy builder
├── backend/
│   └── app.py                  # Flask API (extends LegalInsight)
├── experiments/
│   ├── run_legal.py            # legal domain experiment
│   ├── run_medical.py          # medical domain experiment
│   └── run_cross_domain.py     # cross-domain transfer
├── configs/
│   └── config.yaml             # all hyperparameters
├── results/                    # auto-created, stores outputs
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Running Experiments

```bash
cd hallucination_detection

# Legal domain (bootstraps from LegalInsight baseline)
python experiments/run_legal.py

# Medical domain
python experiments/run_medical.py

# Cross-domain transfer (Legal → Medical, Legal → Scientific)
python experiments/run_cross_domain.py
```

---

## Flask API

```bash
cd hallucination_detection
python backend/app.py
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/analyze` | Full analysis (from LegalInsight) |
| `POST` | `/label` | Run all 3 LFs on a single example |
| `POST` | `/bootstrap` | Trigger bootstrapping iteration |
| `GET`  | `/results/<domain>` | Latest evaluation metrics |
| `GET`  | `/taxonomy/<domain>` | Hallucination taxonomy JSON |

**Example `/label` request:**
```json
{
  "question": "Who are the parties to the contract?",
  "context": "This Agreement is between Acme Corp and Beta Ltd.",
  "answer": "The parties are Acme Corp and Beta Ltd."
}
```

---

## Labeling Functions

### 1. EntailmentLF (DeBERTa NLI)
Uses `cross-encoder/nli-deberta-v3-small` to check if the retrieved context entails the generated answer.

### 2. SemanticConsistencyLF (upgraded from LegalInsight)
**Prior system:** measured response *length* variance at temperatures [0.3, 0.5, 0.7]
**New system:** measures *semantic embedding similarity* across paraphrased query variants
Same threshold values (85% / 70%) as LegalInsight for direct comparability.

### 3. ReflectionTokenLF (Self-RAG-inspired)
Simulates Self-RAG's reflection tokens using:
- Named entity overlap (→ [ISREL])
- Numeric consistency (→ [ISSUP])
- Hedging language detection (→ low [ISSUP])
- **ContractNLI misattribution detection** (LegalInsight key insight)

---

## LegalInsight Baseline Comparison

| Metric | LegalInsight | WeakRAG-Detect |
|--------|-------------|----------------|
| Consistency score | 74.65% | TBD (post-experiment) |
| Labeling approach | Manual | Weak supervision (Snorkel) |
| Training examples | N/A | 30 seeds → bootstrapped |
| Domains | Legal only | Legal + Medical + Scientific |

---

## Key Paper Result

The **label efficiency curve** (`results/<domain>/label_efficiency_curve.png`) shows F1 as a function of seed set size, with a horizontal line for the fully supervised baseline and shading indicating the gap to close.

---

## Citation

If you use this code, please also cite:

- LegalInsight (prior system): [github.com/kavyasridhar1501/LegalInsight-SelfRAG-HallucinationDetection](https://github.com/kavyasridhar1501/LegalInsight-SelfRAG-HallucinationDetection)
- Self-RAG: Asai et al. 2023, arXiv 2310.11511
- Snorkel: Ratner et al. 2017, NeurIPS
