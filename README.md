# WeakRAG-Detect

**Weakly Supervised Hallucination Detection for RAG Outputs**
DSC 253 (Advanced Text Mining) — Course Project

---

## Introduction & Methodology

This project extends [LegalInsight](https://github.com/kavyasridhar1501/LegalInsight-SelfRAG-HallucinationDetection) (Self-RAG + EigenScore, 74.65% consistency baseline on LegalBench-RAG) with weak supervision, iterative bootstrapping, and multi-domain generalization.

The goal is to detect hallucinations across 3 domains — Legal, Medical, and Scientific — using only **30 seed labeled examples** per domain, targeting 80–90% of fully supervised performance.

### Pipeline

```
Seed Examples (30)
       │
       ▼
┌─────────────────────────────────┐
│   Weak Labeling (Snorkel)       │
│  ├─ EntailmentLF (DeBERTa NLI)  │
│  ├─ SemanticConsistencyLF       │
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

Three labeling functions drive weak supervision:

- **EntailmentLF** — uses `cross-encoder/nli-deberta-v3-small` to check if the retrieved context entails the generated answer.
- **SemanticConsistencyLF** — measures semantic embedding similarity across paraphrased query variants (upgraded from LegalInsight's length-variance approach).
- **ReflectionTokenLF** — simulates Self-RAG reflection tokens via named entity overlap, numeric consistency checks, and hedging language detection.

---

## How to Run

**Install dependencies:**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Run experiments** (from the project root):

```bash
# Legal domain
python -m hallucination_detection.experiments.run_legal

# Medical domain
python -m hallucination_detection.experiments.run_medical

# Cross-domain transfer (Legal → Medical, Legal → Scientific)
python -m hallucination_detection.experiments.run_cross_domain
```

---

## Results

Results are saved automatically after each experiment run:

| Domain | Output Path |
|--------|-------------|
| Legal | `hallucination_detection/results/legal/` |
| Medical | `hallucination_detection/results/medical/` |
| Scientific | `hallucination_detection/results/scientific/` |

Each folder contains:
- `results_summary.json` — precision, recall, F1, and accuracy per method
- `label_efficiency_curve.png` — F1 vs. seed set size compared to the fully supervised baseline

**Summary of key findings:**

| Domain | Best Method | F1 | Accuracy |
|--------|-------------|-----|----------|
| Legal | Co-Training | 0.850 | 0.905 |
| Medical | Legal→Medical (few-shot n=30) | 0.595 | 0.455 |
| Scientific | Legal→Scientific (few-shot n=30) | 0.653 | 0.565 |

Co-Training with 30 seeds achieves 90.5% accuracy on Legal, within ~10% of fully supervised. Cross-domain transfer requires few-shot adaptation — zero-shot transfer collapses (F1 ~0.10 on Scientific).
