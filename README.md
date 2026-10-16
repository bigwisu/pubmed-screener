# PubMed-Screener-420M (`bigwisu/PubMed-Screener-420M`)

Official open-source repository for **PubMed-Screener-420M**, an open-weights Small Language Model (SLM) decision architecture developed and calibrated for high-recall title and abstract citation screening in biomedical systematic reviews.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-blue)](https://huggingface.co/bigwisu/PubMed-Screener-420M)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Key Features

- **Compact Parameter Footprint:** 421M parameters (843 MB on disk), combining a ModernBERT-large backbone (`answerdotai/ModernBERT-large`) with a task-adapted propositional decision head (`convaiinnovations/laya`).
- **Clinically Calibrated Operating Point:** Calibrated *a priori* ($\theta^* = 0.028$) on 4 held-out systematic reviews, achieving **95.15% pooled recall** [95% CI: 92.69%–97.12%] across 62 unseen primary reviews ($962,860$ evaluations).
- **100% Offline & Sovereign Inference:** Executes entirely on local hardware without cloud API calls, preserving confidential research protocols.
- **Consumer Hardware Friendly:** Processes 2.4 citations/s on an Apple M3 MacBook Air (107 minutes for a full 15,530-citation search catalog) with only 2.1 GB peak RAM.

---

## Installation

```bash
git clone --recurse-submodules https://github.com/bigwisu/pubmed-screener.git
cd pubmed-screener

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart: Screening Citations

### 1. Download Model Weights
Model weights and configurations are hosted on Hugging Face Hub:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bigwisu/PubMed-Screener-420M",
    local_dir="models/pubmed_screener_420m"
)
```

### 2. Run Inference

```python
from laya import load

# Load model checkpoint
model = load("models/pubmed_screener_420m")

review_objective = (
    "Systematic review evaluating the efficacy and safety of direct oral anticoagulants "
    "versus low-molecular-weight heparin for the treatment of cancer-associated venous thromboembolism."
)

title = "Apixaban for the Treatment of Venous Thromboembolism in Patients with Cancer."
abstract = (
    "BACKGROUND: Direct oral anticoagulants are an alternative to low-molecular-weight heparin for "
    "the treatment of venous thromboembolism in patients with cancer. METHODS: In a randomized trial, "
    "we evaluated oral apixaban versus subcutaneous dalteparin..."
)

# Predict relevance score (threshold theta* = 0.028)
score = model.predict(
    review_objective=review_objective,
    title=title,
    abstract=abstract
)

print(f"Relevance Score: {score:.4f}")
print(f"Decision: {'INCLUDE (Proceed to full-text)' if score >= 0.028 else 'EXCLUDE'}")
```

---

## Evaluation & Benchmark Reproduction

This repository contains evaluation scripts and calibration protocols:

- `calibration_protocol.md`: Detailed specification of the a priori threshold selection on 4 held-out calibration reviews.
- `threshold_analysis.py`: Threshold sweeping, ROC-AUC, PR-AUC, and systematic-review clustered bootstrap CI calculations.
- `rq3_mechanical_analysis.py`: Structured error analysis and false-negative categorization.
- `generate_figures.py`: Publication-grade visualization pipeline generating Figures 1–4.

To run threshold analysis:
```bash
python threshold_analysis.py --db outputs/results.sqlite --out publication/tables
```

---

## Citation & Attribution

This model is built upon the open-source foundations of **Laya** (`convaiinnovations/laya`) and **ModernBERT** (`answerdotai/ModernBERT-large`).

If you use PubMed-Screener-420M in your research, please cite:

```bibtex
@article{pubmed_screener_420m_2026,
  title={Offline AI Decision Models for PubMed Systematic Review Screening: External Validation of a Calibrated Open-Weights Architecture Across 62 Reviews},
  author={Wisu et al.},
  journal={Journal of Biomedical Informatics},
  year={2026},
  note={Under Review}
}
```
