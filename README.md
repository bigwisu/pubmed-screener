# PubMed-Screener-420M (`bigwisu/PubMed-Screener-420M`)

Official repository for **PubMed-Screener-420M**, an open-weights Small Language Model (SLM) decision architecture developed and calibrated for high-recall title and abstract citation screening in biomedical systematic reviews.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-blue)](https://huggingface.co/bigwisu/PubMed-Screener-420M)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Repository Structure

```
pubmed-screener/
├── benchmark/
│   └── README.md                     # Benchmark attribution & dataset availability statement
├── laya/                             # Submodule (fork: github.com/bigwisu/laya of convaiinnovations/laya)
├── scripts/
│   └── build_synergy_dataset.py      # Script to rebuild zero-leakage training data from SYNERGY
├── screen_with_model.py              # Screening engine for local decision models
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Quickstart: Screening Citations with Local Weights

### 1. Installation
```bash
git clone --recurse-submodules https://github.com/bigwisu/pubmed-screener.git
cd pubmed-screener

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Model Weights
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bigwisu/PubMed-Screener-420M",
    local_dir="models/pubmed_screener_420m"
)
```

### 3. Run Inference
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

## Rebuilding and Fine-Tuning the Model

Researchers can fine-tune and train PubMed-Screener-420M from base weights using the bundled `laya` submodule fork with positive class weighting (`--pos-weight 8.0`):

### 1. Rebuild Zero-Leakage Training Data from SYNERGY
```bash
pip install synergy-dataset
python scripts/build_synergy_dataset.py --out data/train_pico_synergy.jsonl
```

### 2. Fine-Tune with the Laya CLI
The fork adds `--pos-weight` to counterbalance positive-class sparsity in systematic review literature screening:

```bash
python -m laya.train_cli \
    --data data/train_pico_synergy.jsonl \
    --model-dir convaiinnovations/laya \
    --output-dir models/pubmed_screener_420m \
    --epochs 3 \
    --batch-size 16 \
    --grad-accum 4 \
    --lr 2e-5 \
    --head-lr 1e-4 \
    --pos-weight 8.0 \
    --loss soft-ce
```

---

## Benchmark Data Availability

The 66-review benchmark corpus was compiled for the BibliZap evaluation study (Bentegeac et al., 2026, *Research Synthesis Methods*, doi:10.1017/rsm.2026.10079) and is **available from the original authors upon reasonable request**. See [`benchmark/README.md`](benchmark/README.md) for details.

---

## Attribution & Citation

PubMed-Screener-420M was developed on the open-source foundations of:
- **Laya (`convaiinnovations/laya`):** Multilingual, non-autoregressive System-1 decision engine evaluating typed propositions in single forward passes, developed with reinforcement learning against strictly proper scoring rules (RLCD) by ConvAI Innovations. The `laya/` submodule in this repository is maintained from our customized fork ([`bigwisu/laya`](https://github.com/bigwisu/laya)).
- **ModernBERT (`answerdotai/ModernBERT-large`):** Modern contextual transformer encoder developed by Answer.ai and LightOn.

```bibtex
@article{pubmed_screener_420m_2026,
  title={Offline AI Decision Models for PubMed Systematic Review Screening: External Validation of a Calibrated Open-Weights Architecture Across 62 Reviews},
  author={Suntoyo et al.},
  journal={Journal of Biomedical Informatics},
  year={2026},
  note={Under Review}
}
```
