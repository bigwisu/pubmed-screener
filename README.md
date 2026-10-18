# PubMed-Screener-420M (`bigwisu/PubMed-Screener-420M`)

> **Development Notice:** The authors developed and maintained this codebase with IBM Bob as an AI coding assistant.

This repository contains **PubMed-Screener-420M**.
PubMed-Screener-420M is an open-weights Small Language Model for Stage-1 systematic review screening.
The model evaluates candidate title and abstract text against natural language PICO objectives.
A single frozen operating threshold ($\theta^* = 0.028$) classifies candidate citations.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-blue)](https://huggingface.co/bigwisu/PubMed-Screener-420M)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## 1. Repository Structure

```
pubmed-screener/
├── pubmed_screener/                           # Production Python package (Apache-2.0)
│   ├── __init__.py                            # Package entry point and constants
│   ├── screener.py                            # PubMedScreener high-level inference interface
│   └── presets.py                             # Standard PICO prompts and schemas
├── benchmark/                                 # Scientific benchmark replication scripts
│   ├── README.md                              # Benchmark attribution and terms of use
│   ├── threshold_analysis.py                  # Primary statistical evaluation and cluster bootstrap
│   ├── design2_and_supplementary_analysis.py  # Design 2 Boolean deployment evaluation
│   ├── level1_category_audit.py               # Level 1 abstract length error categorization
│   ├── rq3_mechanical_analysis.py             # RQ3 noise-floor score decomposition
│   └── sr_review_objectives.json              # PICO objectives for all 66 benchmark reviews
├── docs/                                      # Technical specifications (ASD-STE100 / Arc42)
│   ├── design1_cross_topic_matrix.md          # Design 1 cross-topic matrix specification
│   ├── design2_boolean_search_deployment.md   # Design 2 Boolean deployment specification
│   ├── appendix_a_per_review_recall.md        # Per-review recall tables
│   ├── appendix_b_level1_rule.md              # Level 1 abstract length decision rule
│   └── appendix_c_reporting_checklist.md     # Reporting standards checklist
├── outputs/                                   # Canonical benchmark scores
│   └── soces_pico_laya_e2e_results.parquet   # Benchmark result table (1,024,980 rows)
├── publication/                               # Summary tables and audit results
│   └── tables/                                # Tabular results matching manuscript
├── scripts/                                   # Upstream data builders
│   └── build_synergy_dataset.py               # Zero-leakage training data builder
├── laya/                                      # Laya submodule fork (bigwisu/laya)
├── screen_with_model.py                       # Standalone CLI screening engine
├── requirements.txt                           # Pinned Python dependencies
├── CITATION.cff                               # Machine-readable academic citation metadata
├── LICENSE                                    # Apache License 2.0
└── README.md                                  # Repository documentation
```

---

## 2. Quickstart: Local Citation Screening

### 2.1 Installation
Clone the repository and install dependencies in a virtual environment:

```bash
git clone --recurse-submodules https://github.com/bigwisu/pubmed-screener.git
cd pubmed-screener

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Programmatic Python API
Use `PubMedScreener` directly in Python scripts:

```python
from pubmed_screener import PubMedScreener

# Load calibrated screener (operational threshold theta* = 0.028)
screener = PubMedScreener()

pico_objective = (
    "Does this study provide potentially relevant evidence on direct oral anticoagulants "
    "versus low-molecular-weight heparin for cancer-associated venous thromboembolism?"
)

title = "Apixaban for the Treatment of Venous Thromboembolism in Patients with Cancer."
abstract = (
    "BACKGROUND: Direct oral anticoagulants are an alternative to low-molecular-weight heparin for "
    "the treatment of venous thromboembolism in patients with cancer. METHODS: In a randomized trial, "
    "we evaluated oral apixaban versus subcutaneous dalteparin..."
)

# Screen citation
result = screener.screen(
    review_objective=pico_objective,
    title=title,
    abstract=abstract
)

print(result)
# {'score': 0.0842, 'decision': 'INCLUDE', 'included': True, 'threshold': 0.028}
```

### 2.3 Standalone CLI Screening
Run screening over candidate lists from the command line:

```bash
python screen_with_model.py --model bigwisu/PubMed-Screener-420M --sr 22422870 --threshold 0.028
```

### 2.4 Real-World Example: Screening a CSV Export from PubMed

A researcher exports search results from PubMed as a CSV with columns `pmid`, `title`, and `abstract`, then screens all records against a PICO objective in one pass:

**`my_papers.csv`** (excerpt):
```
pmid,title,abstract
38901234,"Effect of metformin on HbA1c in type 2 diabetes: a randomised controlled trial","BACKGROUND: Metformin is a first-line agent for type 2 diabetes. METHODS: We randomised 240 adults..."
38812345,"Sodium-glucose cotransporter-2 inhibitors versus metformin for glycaemic control","OBJECTIVE: To compare SGLT-2 inhibitors with metformin as initial therapy in adults with newly diagnosed type 2 diabetes..."
38723456,"Physical activity interventions and cardiovascular outcomes: a systematic review","BACKGROUND: Aerobic exercise reduces cardiovascular risk. METHODS: We searched MEDLINE and Embase..."
```

**`screen_my_papers.py`**:
```python
import csv
from pubmed_screener import PubMedScreener

# 1. Define your PICO screening question
pico_objective = (
    "Does this study provide potentially relevant evidence on metformin versus "
    "SGLT-2 inhibitors for glycaemic control in adults with type 2 diabetes?"
)

# 2. Load papers from CSV
with open("my_papers.csv", newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))
# records = [{"pmid": "...", "title": "...", "abstract": "..."}, ...]

# 3. Load screener (downloads weights from HF on first run, ~843 MB cached locally)
screener = PubMedScreener()

# 4. Screen all records
results = screener.screen_batch(pico_objective, records)

# 5. Print and save candidates
included = [r for r in results if r["included"]]
print(f"{len(included)} / {len(results)} papers flagged for full-text review\n")

with open("candidates.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["pmid", "score", "decision", "threshold"])
    writer.writeheader()
    writer.writerows(included)
```

Run it:
```bash
python screen_my_papers.py
# 2 / 3 papers flagged for full-text review
```

**`candidates.csv`** output:
```
pmid,score,decision,threshold
38901234,0.7431,INCLUDE,0.028
38812345,0.6892,INCLUDE,0.028
```

The model runs entirely on-device (Apple MPS on Apple Silicon, CPU fallback otherwise). No data leaves the machine.

---

## 3. Scientific Reproducibility Matrix

The table below maps manuscript findings to replication scripts and outputs:

| Manuscript Item | Description | Replication Command | Output Location |
|---|---|---|---|
| **Table 2 / RQ1** | Operating point and bootstrap CIs | `python benchmark/threshold_analysis.py` | CLI standard output |
| **Table 3 / RQ2** | Dual-design comparison (Design 1 vs 2) | `python benchmark/design2_and_supplementary_analysis.py` | `publication/tables/phase1_summary.json` |
| **Table 4–5 / RQ3** | Level 1 abstract length error audit | `python benchmark/level1_category_audit.py` | `publication/tables/rq3_level1_summary.json` |
| **RQ3 Mechanics** | Noise-floor score decomposition | `python benchmark/rq3_mechanical_analysis.py` | `publication/tables/rq3_miss_table.csv` |
| **Appendix A** | Per-review recall (62 unseen reviews) | `python benchmark/design2_and_supplementary_analysis.py` | `publication/tables/per_review_recall.csv` |

---

## 4. Model Training and Fine-Tuning

Users CAN reproduce model training from base weights.
Training uses the bundled `laya` submodule fork with positive-class weighting (`--pos-weight 8.0`).

### 4.1 Build Training Data from SYNERGY
Construct training pairs from the open SYNERGY dataset:

```bash
pip install synergy-dataset
python scripts/build_synergy_dataset.py --out data/train_pico_synergy.jsonl
```

### 4.2 Fine-Tune Model Weights
Execute fine-tuning with positive-class loss weighting:

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

## 5. Benchmark Results and Data Schemas

### 5.1 Canonical Result File
The file [`outputs/soces_pico_laya_e2e_results.parquet`](outputs/soces_pico_laya_e2e_results.parquet) contains all benchmark scores.
The table stores 1,024,980 rows across 66 systematic reviews.
Table columns:
- `sr_pmid`: Systematic review PubMed identifier.
- `pmid`: Candidate citation PubMed identifier.
- `relevance_score`: Continuous model decision score.
- `confidence`: Model prediction confidence.
- `predicted_include`: Binary decision at threshold $\theta^* = 0.028$.
- `is_inclusion`: Reference systematic review inclusion label.

### 5.2 Legacy Identifier Notice
The file prefix `soces_pico_laya_e2e` is a legacy internal run identifier.
Development logs used this identifier for PubMed-Screener-420M.
All data rows match the published model weights.

### 5.3 Calibration and Test Splits
Four reviews define the calibration set: `22226047`, `22323502`, `22422870`, and `22777524`.
The remaining 62 systematic reviews form the primary test set.

```python
import pandas as pd
df = pd.read_parquet("outputs/soces_pico_laya_e2e_results.parquet")
```

---

## 6. Technical Specifications

Detailed Arc42 specifications in ASD-STE100 standard:
- [`docs/design1_cross_topic_matrix.md`](docs/design1_cross_topic_matrix.md): Design 1 cross-topic matrix evaluation.
- [`docs/design2_boolean_search_deployment.md`](docs/design2_boolean_search_deployment.md): Design 2 Boolean search deployment evaluation.

---

## 7. Attribution and Citation

PubMed-Screener-420M builds upon open-source software:
- **Laya (`convaiinnovations/laya`):** Propositional decision engine from ConvAI Innovations.
- **ModernBERT (`answerdotai/ModernBERT-large`):** Transformer encoder from Answer.ai and LightOn.

```bibtex
@article{pubmed_screener_420m_2026,
  title={Calibrated Small Language Models for Stage-1 Systematic Review Screening: Frozen Thresholds, Noise-Floor Mechanics, and Sovereign Edge Execution},
  author={Suntoyo et al.},
  journal={Journal of Biomedical Informatics},
  year={2026},
  note={Under Review}
}
```
