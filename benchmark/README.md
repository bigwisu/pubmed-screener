# Benchmark Corpus & Evaluation Data

## Benchmark Origin & Terms of Use

The 66 systematic review benchmark evaluation corpus used in this study was compiled and generously shared by **Raphaël Bentegeac and Aghilès Hamroun** (Lille University Hospital / UMR1167 RID-AGE, Pasteur Institute of Lille) for the BibliZap evaluation study:

> **Bentegeac R, Le Guellec B, Leblanc V, Lenain R, Dauchet L, Gauthier V, Gerard E, Chazard E, Amouyel P, Aymes E, Hamroun A.**  
> *BibliZap: an exploratory evaluation of an automated multi-level citation searching tool for systematic and rapid reviews.*  
> Research Synthesis Methods. 2026;17:816–829. doi:10.1017/rsm.2026.10079.

---

## Data Availability Statement

In compliance with data sharing agreements and copyright policies governing third-party literature datasets:

1. **Third-Party Gold Standard Dataset:** The raw inclusion/exclusion mappings and candidate citation lists from the Bentegeac et al. corpus are **available from the original authors upon reasonable request**.
2. **Model Evaluation Code:** All inference engines, calibration protocols, and threshold analysis tools provided in this repository (`screen_with_model.py`, `threshold_analysis.py`, etc.) accept standardized CSV or JSON formatted screening lists and execute fully offline without modification.
3. **Public Replications:** Researchers wishing to evaluate `PubMed-Screener-420M` on open-access benchmark data can also use the open **SYNERGY dataset** (de Bruin et al., 2023, *Data in Brief*) using `scripts/build_synergy_dataset.py`.
