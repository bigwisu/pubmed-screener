# Benchmark Corpus and Evaluation Data

## 1. Benchmark Provenance

Bentegeac et al. (2026) compiled the 66 systematic review benchmark evaluation corpus for the BibliZap study:

> **Bentegeac R, Le Guellec B, Leblanc V, Lenain R, Dauchet L, Gauthier V, Gerard E, Chazard E, Amouyel P, Aymes E, Hamroun A.**  
> *BibliZap: an exploratory evaluation of an automated multi-level citation searching tool for systematic and rapid reviews.*  
> Research Synthesis Methods. 2026;17:816–829. doi:10.1017/rsm.2026.10079.

---

## 2. Data Availability Statement

Data sharing complies with third-party copyright terms and data distribution agreements:

1. **Third-Party Evaluation Corpus:** The raw inclusion lists and candidate citation lists are available from the original authors upon request.
2. **Offline Evaluation Pipelines:** The evaluation scripts (`screen_with_model.py`, `threshold_analysis.py`) execute fully offline on local systems.
3. **Withheld Inputs:** The analysis scripts read `benchmark_candidates.parquet`, `pubmed_documents.parquet` and `sr_document_mappings.parquet`. These files derive from the third-party corpus and are not distributed here. Obtain them from the original authors and place them in `benchmark/`.
4. **Open Training Replications:** Users MAY rebuild open training sets from the SYNERGY corpus using `scripts/build_synergy_dataset.py`.
