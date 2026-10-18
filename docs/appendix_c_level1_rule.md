# Appendix B — Level 1 Decision Rule and Checks

**Procedure.** The classification was produced by `benchmark/rq3_mechanical_analysis.py`, run by a single analyst on the primary results database. Re-running the script reproduces `publication/tables/rq3_miss_table.csv` and `publication/tables/rq3_level1_summary.json` exactly. The audit script `benchmark/level1_category_audit.py` reproduces the counts below. No independent labelling and no second rater were used, and no agreement statistic is needed because no judgement is involved.

**Decision rule.** Each inclusion is classified from its own abstract: *short or missing* if the abstract field has fewer than 100 characters, otherwise *abstract present*. A record is a miss if its relevance score is below $\theta^* = 0.028$. No review is assigned to a category, and no list of reviews chosen by the authors is used.

**Counts.** Of the 118 misses, 11 have an abstract of fewer than 100 characters. In 10 of them the abstract field is the literal string "[No Abstract]" (13 characters); the eleventh has 62 characters. The 11 misses occur in 7 reviews (SR 27802478: 5; SR 26868137, 27548070, 27893131, 30409774, 31200992 and 31585960: 1 each). The other 107 misses have an abstract of 100 characters or more. In the whole candidate file, 538 of 15,530 records have an abstract of fewer than 100 characters, 530 of them "[No Abstract]".

**Check on SR 31200992 (Level 2).** A "10 years" or "ten years" expression appears in the title or abstract of 3 of 88 detected inclusions and of none of the 9 missed ones.

**Per-record changes in Level 2.** The table below counts changes by record at θ\* = 0.028, computed from the stored original and counterfactual scores. The net change is reported as Table 6 in the manuscript.

| SR PMID | Misses recovered | New misses | Net recovered |
|---|--:|--:|--:|
| 27802478 | 15 | 3 | 12 |
| 31585960 | 5 | 39 | −34 |
| 31200992 | 8 | 1 | 7 |
| 30409774 | 3 | 4 | −1 |
| 27802505 | 3 | 2 | 1 |
