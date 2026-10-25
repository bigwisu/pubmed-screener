# Appendix C — Mapping to Reporting Guidance

This study evaluates a literature-screening tool. It makes no clinical assessment and no patient-level prediction, so TRIPOD+AI (Collins et al., BMJ 2024), which was written for clinical prediction models, does not apply to it directly. We used it as a reference for reporting because it is a widely adopted guideline for machine-learning model studies, and many of its items carry over to a benchmark: data sources, evaluation design, threshold selection, performance measures, and model and code availability. Items that concern patients, clinical care or participant eligibility do not carry over and are marked accordingly. TRIPOD+AI guides what a paper reports; it does not judge whether the results are good, and we do not claim full compliance. The table lists the main checklist items, taken from the published checklist table, with our own assessment of where they are addressed. "Reported" means a statement exists in the section named; "partly reported" means the item is addressed incompletely; "not applicable / not reported" covers items that do not fit a benchmark evaluation or that are absent. We assessed the items ourselves; no independent assessor checked the table.

| Item | Topic | Status | Where / remark |
|---|---|---|---|
| 1 | Title | Partly reported | Names model task and design focus; does not use the words development and evaluation. |
| 2 | Abstract | Partly reported | Structured abstract; not written to the TRIPOD+AI abstract checklist. |
| 3, 4 | Background, objectives | Reported | Introduction (research questions RQ1–RQ4). |
| 5 | Data sources | Reported | Methods, Benchmark Dataset; Data and Code Availability. |
| 6 | Participants / eligibility | Partly reported | Selection of the 66 reviews follows Bentegeac et al.; selection criteria are not restated. |
| 7 | Data preparation | Partly reported | Title and abstract format given; handling of the 538 records with abstracts under 100 characters is not described in Methods. |
| 8 | Outcome definition | Reported | Reference label is final review inclusion; limitation discussed. |
| 9 | Predictors (inputs) | Reported | Methods, Model Input. |
| 10 | Sample size | Partly reported | Counts of reviews and inclusions given; no formal sample-size calculation; small calibration set discussed. |
| 11 | Missing data | Partly reported | Records without abstracts are quantified in Appendix B, not handled separately. |
| 12 | Analytical methods | Partly reported | Loss, class weight, threshold rule, metrics and bootstrap given; training hyperparameters are not all listed. |
| 13 | Class imbalance | Reported | Positive class weight and prevalence in both designs. |
| 14 | Fairness | Not applicable / not reported | No patient-level data. |
| 15 | Model output | Reported | Score in [0, 1] and decision rule. |
| 16 | Training vs evaluation | Reported | Training on 103 other reviews; benchmark reviews excluded. |
| 17 | Ethical approval | Reported | Declarations: public bibliographic records only. |
| 18a | Funding | Reported | Declarations: no specific grant. |
| 18b | Conflicts of interest | Reported | Declarations. |
| 18c, 18d | Protocol, registration | Not applicable / not reported | No protocol or registration exists (Section 3.4). |
| 18e, 18f | Data and code sharing | Partly reported | Code and weights public; data terms stated in Data and Code Availability. |
| 19 | Patient and public involvement | Not applicable / not reported | None. |
| 20 | Participants (results) | Partly reported | Reviews, documents and inclusions counted; no per-topic description beyond the appendix. |
| 21 | Model development results | Partly reported | Training-set composition; training curves not reported. |
| 22 | Model specification | Reported | Methods, Decision Model Architecture; checkpoint on Hugging Face. |
| 23 | Model performance | Reported | Results with confidence intervals. |
| 24 | Model updating | Not applicable / not reported | Threshold and model frozen. |
| 25, 26 | Interpretation, limitations | Reported | Discussion. |
| 27 | Usability | Partly reported | Throughput and memory reported; no user study. |
