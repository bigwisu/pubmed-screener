# Pre-Submission Calibration and Thresholding Protocol

**Model:** PubMed-Screener-420M (`bigwisu/PubMed-Screener-420M`, checkpoint: `soces_pico_laya_e2e`)
**Base Architecture:** ModernBERT-large encoder (421M parameters) with a non-autoregressive binary decision head  
**Target Application:** Title and abstract citation screening for systematic reviews  
**Document Status:** Pre-submission protocol specification for TRIPOD+AI and PRISMA-trAIce adherence  

---

## 1. Overview and Objective

In systematic review citation screening, automated decision models require an explicit operational decision threshold ($\theta^*$) to convert continuous model probability scores $s \in [0, 1]$ into binary screening decisions:
$$\hat{y} = \mathbb{I}(s \ge \theta^*)$$
where $\hat{y}=1$ indicates that the citation is retained for full-text review, and $\hat{y}=0$ indicates exclusion.

The objective of this calibration protocol is to establish $\theta^*$ **a priori** using a strictly held-out calibration set of systematic reviews before evaluating the primary validation benchmark, preventing review-specific threshold tuning or data leakage.

---

## 2. Held-Out Calibration Dataset

Four systematic reviews from the SYNERGY benchmark were selected as the dedicated calibration set:
1. **PMID 22226047:** Screening for selective serotonin reuptake inhibitors in pediatric depression.
2. **PMID 22323502:** Interventions for preventing occupational injuries in healthcare workers.
3. **PMID 22422870:** Pharmacological treatments for neuropathic pain.
4. **PMID 22777524:** Diagnostic accuracy of rapid antigen detection tests for group A streptococcal pharyngitis.

### Calibration Data Summary
- **Total Systematic Reviews:** 4
- **Candidate Pool Size:** 15,530 unique PubMed documents (union of all benchmark Boolean searches)
- **Total Calibration Evaluations:** $4 \times 15,530 = 62,120$ (review, document) screening evaluations
- **True Inclusions (Positives):** $N_{\text{pos}} = 95$
- **True Exclusions (Negatives):** $N_{\text{neg}} = 62,025$
- **Calibration Empirical Prevalence:** $\pi_{\text{cal}} = 95 / 62,120 = 0.153\%$

---

## 3. Pre-Specified Decision Rule for Threshold Selection

### 3.1 Primary Optimization Objective
Systematic review methodology prioritizes recall (sensitivity) over precision to minimize the omission of eligible clinical evidence. The pre-specified operational rule is:

$$\theta^* = \max \left\{ \theta \in [0.001, 0.500] \;\middle|\; \text{Recall}_{\text{pooled}}(\theta; \mathcal{D}_{\text{cal}}) \ge 95.0\% \right\}$$

subject to:
$$\text{Specificity}_{\text{pooled}}(\theta; \mathcal{D}_{\text{cal}}) \ge 90.0\%$$

where:
$$\text{Recall}_{\text{pooled}}(\theta) = \frac{\sum_{i \in \text{cal}} \text{TP}_i(\theta)}{\sum_{i \in \text{cal}} (\text{TP}_i(\theta) + \text{FN}_i(\theta))}$$

### 3.2 Discrete Threshold Grid Search on Calibration Data
Evaluating the candidate thresholds across $\theta \in [0.020, 0.035]$ on the 62,120 calibration evaluations yielded:

| Candidate Threshold ($\theta$) | True Positives (TP / 95) | False Negatives (FN / 95) | Pooled Recall (%) | Pooled Specificity (%) | Workload Retained (%) |
|---|---|---|---|---|---|
| 0.020 | 95 | 0 | 100.00% | 8.12% | 91.88% |
| 0.022 | 95 | 0 | 100.00% | 29.41% | 70.59% |
| 0.024 | 95 | 0 | 100.00% | 71.05% | 28.95% |
| 0.025 | 95 | 0 | 100.00% | 83.12% | 16.88% |
| 0.026 | 95 | 0 | 100.00% | 90.54% | 9.46% |
| 0.027 | 95 | 0 | 100.00% | 94.10% | 5.90% |
| **0.028 ($\theta^*$)** | **94** | **1** | **98.95%** | **96.58%** | **3.42%** |
| 0.029 | 89 | 6 | 93.68% | 97.80% | 2.20% |
| 0.030 | 85 | 10 | 89.47% | 98.45% | 1.55% |
| 0.035 | 71 | 24 | 74.74% | 99.41% | 0.59% |

### 3.3 Selected Operating Threshold
- **Selected Value:** $\theta^* = 0.028$
- **Calibration Performance:**
  - Pooled Recall: $94 / 95 = 98.95\%$ (exceeds $95.0\%$ target)
  - Pooled Specificity: $59,903 / 62,025 = 96.58\%$ (exceeds $90.0\%$ target)
  - Calibration Workload: $3.42\%$ (retains $2,122 / 62,120$ records)

---

## 4. Frozen Reporting Contract for External Validation

1. **Parameter Immutability:** $\theta^* = 0.028$ was frozen and committed to version control prior to analyzing the 62 unseen primary systematic reviews ($962,860$ evaluations).
2. **No Post-Hoc Tuning:** $\theta^*$ was not adjusted per review, per clinical domain, or post-hoc after inspecting external validation results.
3. **Primary Reporting Separation:** Primary validation results are reported strictly on the $62$ non-calibration reviews ($2,431$ inclusions). Secondary results are reported on the pooled set of all $66$ reviews to confirm that calibration inclusion does not introduce estimation bias.
4. **Dual Evaluation Reporting:**
   - *Cross-Topic Design:* Evaluated on all $962,860$ pairs ($0.25\%$ prevalence) to measure ranking integrity under extreme class imbalance.
   - *Boolean-Search Design:* Evaluated strictly on the $16,598$ topic-relevant pairs ($16.4\%$ prevalence) to reflect real-world workload reduction ($77.6\%$ mean workload, $28.2\%$ specificity, $10.4\%$ WSS@95).

---

## 5. Software Implementation and Verification

The thresholding logic is implemented in [`benchmark/threshold_analysis.py`](threshold_analysis.py:1) and executed against [`outputs/soces_pico_laya_e2e_results.sqlite`](outputs/soces_pico_laya_e2e_results.sqlite). The implementation verifies that:
1. Every candidate document is scored independently with a floating-point score $s \in [0, 1]$.
2. $\hat{y} = 1$ if $s \ge 0.028$ else $0$.
3. All metrics (Recall, Specificity, Workload, Work Saved over Sampling) are calculated strictly with systematic review clustered bootstrapping (5,000 resamples).
