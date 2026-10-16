#!/usr/bin/env python3
"""
benchmark/threshold_analysis.py — Statistical Evaluation & Threshold Analysis
for SOCES-PubMed Decision Models across Benchmark Systematic Reviews.

Computes:
1. Primary (non-calibration SRs) and Secondary (all SRs) splits.
2. Operating point performance at fixed reporting parameter theta* = 0.028:
   - Pooled & Macro Recall, Specificity, Precision, F1, F2, MCC.
   - Macro PR-AUC and ROC-AUC per SR.
   - Rank-based WSS@95 and WSS@100 per SR.
   - Cluster-bootstrap (SR-level) 95% Confidence Intervals (5,000 resamples).
3. Empirical Threshold Sweep (0.020 to 0.040):
   - Recall, Specificity, Workload reduction curves.
4. Post-hoc Optimum Analysis:
   - Maximum threshold for Recall >= 95%
   - Maximum threshold for Recall >= 98%
   - Youden's J statistic index max(Sens + Spec - 1)
   - Cluster-bootstrap distribution / range of optimal thresholds.

Usage:
  ./venv/bin/python benchmark/threshold_analysis.py
  ./venv/bin/python benchmark/threshold_analysis.py --db outputs/soces_pico_laya_e2e_results.sqlite --theta 0.028
"""

import os
import sys
import sqlite3
import argparse
import numpy as np
import pandas as pd

CALIBRATION_SRS = ["22226047", "22323502", "22422870", "22777524"]


def parse_args():
    parser = argparse.ArgumentParser(description="Threshold and statistical analysis on SQLite screening results")
    parser.add_argument("--db", type=str, default="outputs/soces_pico_laya_e2e_results.sqlite", help="Path to SQLite results DB")
    parser.add_argument("--theta", type=float, default=0.028, help="Pre-specified reporting threshold theta*")
    parser.add_argument("--bootstraps", type=int, default=5000, help="Number of SR-clustered bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap reproducibility")
    return parser.parse_args()


def load_data(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}", file=sys.stderr)
        sys.exit(1)
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT sr_pmid, pmid, relevance_score, is_inclusion FROM screening_results", conn)
    conn.close()
    
    df["sr_pmid"] = df["sr_pmid"].astype(str)
    df["is_inclusion"] = df["is_inclusion"].astype(int)
    df["relevance_score"] = df["relevance_score"].astype(float)
    return df


def compute_roc_auc_np(y_true, scores):
    """Fast ROC-AUC via rank sum."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    
    # Fast tie correction
    s_sorted = scores[order]
    dups = np.where(s_sorted[:-1] == s_sorted[1:])[0]
    if len(dups) > 0:
        unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
        tie_indices = np.where(counts > 1)[0]
        for i in tie_indices:
            mask = (inverse == i)
            ranks[mask] = np.mean(ranks[mask])
            
    pos_ranks = ranks[y_true == 1]
    rank_sum = np.sum(pos_ranks)
    u_stat = rank_sum - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def compute_pr_auc_np(y_true, scores):
    """Compute Average Precision (PR-AUC) from sorted scores."""
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return np.nan
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    precisions = cum_tp / ranks
    pos_precisions = precisions[y_sorted == 1]
    return float(np.sum(pos_precisions) / n_pos)


class SRData:
    """Precomputed summary container for a single systematic review."""
    def __init__(self, sr_pmid, y_true, scores):
        self.sr_pmid = sr_pmid
        self.y_true = np.asarray(y_true, dtype=int)
        self.scores = np.asarray(scores, dtype=float)
        self.n_total = len(self.y_true)
        self.n_pos = int(np.sum(self.y_true))
        self.n_neg = self.n_total - self.n_pos
        
        # Pre-extract positive and negative scores
        self.pos_scores = self.scores[self.y_true == 1]
        self.neg_scores = self.scores[self.y_true == 0]
        
        # Threshold-independent rank metrics
        self.roc_auc = compute_roc_auc_np(self.y_true, self.scores)
        self.pr_auc = compute_pr_auc_np(self.y_true, self.scores)
        
        # WSS@95 and WSS@100
        order = np.argsort(-self.scores)
        y_sorted = self.y_true[order]
        cum_pos = np.cumsum(y_sorted)
        target_95 = int(np.ceil(0.95 * self.n_pos)) if self.n_pos > 0 else 0
        idx_95 = np.where(cum_pos >= target_95)[0]
        idx_100 = np.where(cum_pos >= self.n_pos)[0]
        
        if len(idx_95) > 0 and self.n_pos > 0:
            k_95 = idx_95[0] + 1
            rec_95 = cum_pos[idx_95[0]] / self.n_pos
            self.wss_95 = (self.n_total - k_95) / self.n_total - (1.0 - rec_95)
        else:
            self.wss_95 = np.nan
            
        if len(idx_100) > 0 and self.n_pos > 0:
            k_100 = idx_100[0] + 1
            rec_100 = cum_pos[idx_100[0]] / self.n_pos
            self.wss_100 = (self.n_total - k_100) / self.n_total - (1.0 - rec_100)
        else:
            self.wss_100 = np.nan

    def eval_threshold(self, theta):
        tp = int(np.sum(self.pos_scores >= theta))
        fn = self.n_pos - tp
        fp = int(np.sum(self.neg_scores >= theta))
        tn = self.n_neg - fp
        recall = tp / self.n_pos if self.n_pos > 0 else 0.0
        specificity = tn / self.n_neg if self.n_neg > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
        return {
            "sr_pmid": self.sr_pmid,
            "n_total": self.n_total,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "recall": recall,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "f2": f2,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "wss_95": self.wss_95,
            "wss_100": self.wss_100,
        }


def compute_group_metrics(sr_objs, theta):
    """Compute pooled and macro summary metrics across a list of SRData objects."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_pos = 0
    total_neg = 0
    total_docs = 0
    
    recalls = []
    specificities = []
    precisions = []
    f1s = []
    f2s = []
    roc_aucs = []
    pr_aucs = []
    wss_95s = []
    wss_100s = []
    sr_results = {}
    
    for sr in sr_objs:
        m = sr.eval_threshold(theta)
        sr_results[sr.sr_pmid] = m
        
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
        total_tn += m["tn"]
        total_pos += m["n_pos"]
        total_neg += m["n_neg"]
        total_docs += m["n_total"]
        
        recalls.append(m["recall"])
        specificities.append(m["specificity"])
        precisions.append(m["precision"])
        f1s.append(m["f1"])
        f2s.append(m["f2"])
        if not np.isnan(m["roc_auc"]):
            roc_aucs.append(m["roc_auc"])
        if not np.isnan(m["pr_auc"]):
            pr_aucs.append(m["pr_auc"])
        if not np.isnan(m["wss_95"]):
            wss_95s.append(m["wss_95"])
        if not np.isnan(m["wss_100"]):
            wss_100s.append(m["wss_100"])
            
    pooled_recall = total_tp / total_pos if total_pos > 0 else 0.0
    pooled_specificity = total_tn / total_neg if total_neg > 0 else 0.0
    pooled_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    pooled_f1 = 2 * pooled_precision * pooled_recall / (pooled_precision + pooled_recall) if (pooled_precision + pooled_recall) > 0 else 0.0
    pooled_f2 = 5 * pooled_precision * pooled_recall / (4 * pooled_precision + pooled_recall) if (4 * pooled_precision + pooled_recall) > 0 else 0.0
    
    denom = np.sqrt(float(total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn))
    pooled_mcc = (total_tp * total_tn - total_fp * total_fn) / denom if denom > 0 else 0.0
    
    return {
        "n_srs": len(sr_objs),
        "total_docs": total_docs,
        "total_pos": total_pos,
        "total_neg": total_neg,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
        "pooled_recall": pooled_recall,
        "pooled_specificity": pooled_specificity,
        "pooled_precision": pooled_precision,
        "pooled_f1": pooled_f1,
        "pooled_f2": pooled_f2,
        "pooled_mcc": pooled_mcc,
        "macro_recall_mean": np.mean(recalls),
        "macro_recall_median": np.median(recalls),
        "macro_recall_iqr": (np.percentile(recalls, 25), np.percentile(recalls, 75)),
        "macro_recall_min": np.min(recalls),
        "macro_recall_max": np.max(recalls),
        "macro_specificity_mean": np.mean(specificities),
        "macro_specificity_median": np.median(specificities),
        "macro_roc_auc_mean": np.mean(roc_aucs) if roc_aucs else np.nan,
        "macro_roc_auc_median": np.median(roc_aucs) if roc_aucs else np.nan,
        "macro_pr_auc_mean": np.mean(pr_aucs) if pr_aucs else np.nan,
        "macro_pr_auc_median": np.median(pr_aucs) if pr_aucs else np.nan,
        "macro_pr_auc_min": np.min(pr_aucs) if pr_aucs else np.nan,
        "macro_pr_auc_max": np.max(pr_aucs) if pr_aucs else np.nan,
        "macro_wss_95_mean": np.mean(wss_95s) if wss_95s else np.nan,
        "macro_wss_95_median": np.median(wss_95s) if wss_95s else np.nan,
        "macro_wss_100_mean": np.mean(wss_100s) if wss_100s else np.nan,
        "macro_wss_100_median": np.median(wss_100s) if wss_100s else np.nan,
        "pct_srs_recall_ge_95": np.mean([r >= 0.95 for r in recalls]) * 100,
        "pct_srs_recall_100": np.mean([r == 1.0 for r in recalls]) * 100,
        "sr_results": sr_results,
    }


def cluster_bootstrap(sr_objs, theta, n_bootstraps=5000, seed=42):
    """Fast SR-clustered bootstrap."""
    rng = np.random.default_rng(seed)
    n_srs = len(sr_objs)
    
    # Pre-evaluate all SRs at theta
    evals = [sr.eval_threshold(theta) for sr in sr_objs]
    
    tps = np.array([e["tp"] for e in evals])
    fps = np.array([e["fp"] for e in evals])
    fns = np.array([e["fn"] for e in evals])
    tns = np.array([e["tn"] for e in evals])
    poss = np.array([e["n_pos"] for e in evals])
    negs = np.array([e["n_neg"] for e in evals])
    recs = np.array([e["recall"] for e in evals])
    specs = np.array([e["specificity"] for e in evals])
    praucs = np.array([e["pr_auc"] for e in evals])
    wss95s = np.array([e["wss_95"] for e in evals])
    wss100s = np.array([e["wss_100"] for e in evals])
    
    boot_indices = rng.integers(0, n_srs, size=(n_bootstraps, n_srs))
    
    sample_tp = np.sum(tps[boot_indices], axis=1)
    sample_fp = np.sum(fps[boot_indices], axis=1)
    sample_fn = np.sum(fns[boot_indices], axis=1)
    sample_tn = np.sum(tns[boot_indices], axis=1)
    sample_pos = np.sum(poss[boot_indices], axis=1)
    sample_neg = np.sum(negs[boot_indices], axis=1)
    
    boot_pooled_rec = sample_tp / np.maximum(sample_pos, 1)
    boot_pooled_spec = sample_tn / np.maximum(sample_neg, 1)
    boot_pooled_prec = sample_tp / np.maximum(sample_tp + sample_fp, 1)
    
    boot_macro_rec = np.mean(recs[boot_indices], axis=1)
    boot_macro_spec = np.mean(specs[boot_indices], axis=1)
    boot_macro_pr_auc = np.mean(praucs[boot_indices], axis=1)
    boot_macro_wss95 = np.mean(wss95s[boot_indices], axis=1)
    boot_macro_wss100 = np.mean(wss100s[boot_indices], axis=1)
    
    def ci(arr):
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        
    return {
        "pooled_recall_ci": ci(boot_pooled_rec),
        "pooled_specificity_ci": ci(boot_pooled_spec),
        "pooled_precision_ci": ci(boot_pooled_prec),
        "macro_recall_ci": ci(boot_macro_rec),
        "macro_specificity_ci": ci(boot_macro_spec),
        "macro_pr_auc_ci": ci(boot_macro_pr_auc),
        "macro_wss_95_ci": ci(boot_macro_wss95),
        "macro_wss_100_ci": ci(boot_macro_wss100),
    }


def threshold_sweep(sr_objs, theta_min=0.020, theta_max=0.040, step=0.001):
    thetas = np.arange(theta_min, theta_max + 1e-6, step)
    records = []
    
    for t in thetas:
        t_val = round(float(t), 4)
        m = compute_group_metrics(sr_objs, t_val)
        records.append({
            "theta": t_val,
            "pooled_recall": m["pooled_recall"],
            "pooled_specificity": m["pooled_specificity"],
            "pooled_precision": m["pooled_precision"],
            "pooled_f1": m["pooled_f1"],
            "pooled_f2": m["pooled_f2"],
            "pooled_mcc": m["pooled_mcc"],
            "macro_recall": m["macro_recall_mean"],
            "macro_specificity": m["macro_specificity_mean"],
            "youden_j": m["pooled_recall"] + m["pooled_specificity"] - 1.0,
            "workload_pct": (m["total_tp"] + m["total_fp"]) / m["total_docs"] * 100,
        })
    return pd.DataFrame(records)


def post_hoc_optima_with_bootstrap(sr_objs, n_bootstraps=2000, seed=42):
    """Compute post-hoc optimal thresholds and their bootstrap stability."""
    thetas = np.round(np.arange(0.015, 0.045 + 1e-6, 0.0005), 4)
    n_srs = len(sr_objs)
    
    # Pre-evaluate TP and FP matrices: shape [n_srs, n_thetas]
    tp_mat = np.zeros((n_srs, len(thetas)), dtype=int)
    fp_mat = np.zeros((n_srs, len(thetas)), dtype=int)
    pos_arr = np.array([sr.n_pos for sr in sr_objs], dtype=int)
    neg_arr = np.array([sr.n_neg for sr in sr_objs], dtype=int)
    
    for i, sr in enumerate(sr_objs):
        for j, t in enumerate(thetas):
            tp_mat[i, j] = int(np.sum(sr.pos_scores >= t))
            fp_mat[i, j] = int(np.sum(sr.neg_scores >= t))
            
    def find_optima(indices):
        sum_tp = np.sum(tp_mat[indices, :], axis=0)
        sum_fp = np.sum(fp_mat[indices, :], axis=0)
        sum_pos = np.sum(pos_arr[indices])
        sum_neg = np.sum(neg_arr[indices])
        
        recs = sum_tp / sum_pos if sum_pos > 0 else np.zeros_like(sum_tp)
        specs = (sum_neg - sum_fp) / sum_neg if sum_neg > 0 else np.zeros_like(sum_fp)
        youdens = recs + specs - 1.0
        
        # Rec >= 95%: highest theta where rec >= 0.95
        valid_95 = np.where(recs >= 0.95)[0]
        t_95 = thetas[valid_95[-1]] if len(valid_95) > 0 else thetas[0]
        
        valid_98 = np.where(recs >= 0.98)[0]
        t_98 = thetas[valid_98[-1]] if len(valid_98) > 0 else thetas[0]
        
        best_yj_idx = np.argmax(youdens)
        t_yj = thetas[best_yj_idx]
        
        return float(t_95), float(t_98), float(t_yj)

    point_95, point_98, point_yj = find_optima(np.arange(n_srs))
    
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n_srs, size=(n_bootstraps, n_srs))
    
    boot_95 = []
    boot_98 = []
    boot_yj = []
    
    for b in range(n_bootstraps):
        t95, t98, tyj = find_optima(boot_indices[b])
        boot_95.append(t95)
        boot_98.append(t98)
        boot_yj.append(tyj)
        
    def ci_range(arr):
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        
    return {
        "point_rec95": point_95,
        "ci_rec95": ci_range(boot_95),
        "point_rec98": point_98,
        "ci_rec98": ci_range(boot_98),
        "point_youden": point_yj,
        "ci_youden": ci_range(boot_yj),
    }


def print_section(title):
    print("\n" + "=" * 90)
    print(f" {title}")
    print("=" * 90)


def main():
    args = parse_args()
    df = load_data(args.db)
    all_srs = sorted(df["sr_pmid"].unique())
    calib_srs = [s for s in all_srs if s in CALIBRATION_SRS]
    non_calib_srs = [s for s in all_srs if s not in CALIBRATION_SRS]
    
    print(f"Loaded {len(df):,} total evaluations across {len(all_srs)} systematic reviews from {args.db}.")
    print(f"Calibration SRs present ({len(calib_srs)}): {calib_srs}")
    print(f"Non-calibration SRs ({len(non_calib_srs)}): {len(non_calib_srs)} reviews")
    
    # Pre-parse into SRData objects
    sr_objs_all = []
    for sr_id in all_srs:
        df_sr = df[df["sr_pmid"] == sr_id]
        sr_objs_all.append(SRData(sr_id, df_sr["is_inclusion"].values, df_sr["relevance_score"].values))
        
    sr_objs_primary = [s for s in sr_objs_all if s.sr_pmid not in CALIBRATION_SRS]
    sr_objs_calib = [s for s in sr_objs_all if s.sr_pmid in CALIBRATION_SRS]
    
    # ---------------------------------------------------------
    # 1. Primary Evaluation (Non-Calibration SRs) at theta*
    # ---------------------------------------------------------
    print_section(f"PRIMARY EVALUATION: {len(sr_objs_primary)} Non-Calibration SRs (theta* = {args.theta})")
    m_prim = compute_group_metrics(sr_objs_primary, args.theta)
    ci_prim = cluster_bootstrap(sr_objs_primary, args.theta, n_bootstraps=args.bootstraps, seed=args.seed)
    
    print(f"Total Evaluations : {m_prim['total_docs']:,}")
    print(f"Total Inclusions  : {m_prim['total_pos']} (Prevalence: {m_prim['total_pos']/m_prim['total_docs']*100:.3f}%)")
    print(f"Confusion Matrix  : TP={m_prim['total_tp']}, FP={m_prim['total_fp']}, FN={m_prim['total_fn']}, TN={m_prim['total_tn']}")
    print(f"Pooled Recall     : {m_prim['pooled_recall']*100:.2f}% ({m_prim['total_tp']}/{m_prim['total_pos']}) [95% CI: {ci_prim['pooled_recall_ci'][0]*100:.2f}% - {ci_prim['pooled_recall_ci'][1]*100:.2f}%]")
    print(f"Pooled Specificity: {m_prim['pooled_specificity']*100:.2f}% ({m_prim['total_tn']}/{m_prim['total_neg']}) [95% CI: {ci_prim['pooled_specificity_ci'][0]*100:.2f}% - {ci_prim['pooled_specificity_ci'][1]*100:.2f}%]")
    print(f"Pooled Precision  : {m_prim['pooled_precision']*100:.2f}% [95% CI: {ci_prim['pooled_precision_ci'][0]*100:.2f}% - {ci_prim['pooled_precision_ci'][1]*100:.2f}%]")
    print(f"Pooled F1 / F2    : F1={m_prim['pooled_f1']:.4f}, F2={m_prim['pooled_f2']:.4f}, MCC={m_prim['pooled_mcc']:.4f}")
    print(f"Macro Recall      : Mean={m_prim['macro_recall_mean']*100:.2f}%, Median={m_prim['macro_recall_median']*100:.2f}%, Range=[{m_prim['macro_recall_min']*100:.1f}%, {m_prim['macro_recall_max']*100:.1f}%] [95% CI: {ci_prim['macro_recall_ci'][0]*100:.2f}% - {ci_prim['macro_recall_ci'][1]*100:.2f}%]")
    print(f"Macro Specificity : Mean={m_prim['macro_specificity_mean']*100:.2f}%, Median={m_prim['macro_specificity_median']*100:.2f}% [95% CI: {ci_prim['macro_specificity_ci'][0]*100:.2f}% - {ci_prim['macro_specificity_ci'][1]*100:.2f}%]")
    print(f"Macro ROC-AUC     : Mean={m_prim['macro_roc_auc_mean']:.4f}, Median={m_prim['macro_roc_auc_median']:.4f}")
    print(f"Macro PR-AUC      : Mean={m_prim['macro_pr_auc_mean']:.4f}, Median={m_prim['macro_pr_auc_median']:.4f}, Range=[{m_prim['macro_pr_auc_min']:.4f}, {m_prim['macro_pr_auc_max']:.4f}] [95% CI: {ci_prim['macro_pr_auc_ci'][0]:.4f} - {ci_prim['macro_pr_auc_ci'][1]:.4f}]")
    print(f"Rank WSS@95       : Mean={m_prim['macro_wss_95_mean']*100:.2f}%, Median={m_prim['macro_wss_95_median']*100:.2f}% [95% CI: {ci_prim['macro_wss_95_ci'][0]*100:.2f}% - {ci_prim['macro_wss_95_ci'][1]*100:.2f}%]")
    print(f"Rank WSS@100      : Mean={m_prim['macro_wss_100_mean']*100:.2f}%, Median={m_prim['macro_wss_100_median']*100:.2f}% [95% CI: {ci_prim['macro_wss_100_ci'][0]*100:.2f}% - {ci_prim['macro_wss_100_ci'][1]*100:.2f}%]")
    print(f"Safety Compliance : {m_prim['pct_srs_recall_ge_95']:.1f}% of SRs have Recall >= 95%; {m_prim['pct_srs_recall_100']:.1f}% have Recall = 100%")

    # ---------------------------------------------------------
    # 2. Secondary Evaluation (All Available SRs) at theta*
    # ---------------------------------------------------------
    print_section(f"SECONDARY EVALUATION: All {len(sr_objs_all)} SRs (theta* = {args.theta})")
    m_sec = compute_group_metrics(sr_objs_all, args.theta)
    ci_sec = cluster_bootstrap(sr_objs_all, args.theta, n_bootstraps=args.bootstraps, seed=args.seed)
    
    print(f"Total Evaluations : {m_sec['total_docs']:,}")
    print(f"Total Inclusions  : {m_sec['total_pos']}")
    print(f"Confusion Matrix  : TP={m_sec['total_tp']}, FP={m_sec['total_fp']}, FN={m_sec['total_fn']}, TN={m_sec['total_tn']}")
    print(f"Pooled Recall     : {m_sec['pooled_recall']*100:.2f}% ({m_sec['total_tp']}/{m_sec['total_pos']}) [95% CI: {ci_sec['pooled_recall_ci'][0]*100:.2f}% - {ci_sec['pooled_recall_ci'][1]*100:.2f}%]")
    print(f"Pooled Specificity: {m_sec['pooled_specificity']*100:.2f}% ({m_sec['total_tn']}/{m_sec['total_neg']}) [95% CI: {ci_sec['pooled_specificity_ci'][0]*100:.2f}% - {ci_sec['pooled_specificity_ci'][1]*100:.2f}%]")
    print(f"Pooled Precision  : {m_sec['pooled_precision']*100:.2f}% [95% CI: {ci_sec['pooled_precision_ci'][0]*100:.2f}% - {ci_sec['pooled_precision_ci'][1]*100:.2f}%]")
    print(f"Macro Recall      : Mean={m_sec['macro_recall_mean']*100:.2f}%, Median={m_sec['macro_recall_median']*100:.2f}% [95% CI: {ci_sec['macro_recall_ci'][0]*100:.2f}% - {ci_sec['macro_recall_ci'][1]*100:.2f}%]")
    print(f"Macro Specificity : Mean={m_sec['macro_specificity_mean']*100:.2f}%, Median={m_sec['macro_specificity_median']*100:.2f}% [95% CI: {ci_sec['macro_specificity_ci'][0]*100:.2f}% - {ci_sec['macro_specificity_ci'][1]*100:.2f}%]")
    print(f"Macro PR-AUC      : Mean={m_sec['macro_pr_auc_mean']:.4f}, Median={m_sec['macro_pr_auc_median']:.4f} [95% CI: {ci_sec['macro_pr_auc_ci'][0]:.4f} - {ci_sec['macro_pr_auc_ci'][1]:.4f}]")
    print(f"Rank WSS@95       : Mean={m_sec['macro_wss_95_mean']*100:.2f}%, Median={m_sec['macro_wss_95_median']*100:.2f}% [95% CI: {ci_sec['macro_wss_95_ci'][0]*100:.2f}% - {ci_sec['macro_wss_95_ci'][1]*100:.2f}%]")
    print(f"Rank WSS@100      : Mean={m_sec['macro_wss_100_mean']*100:.2f}%, Median={m_sec['macro_wss_100_median']*100:.2f}% [95% CI: {ci_sec['macro_wss_100_ci'][0]*100:.2f}% - {ci_sec['macro_wss_100_ci'][1]*100:.2f}%]")

    # ---------------------------------------------------------
    # 3. Calibration SRs Performance
    # ---------------------------------------------------------
    if len(sr_objs_calib) > 0:
        print_section(f"CALIBRATION SET: {len(sr_objs_calib)} Calibration SRs (theta* = {args.theta})")
        m_cal = compute_group_metrics(sr_objs_calib, args.theta)
        print(f"Total Inclusions  : {m_cal['total_pos']}")
        print(f"Pooled Recall     : {m_cal['pooled_recall']*100:.2f}% ({m_cal['total_tp']}/{m_cal['total_pos']})")
        print(f"Pooled Specificity: {m_cal['pooled_specificity']*100:.2f}% ({m_cal['total_tn']}/{m_cal['total_neg']})")

    # ---------------------------------------------------------
    # 4. Threshold Sweep Table (Primary Non-Calibration Set)
    # ---------------------------------------------------------
    print_section(f"THRESHOLD SWEEP (0.020 - 0.040) on Primary Set ({len(sr_objs_primary)} SRs)")
    sweep_df = threshold_sweep(sr_objs_primary, 0.020, 0.040, 0.001)
    print(f"{'Theta':<7} | {'Pooled Rec':<11} | {'Pooled Spec':<12} | {'Pooled Prec':<12} | {'Macro Rec':<10} | {'Macro Spec':<11} | {'Youden J':<9} | {'Workload %':<10}")
    print("-" * 90)
    for _, row in sweep_df.iterrows():
        star = " ★" if abs(row["theta"] - args.theta) < 1e-4 else ""
        print(f"{row['theta']:<7.3f} | {row['pooled_recall']*100:>9.2f}% | {row['pooled_specificity']*100:>10.2f}% | {row['pooled_precision']*100:>10.2f}% | {row['macro_recall']*100:>8.2f}% | {row['macro_specificity']*100:>9.2f}% | {row['youden_j']:>9.4f} | {row['workload_pct']:>8.2f}%{star}")

    # ---------------------------------------------------------
    # 5. Post-Hoc Optimums and Stability Analysis
    # ---------------------------------------------------------
    print_section(f"POST-HOC OPTIMA & STABILITY (Primary Set: {len(sr_objs_primary)} SRs)")
    optima = post_hoc_optima_with_bootstrap(sr_objs_primary, n_bootstraps=2000, seed=args.seed)
    print(f"Post-hoc Optimum (Recall >= 95%): theta = {optima['point_rec95']:.4f} [95% Bootstrap Range: {optima['ci_rec95'][0]:.4f} - {optima['ci_rec95'][1]:.4f}]")
    print(f"Post-hoc Optimum (Recall >= 98%): theta = {optima['point_rec98']:.4f} [95% Bootstrap Range: {optima['ci_rec98'][0]:.4f} - {optima['ci_rec98'][1]:.4f}]")
    print(f"Post-hoc Optimum (Youden J max) : theta = {optima['point_youden']:.4f} [95% Bootstrap Range: {optima['ci_youden'][0]:.4f} - {optima['ci_youden'][1]:.4f}]")

    # ---------------------------------------------------------
    # 6. Detailed Per-SR Breakdown (Primary Set Misses / Boundary Cases)
    # ---------------------------------------------------------
    print_section(f"PER-SR RECALL BREAKDOWN (Primary Set: {len(sr_objs_primary)} SRs, theta* = {args.theta})")
    sr_res = m_prim["sr_results"]
    miss_srs = [s for s, r in sr_res.items() if r["fn"] > 0]
    perfect_srs = [s for s, r in sr_res.items() if r["fn"] == 0]
    print(f"Perfect Recall (100%) SRs ({len(perfect_srs)}/{len(sr_objs_primary)}): {perfect_srs}")
    print(f"SRs with Missed Inclusions ({len(miss_srs)}/{len(sr_objs_primary)}):")
    for s in sorted(miss_srs):
        r = sr_res[s]
        print(f"  SR {s}: Recall = {r['recall']*100:.1f}% ({r['tp']}/{r['n_pos']}), Misses = {r['fn']}, Spec = {r['specificity']*100:.2f}%, PR-AUC = {r['pr_auc']:.4f}, WSS@95 = {r['wss_95']*100:.1f}%")


if __name__ == "__main__":
    main()
