#!/usr/bin/env python3
"""
benchmark/design2_and_supplementary_analysis.py

Peer-review revision analyses on existing scores (no model inference):

  A. Design 2 (Boolean-Search) metrics at the frozen theta* = 0.028, recomputed from
     the exact (review, document) pairs retrieved by each review's Boolean search.
  B. Per-review recall table for the 62 primary reviews (supplementary table).
  C. Leave-one-out sensitivity of the calibration rule over the 4 calibration reviews
     (post hoc, descriptive; theta* stays frozen at 0.028).

Reuses SRData / compute_group_metrics / cluster_bootstrap from threshold_analysis.py.
Reads the canonical parquet scores (SQLite is git-ignored).

Usage:
  .venv/bin/python benchmark/design2_and_supplementary_analysis.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import threshold_analysis as ta  # noqa: E402

THETA = 0.028


def find_file(candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PARENT_DIR = os.path.dirname(PROJECT_DIR)

SCORES = find_file([
    os.path.join(PROJECT_DIR, "outputs/soces_pico_laya_e2e_results.parquet"),
    os.path.join(PARENT_DIR, "outputs/soces_pico_laya_e2e_results.parquet"),
    "outputs/soces_pico_laya_e2e_results.parquet",
])

CANDIDATES = find_file([
    os.path.join(PROJECT_DIR, "benchmark/benchmark_candidates.parquet"),
    os.path.join(PARENT_DIR, "benchmark/benchmark_candidates.parquet"),
    "benchmark/benchmark_candidates.parquet",
])

OBJECTIVES = find_file([
    os.path.join(PROJECT_DIR, "benchmark/sr_review_objectives.json"),
    os.path.join(PARENT_DIR, "benchmark/sr_review_objectives.json"),
    "benchmark/sr_review_objectives.json",
])

OUT_DIR = os.path.join(PROJECT_DIR, "publication/tables")


def norm(df):
    for k in ("sr_pmid", "pmid"):
        df[k] = df[k].astype(str).str.split(".").str[0]
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scores = norm(pd.read_parquet(SCORES))
    cand = norm(pd.read_parquet(CANDIDATES))
    cal = set(ta.CALIBRATION_SRS)
    excluded = set(cand[cand.excluded_from_study.astype(bool)].sr_pmid)

    # ---------------- A. Design 2 ----------------
    pairs = cand[~cand.sr_pmid.isin(cal | excluded)][["sr_pmid", "pmid", "is_inclusion"]]
    d2 = pairs.merge(scores[["sr_pmid", "pmid", "relevance_score", "is_inclusion"]],
                     on=["sr_pmid", "pmid"], how="left", suffixes=("_cand", ""))
    assert d2.relevance_score.notna().all(), "Boolean-search pair missing from score table"
    assert (d2.is_inclusion.astype(int) == d2.is_inclusion_cand.astype(int)).all(), "label mismatch"
    objs = [ta.SRData(s, g.is_inclusion.values, g.relevance_score.values)
            for s, g in d2.groupby("sr_pmid")]
    m = ta.compute_group_metrics(objs, THETA)
    ci = ta.cluster_bootstrap(objs, THETA, n_bootstraps=5000, seed=42)
    n = m["total_docs"]
    per = list(m["sr_results"].values())
    workload = np.array([(r["tp"] + r["fp"]) / r["n_total"] for r in per])
    prev = np.array([r["n_pos"] / r["n_total"] for r in per])
    sizes = np.array([r["n_total"] for r in per])

    design2 = {
        "n_reviews": m["n_srs"],
        "n_pairs": int(n),
        "n_unique_documents": int(d2.pmid.nunique()),
        "n_inclusions": int(m["total_pos"]),
        "pooled_prevalence": m["total_pos"] / n,
        "mean_per_review_prevalence": float(prev.mean()),
        "median_per_review_prevalence": float(np.median(prev)),
        "mean_pairs_per_review": float(sizes.mean()),
        "median_pairs_per_review": float(np.median(sizes)),
        "tp": m["total_tp"], "fp": m["total_fp"], "fn": m["total_fn"], "tn": m["total_tn"],
        "pooled_recall": m["pooled_recall"], "pooled_recall_ci": ci["pooled_recall_ci"],
        "pooled_specificity": m["pooled_specificity"], "pooled_specificity_ci": ci["pooled_specificity_ci"],
        "pooled_workload": (m["total_tp"] + m["total_fp"]) / n,
        "macro_mean_workload": float(workload.mean()),
        "median_workload": float(np.median(workload)),
        "macro_specificity_mean": m["macro_specificity_mean"],
        "macro_wss95_mean": m["macro_wss_95_mean"], "macro_wss95_ci": ci["macro_wss_95_ci"],
        "macro_wss95_median": m["macro_wss_95_median"],
        "macro_pr_auc_mean": m["macro_pr_auc_mean"], "macro_pr_auc_ci": ci["macro_pr_auc_ci"],
        "macro_pr_auc_min": m["macro_pr_auc_min"], "macro_pr_auc_max": m["macro_pr_auc_max"],
        "macro_roc_auc_mean": m["macro_roc_auc_mean"],
        "reviews_recall_ge_95": int(sum(r["recall"] >= 0.95 for r in per)),
    }

    # Review-level percentile bootstrap (seed 42, 5,000 draws) for per-review quantities
    def boot_mean_ci(vals, n_boot=5000, seed=42):
        vals = np.asarray(vals, dtype=float)
        rng = np.random.default_rng(seed)
        b = [rng.choice(vals, len(vals)).mean() for _ in range(n_boot)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    thr_wss = np.array([(r["tn"] + r["fn"]) / r["n_total"] - (1 - r["recall"]) for r in per])
    roc = np.array([r["roc_auc"] for r in per])
    design2["macro_threshold_wss_mean"] = float(thr_wss.mean())
    design2["macro_threshold_wss_ci"] = boot_mean_ci(thr_wss)
    design2["macro_roc_auc_ci"] = boot_mean_ci(roc)

    # ---------------- B. Per-review table (Design 1, 62 primary reviews) ----------------
    prim = [ta.SRData(s, g.is_inclusion.values, g.relevance_score.values)
            for s, g in scores[~scores.sr_pmid.isin(cal)].groupby("sr_pmid")]
    m1 = ta.compute_group_metrics(prim, THETA)
    thr_wss1 = np.array([(r["tn"] + r["fn"]) / r["n_total"] - (1 - r["recall"])
                         for r in m1["sr_results"].values()])
    design2["design1_threshold_wss_mean"] = float(thr_wss1.mean())
    design2["design1_threshold_wss_ci"] = boot_mean_ci(thr_wss1)
    titles = {str(o["sr_pmid"]).split(".")[0]: o["title"] for o in json.load(open(OBJECTIVES))}
    rows = []
    for s, r in m1["sr_results"].items():
        rows.append({"sr_pmid": s, "title": titles.get(s, ""), "n_inclusions": r["n_pos"],
                     "tp": r["tp"], "misses": r["fn"], "recall": round(r["recall"], 4),
                     "meets_95": r["recall"] >= 0.95, "roc_auc": round(r["roc_auc"], 4),
                     "pr_auc": round(r["pr_auc"], 4), "wss95": round(r["wss_95"], 4)})
    per_review = pd.DataFrame(rows).sort_values(["recall", "misses"], ascending=[True, False])
    per_review.to_csv(f"{OUT_DIR}/per_review_recall.csv", index=False)

    below = per_review[~per_review.meets_95]
    n_below_pos = int(below.n_inclusions.lt(20).sum())
    summary_b = {
        "n_reviews": len(per_review),
        "n_meeting_95": int(per_review.meets_95.sum()),
        "n_below_95": int(len(below)),
        "n_perfect": int((per_review.misses == 0).sum()),
        "below_95_n_inclusions_median": float(below.n_inclusions.median()),
        "meeting_95_n_inclusions_median": float(per_review[per_review.meets_95].n_inclusions.median()),
        "below_95_with_lt20_inclusions": n_below_pos,
        "t2_sensitivity_excl_two": None,
    }
    excl2 = per_review[~per_review.sr_pmid.isin(["24046285", "25770113"])]
    summary_b["t2_sensitivity_excl_two"] = [int(excl2.meets_95.sum()), len(excl2)]
    # minimum extra reviews needed / fragility: recall of reviews near the boundary
    near = per_review[(per_review.recall >= 0.90) & (per_review.recall < 0.95)]
    summary_b["reviews_recall_90_to_95"] = [(r.sr_pmid, r.recall, r.n_inclusions) for r in near.itertuples()]
    # fragility: reviews meeting 95% with exactly the minimum-passing margin
    summary_b["meeting_95_with_misses"] = int(((per_review.meets_95) & (per_review.misses > 0)).sum())

    # ---------------- C. Leave-one-out on calibration reviews ----------------
    cal_objs = {s: ta.SRData(s, g.is_inclusion.values, g.relevance_score.values)
                for s, g in scores[scores.sr_pmid.isin(cal)].groupby("sr_pmid")}
    grid = np.round(np.arange(0.020, 0.0401, 0.001), 4)

    def select(objs_):
        # Calibration rule: highest theta with pooled recall >= 95.0%
        best = None
        for t in grid:
            mm = ta.compute_group_metrics(objs_, float(t))
            if mm["pooled_recall"] >= 0.95:
                best = float(t)
        return best

    loo = {"all_four": select(list(cal_objs.values()))}
    for s in cal_objs:
        loo[f"drop_{s}"] = select([o for k, o in cal_objs.items() if k != s])
    cal_detail = {s: {"n_pos": o.n_pos, "tp_at_theta": ta.SRData.eval_threshold(o, THETA)["tp"]}
                  for s, g in scores[scores.sr_pmid.isin(cal)].groupby("sr_pmid")
                  for o in [cal_objs[s]]}

    out = {"design2": design2, "design1_per_review_summary": summary_b,
           "calibration_loo_selected_theta": loo, "calibration_detail": cal_detail}
    with open(f"{OUT_DIR}/phase1_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
