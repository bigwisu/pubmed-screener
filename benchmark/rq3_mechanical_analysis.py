#!/usr/bin/env python3
"""
rq3_mechanical_analysis.py — RQ3 Level 1: mechanical error analysis

Produces all Level 1 deliverables from the primary SQLite results without
any human rater:

  L1-A  SR-level miss table (CSV)
  L1-B  Cluster-bootstrap miss rate CI (5,000 resamples, SR clusters)
  L1-C  Noise-floor score distribution (FNs vs TPs vs TNs)
  L1-D  Mechanical category counts
  L1-E  T2 statement with pre-declared wording

Mechanical categories (assigned per missed record, rater-free, no hand-picked
review lists):
  1. short_or_missing_abstract — abstract < 100 chars (includes the literal
     "[No Abstract]" placeholder)
  2. abstract_present          — abstract >= 100 chars

Usage:
  python benchmark/rq3_mechanical_analysis.py
  python benchmark/rq3_mechanical_analysis.py --db outputs/soces_pico_laya_e2e_results.sqlite
  python benchmark/rq3_mechanical_analysis.py --out publication/tables
"""

import argparse
import json
import os
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pre-declared constants (do not change after data is collected)
# ---------------------------------------------------------------------------
THETA_STAR = 0.028
N_PRIMARY_SRS = 62
T2_TARGET_RATE = 0.75          # ≥75% of SRs must have recall ≥95%
T2_RECALL_THRESHOLD = 0.95

# 4 calibration SRs — excluded from the primary 62-SR evaluation set
CALIBRATION_SR_PMIDS = {"22226047", "22323502", "22422870", "22777524"}

SHORT_ABSTRACT_CHARS = 100     # abstract < this → short or missing abstract

N_BOOTSTRAP = 5_000
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_results(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM screening_results", conn)
    conn.close()
    df["sr_pmid"] = df["sr_pmid"].astype(str).str.split(".").str[0]
    df["pmid"] = df["pmid"].astype(str).str.split(".").str[0]
    return df


def load_documents(benchmark_dir: str) -> pd.DataFrame:
    """Load pubmed_documents for abstract-length check."""
    docs_path = os.path.join(benchmark_dir, "pubmed_documents.parquet")
    if not os.path.exists(docs_path):
        print(f"  [warn] pubmed_documents.parquet not found at {docs_path}; abstract-length category skipped.")
        return pd.DataFrame(columns=["pmid", "abstract"])
    df = pd.read_parquet(docs_path, columns=["pmid", "abstract"])
    df["pmid"] = df["pmid"].astype(str).str.split(".").str[0]
    df["abstract_len"] = df["abstract"].fillna("").str.len()
    return df


# ---------------------------------------------------------------------------
# L1-A: SR-level miss table
# ---------------------------------------------------------------------------

def build_sr_miss_table(df: pd.DataFrame, df_docs: pd.DataFrame) -> pd.DataFrame:
    """One row per SR with ≥1 false negative."""
    # Compute per-SR metrics
    rows = []
    for sr_pmid, grp in df.groupby("sr_pmid"):
        tp = int(((grp["relevance_score"] >= THETA_STAR) & (grp["is_inclusion"] == 1)).sum())
        fn = int(((grp["relevance_score"] < THETA_STAR) & (grp["is_inclusion"] == 1)).sum())
        n_gt = tp + fn
        recall = tp / n_gt if n_gt > 0 else 0.0

        # PR-AUC (average precision)
        from sklearn.metrics import average_precision_score  # type: ignore
        if grp["is_inclusion"].nunique() == 2:
            pr_auc = float(average_precision_score(grp["is_inclusion"], grp["relevance_score"]))
        else:
            pr_auc = float("nan")

        rows.append({
            "sr_pmid": sr_pmid,
            "inclusion_count": n_gt,
            "tp": tp,
            "miss_count": fn,
            "recall_pct": round(recall * 100, 1),
            "pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else None,
        })

    df_sr = pd.DataFrame(rows)
    df_misses = df_sr[df_sr["miss_count"] > 0].copy()

    # Add abstract-length counts per SR for missed documents
    if not df_docs.empty:
        fn_pmids = df[
            (df["relevance_score"] < THETA_STAR) & (df["is_inclusion"] == 1)
        ][["sr_pmid", "pmid"]].copy()
        fn_with_len = fn_pmids.merge(df_docs[["pmid", "abstract_len"]], on="pmid", how="left")
        fn_with_len["is_short"] = fn_with_len["abstract_len"].fillna(0) < SHORT_ABSTRACT_CHARS
        short_counts = fn_with_len.groupby("sr_pmid")["is_short"].sum().reset_index()
        short_counts.columns = ["sr_pmid", "short_abstract_count"]
        df_misses = df_misses.merge(short_counts, on="sr_pmid", how="left")
        df_misses["short_abstract_count"] = df_misses["short_abstract_count"].fillna(0).astype(int)
    else:
        df_misses["short_abstract_count"] = None

    return df_misses.sort_values("miss_count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# L1-B: Cluster-bootstrap miss rate CI
# ---------------------------------------------------------------------------

def cluster_bootstrap_miss_rate(df: pd.DataFrame, n_boot: int = N_BOOTSTRAP) -> dict:
    """Bootstrap miss rate with SR as cluster (resamples whole SRs with replacement)."""
    rng = np.random.default_rng(RNG_SEED)
    sr_ids = df["sr_pmid"].unique()
    n_srs = len(sr_ids)

    # Per-SR miss count and total inclusions
    sr_stats = (
        df[df["is_inclusion"] == 1]
        .groupby("sr_pmid")
        .apply(lambda g: pd.Series({
            "fn": int((g["relevance_score"] < THETA_STAR).sum()),
            "n_gt": len(g),
        }))
        .reset_index()
    )
    sr_stats_map = sr_stats.set_index("sr_pmid").to_dict("index")

    # Observed pooled miss rate
    total_fn = sr_stats["fn"].sum()
    total_gt = sr_stats["n_gt"].sum()
    obs_rate = total_fn / total_gt if total_gt > 0 else 0.0

    # Bootstrap
    boot_rates = []
    for _ in range(n_boot):
        sampled = rng.choice(sr_ids, size=n_srs, replace=True)
        b_fn = sum(sr_stats_map.get(s, {}).get("fn", 0) for s in sampled)
        b_gt = sum(sr_stats_map.get(s, {}).get("n_gt", 0) for s in sampled)
        boot_rates.append(b_fn / b_gt if b_gt > 0 else 0.0)

    boot_rates = np.array(boot_rates)
    ci_lo, ci_hi = np.percentile(boot_rates, [2.5, 97.5])

    return {
        "total_fn": int(total_fn),
        "total_gt": int(total_gt),
        "miss_rate": round(obs_rate, 4),
        "miss_rate_pct": round(obs_rate * 100, 2),
        "ci_95_lo": round(ci_lo * 100, 2),
        "ci_95_hi": round(ci_hi * 100, 2),
        "n_srs_clustered": n_srs,
        "n_bootstrap": n_boot,
    }


# ---------------------------------------------------------------------------
# L1-C: Noise-floor score distribution
# ---------------------------------------------------------------------------

def noise_floor_stats(df: pd.DataFrame) -> dict:
    fn_scores = df[(df["relevance_score"] < THETA_STAR) & (df["is_inclusion"] == 1)]["relevance_score"]
    tp_scores = df[(df["relevance_score"] >= THETA_STAR) & (df["is_inclusion"] == 1)]["relevance_score"]
    tn_scores = df[(df["relevance_score"] < THETA_STAR) & (df["is_inclusion"] == 0)]["relevance_score"]

    p5, p95 = np.percentile(df["relevance_score"], [5, 95])

    return {
        "fn_score_min": round(float(fn_scores.min()), 4),
        "fn_score_max": round(float(fn_scores.max()), 4),
        "fn_score_median": round(float(fn_scores.median()), 4),
        "tn_score_median": round(float(tn_scores.median()), 4),
        "tp_score_median": round(float(tp_scores.median()), 4),
        "all_docs_p5": round(float(p5), 4),
        "all_docs_p95": round(float(p95), 4),
        "n_fn": int(len(fn_scores)),
        "n_tp": int(len(tp_scores)),
        "n_tn": int(len(tn_scores)),
        "theta_star": THETA_STAR,
    }


# ---------------------------------------------------------------------------
# L1-D: Category summary
# ---------------------------------------------------------------------------

def category_summary(df: pd.DataFrame, df_docs: pd.DataFrame) -> pd.DataFrame:
    """Record-level split of the misses by abstract length.

    A review can have misses in both categories, so the review counts add up
    to more than the number of reviews with a miss.
    """
    inc = df[df["is_inclusion"] == 1].merge(
        df_docs[["pmid", "abstract_len"]], on="pmid", how="left"
    )
    inc["abstract_len"] = inc["abstract_len"].fillna(0)
    inc["category"] = np.where(
        inc["abstract_len"] < SHORT_ABSTRACT_CHARS,
        "short_or_missing_abstract",
        "abstract_present",
    )
    inc["is_miss"] = inc["relevance_score"] < THETA_STAR

    rows = []
    total_fn = int(inc["is_miss"].sum())
    for cat, grp in inc.groupby("category"):
        n_miss = int(grp["is_miss"].sum())
        rows.append({
            "category": cat,
            "miss_count": n_miss,
            "sr_count_with_miss": int(grp.loc[grp["is_miss"], "sr_pmid"].nunique()),
            "pct_of_all_fn": round(n_miss / total_fn * 100, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# L1-E: T2 statement
# ---------------------------------------------------------------------------

def t2_statement(df: pd.DataFrame) -> dict:
    sr_recall = (
        df[df["is_inclusion"] == 1]
        .groupby("sr_pmid")
        .apply(lambda g: pd.Series({
            "recall": int((g["relevance_score"] >= THETA_STAR).sum()) / len(g)
        }))
        .reset_index()
    )
    n_srs = len(sr_recall)
    n_at_95 = int((sr_recall["recall"] >= T2_RECALL_THRESHOLD).sum())
    t2_rate = n_at_95 / n_srs if n_srs > 0 else 0.0

    return {
        "n_srs": n_srs,
        "n_at_95_recall": n_at_95,
        "t2_rate": round(t2_rate * 100, 1),
        "t2_target_pct": T2_TARGET_RATE * 100,
        "t2_met": t2_rate >= T2_TARGET_RATE,
        "statement": (
            f"T2 (per-SR recall ≥ 95% in ≥ 75% of {n_srs} primary SRs) was "
            f"{'met' if t2_rate >= T2_TARGET_RATE else 'not met'}: "
            f"{n_at_95}/{n_srs} = {t2_rate * 100:.1f}%."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RQ3 Level 1: mechanical error analysis (no human rater)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db", type=str,
        default="outputs/soces_pico_laya_e2e_results.sqlite",
        help="Path to primary results SQLite database",
    )
    parser.add_argument(
        "--out", type=str,
        default="publication/tables",
        help="Output directory for CSV tables and JSON summary",
    )
    parser.add_argument(
        "--benchmark-dir", type=str,
        default="benchmark",
        help="Directory containing pubmed_documents.parquet",
    )
    parser.add_argument(
        "--all-srs", action="store_true",
        help="Include calibration SRs (default: primary 62 non-calibration SRs only)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading results from: {args.db}")
    df = load_results(args.db)
    df_docs = load_documents(args.benchmark_dir)

    if not args.all_srs:
        df = df[~df["sr_pmid"].isin(CALIBRATION_SR_PMIDS)].copy()
        print(f"  Restricted to primary non-calibration SRs (excluding {len(CALIBRATION_SR_PMIDS)} calibration SRs)")

    print(f"  {len(df):,} rows, {df['sr_pmid'].nunique()} unique SRs")

    print("\n── L1-A: SR-level miss table ────────────────────────────")
    df_miss = build_sr_miss_table(df, df_docs)
    miss_path = os.path.join(args.out, "rq3_miss_table.csv")
    df_miss.to_csv(miss_path, index=False)
    print(df_miss.to_string(index=False))
    print(f"\n  → saved: {miss_path}")

    print("\n── L1-B: Cluster-bootstrap miss rate CI ─────────────────")
    ci = cluster_bootstrap_miss_rate(df)
    print(f"  Miss rate: {ci['miss_rate_pct']:.2f}%  "
          f"(95% CI: {ci['ci_95_lo']:.2f}%–{ci['ci_95_hi']:.2f}%)  "
          f"[{ci['total_fn']} FNs / {ci['total_gt']} inclusions, "
          f"{ci['n_srs_clustered']} SR clusters, {ci['n_bootstrap']:,} resamples]")

    print("\n── L1-C: Noise-floor score distribution ─────────────────")
    nf = noise_floor_stats(df)
    print(f"  FN scores:  min={nf['fn_score_min']}  median={nf['fn_score_median']}  max={nf['fn_score_max']}")
    print(f"  TN median:  {nf['tn_score_median']}    TP median: {nf['tp_score_median']}")
    print(f"  All-docs 5th–95th pct: [{nf['all_docs_p5']}, {nf['all_docs_p95']}]")
    print(f"  θ* = {nf['theta_star']}")

    print("\n── L1-D: Mechanical category counts ─────────────────────")
    cats = category_summary(df, df_docs)
    print(cats.to_string(index=False))

    print("\n── L1-E: T2 statement ───────────────────────────────────")
    t2 = t2_statement(df)
    print(f"\n  {t2['statement']}")

    # Save combined JSON summary
    summary = {
        "theta_star": THETA_STAR,
        "ci": ci,
        "noise_floor": nf,
        "categories": cats.to_dict(orient="records"),
        "t2": t2,
    }
    summary_path = os.path.join(args.out, "rq3_level1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → saved: {summary_path}")
    print("\n✅  Level 1 analysis complete.")


if __name__ == "__main__":
    main()
