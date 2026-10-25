#!/usr/bin/env python3
"""
expanded_specificity_analysis.py — Full-corpus specificity-widening counterfactual.

Validates whether widening instruction specificity (ori → v2) recovers false
negatives without inflating false positives across the full 15,530-document
PubMed corpus for each of the 5 target systematic reviews.

For each SR, each document is scored with BOTH the original (ori) and the
widened (v2) instruction. Delta scores (v2 − ori) are recorded along with
the resulting confusion matrix shift.

Input:
  benchmark/rq3_instruction_counterfactual/expanded_specifity.json
    Fields: pmid (SR PMID), ori (original instruction), v2 (widened instruction)

Outputs:
  outputs/expanded_specificity/
    per_doc_deltas.csv          — one row per (sr_pmid, pmid): ori, v2, delta, label change
    summary.csv                 — one row per SR: confusion matrix counts + delta summary
    summary.json                — same, machine-readable

Usage (from pubmed-screener/ directory):
  python benchmark/rq3_instruction_counterfactual/expanded_specificity_analysis.py \\
      --model models/soces_pico_laya_e2e \\
      --batch-size 64

  # Single SR for smoke-testing:
  python benchmark/rq3_instruction_counterfactual/expanded_specificity_analysis.py \\
      --model models/soces_pico_laya_e2e \\
      --sr 31585960 --batch-size 32

  # Limit candidates per SR (quick iteration):
  python benchmark/rq3_instruction_counterfactual/expanded_specificity_analysis.py \\
      --model models/soces_pico_laya_e2e \\
      --limit 500 --batch-size 64
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — resolve laya submodule from any working directory
# ---------------------------------------------------------------------------

# Script lives at: <root>/benchmark/rq3_instruction_counterfactual/
# Root repo:       <root>/   (contains laya/, models/, benchmark/, outputs/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_LAYA_DIR   = os.path.join(_ROOT_DIR, "laya")

if _LAYA_DIR not in sys.path:
    sys.path.insert(0, _LAYA_DIR)

import laya  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THRESHOLD = 0.028
SPEC_JSON = os.path.join(_SCRIPT_DIR, "expanded_specifity.json")

DOCS_PARQUET     = os.path.join(_ROOT_DIR, "benchmark", "pubmed_documents.parquet")
MAPPINGS_PARQUET = os.path.join(_ROOT_DIR, "benchmark", "sr_document_mappings.parquet")

DEFAULT_OUTPUT_DIR = os.path.join(_ROOT_DIR, "outputs", "expanded_specificity")
DEFAULT_MODEL      = os.path.join(_ROOT_DIR, "models", "soces_pico_laya_e2e")


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def score_batch(agent, instruction: str, states: list[str]) -> list[float]:
    """Score a list of (title+abstract) states against a single instruction."""
    questions = {"relevance": {"type": "noul", "instructions": instruction.strip()}}
    results = agent.predict_batch(states, questions)
    scores = []
    for res in results:
        ans = res["answers"]["relevance"]
        scores.append(float(ans.get("noul", ans.get("probability", 0.0))))
    return scores


def build_states(df: pd.DataFrame) -> list[str]:
    states = []
    for _, row in df.iterrows():
        title    = str(row["title"])    if pd.notna(row.get("title"))    else ""
        abstract = str(row["abstract"]) if pd.notna(row.get("abstract")) else ""
        states.append(f"Title: {title.strip()}\nAbstract: {abstract.strip()}")
    return states


# ---------------------------------------------------------------------------
# Per-SR analysis
# ---------------------------------------------------------------------------

def analyse_sr(
    agent,
    sr_pmid: str,
    ori: str,
    v2: str,
    df_corpus: pd.DataFrame,
    threshold: float,
    batch_size: int,
    v3: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Score all corpus documents for one SR with ori, v2, and optionally v3 instructions.

    Returns:
        df_results  — per-document DataFrame with scores, deltas, and change labels
        summary     — dict with confusion matrix shift and delta statistics
    """
    states = build_states(df_corpus)
    n = len(states)
    n_passes = 3 if v3 else 2

    ori_scores: list[float] = []
    v2_scores:  list[float] = []
    v3_scores:  list[float] = []

    desc = f"  {sr_pmid}"
    with tqdm(total=n * n_passes, desc=desc, unit="doc", leave=False) as pbar:
        # Ori pass
        for i in range(0, n, batch_size):
            batch = states[i:i + batch_size]
            ori_scores.extend(score_batch(agent, ori, batch))
            pbar.update(len(batch))
        # V2 pass
        for i in range(0, n, batch_size):
            batch = states[i:i + batch_size]
            v2_scores.extend(score_batch(agent, v2, batch))
            pbar.update(len(batch))
        # V3 pass (optional)
        if v3:
            for i in range(0, n, batch_size):
                batch = states[i:i + batch_size]
                v3_scores.extend(score_batch(agent, v3, batch))
                pbar.update(len(batch))

    df = df_corpus[["pmid", "is_inclusion"]].copy()
    df["sr_pmid"]     = sr_pmid
    df["ori_score"]   = ori_scores
    df["v2_score"]    = v2_scores
    df["delta_v2"]    = df["v2_score"] - df["ori_score"]
    df["delta"]       = df["delta_v2"]   # backward-compat alias
    df["ori_include"] = df["ori_score"] >= threshold
    df["v2_include"]  = df["v2_score"]  >= threshold
    df["is_inclusion"] = df["is_inclusion"].astype(bool)

    if v3:
        df["v3_score"]   = v3_scores
        df["delta_v3"]   = df["v3_score"] - df["ori_score"]
        df["v3_include"] = df["v3_score"] >= threshold

    # V2 change categories (ori → v2)
    conditions = [
        ( df["ori_include"] &  df["v2_include"] &  df["is_inclusion"]),
        ( df["ori_include"] &  df["v2_include"] & ~df["is_inclusion"]),
        (~df["ori_include"] & ~df["v2_include"] &  df["is_inclusion"]),
        (~df["ori_include"] & ~df["v2_include"] & ~df["is_inclusion"]),
        (~df["ori_include"] &  df["v2_include"] &  df["is_inclusion"]),
        (~df["ori_include"] &  df["v2_include"] & ~df["is_inclusion"]),
        ( df["ori_include"] & ~df["v2_include"] &  df["is_inclusion"]),
        ( df["ori_include"] & ~df["v2_include"] & ~df["is_inclusion"]),
    ]
    labels = [
        "TP→TP", "FP→FP", "FN→FN", "TN→TN",
        "FN→TP (recovered)", "TN→FP (new_fp)",
        "TP→FN (lost)", "FP→TN (narrowed_out)",
    ]
    df["change"] = "other"
    for cond, label in zip(conditions, labels):
        df.loc[cond, "change"] = label

    # V3 change categories (ori → v3)
    if v3:
        v3_conditions = [
            ( df["ori_include"] &  df["v3_include"] &  df["is_inclusion"]),
            ( df["ori_include"] &  df["v3_include"] & ~df["is_inclusion"]),
            (~df["ori_include"] & ~df["v3_include"] &  df["is_inclusion"]),
            (~df["ori_include"] & ~df["v3_include"] & ~df["is_inclusion"]),
            (~df["ori_include"] &  df["v3_include"] &  df["is_inclusion"]),
            (~df["ori_include"] &  df["v3_include"] & ~df["is_inclusion"]),
            ( df["ori_include"] & ~df["v3_include"] &  df["is_inclusion"]),
            ( df["ori_include"] & ~df["v3_include"] & ~df["is_inclusion"]),
        ]
        df["change_v3"] = "other"
        for cond, label in zip(v3_conditions, labels):
            df.loc[cond, "change_v3"] = label

    # Confusion matrix counts
    def cm(include_col: str) -> dict:
        inc = df[include_col].astype(bool)
        pos = df["is_inclusion"]
        return {
            "tp": int(( inc &  pos).sum()),
            "fp": int(( inc & ~pos).sum()),
            "fn": int((~inc &  pos).sum()),
            "tn": int((~inc & ~pos).sum()),
        }

    ori_cm = cm("ori_include")
    v2_cm  = cm("v2_include")

    def recall(m):    return m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0.0
    def precision(m): return m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0.0

    fn_mask = ~df["ori_include"] & df["is_inclusion"]

    summary = {
        "sr_pmid": sr_pmid,
        "n_corpus": n,
        "n_inclusions": int(df["is_inclusion"].sum()),
        "threshold": threshold,

        # Ori confusion matrix
        "ori_tp": ori_cm["tp"], "ori_fp": ori_cm["fp"],
        "ori_fn": ori_cm["fn"], "ori_tn": ori_cm["tn"],
        "ori_recall":    round(recall(ori_cm),    4),
        "ori_precision": round(precision(ori_cm), 4),

        # V2 confusion matrix
        "v2_tp": v2_cm["tp"],  "v2_fp": v2_cm["fp"],
        "v2_fn": v2_cm["fn"],  "v2_tn": v2_cm["tn"],
        "v2_recall":    round(recall(v2_cm),    4),
        "v2_precision": round(precision(v2_cm), 4),

        # V2 delta counts
        "fn_recovered":    int((df["change"] == "FN→TP (recovered)").sum()),
        "new_fp":          int((df["change"] == "TN→FP (new_fp)").sum()),
        "tp_lost":         int((df["change"] == "TP→FN (lost)").sum()),
        "fp_narrowed_out": int((df["change"] == "FP→TN (narrowed_out)").sum()),

        # V2 score delta statistics
        "delta_mean":      round(float(df["delta_v2"].mean()), 6),
        "delta_median":    round(float(df["delta_v2"].median()), 6),
        "delta_std":       round(float(df["delta_v2"].std()), 6),
        "delta_p95":       round(float(np.percentile(df["delta_v2"], 95)), 6),
        "delta_p5":        round(float(np.percentile(df["delta_v2"], 5)), 6),
        "fn_delta_mean":   round(float(df.loc[fn_mask, "delta_v2"].mean()), 6) if fn_mask.sum() > 0 else None,
        "fn_delta_median": round(float(df.loc[fn_mask, "delta_v2"].median()), 6) if fn_mask.sum() > 0 else None,

        # Instructions (for audit trail)
        "ori": ori,
        "v2":  v2,
        "v3":  v3,
    }

    # V3 metrics (appended only when v3 instruction provided)
    if v3:
        v3_cm = cm("v3_include")
        summary.update({
            "v3_tp": v3_cm["tp"],  "v3_fp": v3_cm["fp"],
            "v3_fn": v3_cm["fn"],  "v3_tn": v3_cm["tn"],
            "v3_recall":    round(recall(v3_cm),    4),
            "v3_precision": round(precision(v3_cm), 4),
            "v3_fn_recovered":    int((df["change_v3"] == "FN→TP (recovered)").sum()),
            "v3_new_fp":          int((df["change_v3"] == "TN→FP (new_fp)").sum()),
            "v3_tp_lost":         int((df["change_v3"] == "TP→FN (lost)").sum()),
            "v3_fp_narrowed_out": int((df["change_v3"] == "FP→TN (narrowed_out)").sum()),
            "v3_delta_mean":      round(float(df["delta_v3"].mean()), 6),
            "v3_delta_median":    round(float(df["delta_v3"].median()), 6),
            "v3_fn_delta_mean":   round(float(df.loc[fn_mask, "delta_v3"].mean()), 6) if fn_mask.sum() > 0 else None,
            "v3_fn_delta_median": round(float(df.loc[fn_mask, "delta_v3"].median()), 6) if fn_mask.sum() > 0 else None,
        })

    return df, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full-corpus expanded-specificity counterfactual (5 SRs × 15,530 documents)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      type=str, default=DEFAULT_MODEL,
                        help="Path or HF repo ID of the model checkpoint")
    parser.add_argument("--device",     type=str, default=None,
                        help="Compute device ('cuda', 'mps', 'cpu', or auto if None)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size")
    parser.add_argument("--threshold",  type=float, default=THRESHOLD,
                        help="Decision threshold theta*")
    parser.add_argument("--sr",         type=str, default=None,
                        help="Restrict to one SR PMID (e.g. 31585960)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Limit candidates per SR for quick iteration")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for CSV/JSON results")
    parser.add_argument("--spec-file",  type=str, default=None,
                        help="Override path to expanded_specifity.json")
    parser.add_argument("--docs",       type=str, default=DOCS_PARQUET,
                        help="Path to pubmed_documents.parquet")
    parser.add_argument("--mappings",   type=str, default=MAPPINGS_PARQUET,
                        help="Path to sr_document_mappings.parquet")
    parser.add_argument("--force",      action="store_true",
                        help="Re-run even if output files already exist")
    args = parser.parse_args()

    # ── Load spec file ──────────────────────────────────────────────────────
    spec_path = args.spec_file or SPEC_JSON
    if not os.path.exists(spec_path):
        sys.exit(f"ERROR: Cannot find expanded_specifity.json at: {spec_path}")

    with open(spec_path) as f:
        spec_entries = json.load(f)

    # Normalise: support both {pmid,ori,v2} and {sr_pmid,ori,v2}
    entries = []
    for e in spec_entries:
        sr = str(e.get("pmid") or e.get("sr_pmid", "")).split(".")[0]
        if not sr:
            continue
        entries.append({"sr_pmid": sr, "ori": e["ori"], "v2": e["v2"]})

    if args.sr:
        target = args.sr.strip().split(".")[0]
        entries = [e for e in entries if e["sr_pmid"] == target]
        if not entries:
            sys.exit(f"ERROR: SR PMID {args.sr} not found in spec file.")

    print(f"Spec entries loaded: {len(entries)} SR(s)")

    # ── Load corpus ─────────────────────────────────────────────────────────
    if not os.path.exists(args.docs):
        sys.exit(f"ERROR: pubmed_documents.parquet not found at {args.docs}")
    if not os.path.exists(args.mappings):
        sys.exit(f"ERROR: sr_document_mappings.parquet not found at {args.mappings}")

    print(f"Loading corpus from: {args.docs}")
    df_docs = pd.read_parquet(args.docs)
    df_docs["pmid"] = df_docs["pmid"].astype(str).str.split(".").str[0]

    print(f"Loading mappings from: {args.mappings}")
    df_mappings = pd.read_parquet(args.mappings)
    df_mappings["sr_pmid"] = df_mappings["sr_pmid"].astype(str).str.split(".").str[0]
    df_mappings["pmid"]    = df_mappings["pmid"].astype(str).str.split(".").str[0]

    print(f"Corpus: {len(df_docs):,} documents")

    # ── Load model ──────────────────────────────────────────────────────────
    model_path = args.model
    print(f"\nLoading model: {model_path} ...")
    agent = laya.load(model_path, device=args.device)
    print(f"Model loaded on device: {agent.device}")

    # ── Output dir ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    per_doc_path = os.path.join(args.output_dir, "per_doc_deltas.csv")
    summary_csv  = os.path.join(args.output_dir, "summary.csv")
    summary_json = os.path.join(args.output_dir, "summary.json")

    all_summaries: list[dict] = []
    all_deltas:    list[pd.DataFrame] = []

    # ── Per-SR loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EXPANDED SPECIFICITY ANALYSIS — Full Corpus")
    print(f"Threshold   : {args.threshold}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Output dir  : {args.output_dir}")
    print(f"{'='*70}\n")

    for entry in tqdm(entries, desc="SRs", unit="SR"):
        sr_pmid = entry["sr_pmid"]
        ori     = entry["ori"]
        v2      = entry["v2"]

        # Build per-SR candidate frame (full corpus + is_inclusion labels)
        sr_map = df_mappings[df_mappings["sr_pmid"] == sr_pmid][["pmid", "is_inclusion"]]
        df_c   = df_docs[["pmid", "title", "abstract"]].merge(sr_map, on="pmid", how="left")
        df_c["is_inclusion"] = df_c["is_inclusion"].fillna(False).astype(bool)

        if args.limit:
            df_c = df_c.head(args.limit).copy()

        if df_c.empty:
            tqdm.write(f"  WARNING: {sr_pmid} — no candidates found, skipping.")
            continue

        n_inc = int(df_c["is_inclusion"].sum())
        tqdm.write(f"  {sr_pmid}: {len(df_c):,} docs | {n_inc} inclusions")

        t0 = time.time()
        df_results, summary = analyse_sr(
            agent=agent,
            sr_pmid=sr_pmid,
            ori=ori,
            v2=v2,
            df_corpus=df_c,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - t0

        summary["elapsed_s"] = round(elapsed, 1)
        all_summaries.append(summary)
        all_deltas.append(df_results)

        tqdm.write(
            f"  {sr_pmid}: ori recall={summary['ori_recall']:.3f} → v2 recall={summary['v2_recall']:.3f} | "
            f"FNs recovered={summary['fn_recovered']} | new FPs={summary['new_fp']} | "
            f"elapsed={elapsed:.0f}s"
        )

    # ── Write outputs ────────────────────────────────────────────────────────
    if all_deltas:
        df_all_deltas = pd.concat(all_deltas, ignore_index=True)
        df_all_deltas.to_csv(per_doc_path, index=False)
        print(f"\nPer-doc deltas : {per_doc_path}  ({len(df_all_deltas):,} rows)")

    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        # Put instructions last for readability
        col_order = [c for c in df_summary.columns if c not in ("ori", "v2")] + ["ori", "v2"]
        df_summary = df_summary[[c for c in col_order if c in df_summary.columns]]
        df_summary.to_csv(summary_csv, index=False)
        with open(summary_json, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"Summary CSV    : {summary_csv}")
        print(f"Summary JSON   : {summary_json}")

        # ── Print console summary table ─────────────────────────────────────
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'SR PMID':<12} {'N':<7} {'OriRec':>7} {'V2Rec':>7} {'RecΔ':>7}  {'NewFP':>7}  {'FNRec':>7}")
        print(f"{'-'*70}")
        for s in all_summaries:
            rec_delta = s["v2_recall"] - s["ori_recall"]
            print(
                f"{s['sr_pmid']:<12} {s['n_corpus']:<7,} "
                f"{s['ori_recall']:>7.3f} {s['v2_recall']:>7.3f} {rec_delta:>+7.3f}  "
                f"{s['new_fp']:>7,}  {s['fn_recovered']:>7,}"
            )
        print(f"{'='*70}")

    print("\nDone.")


if __name__ == "__main__":
    main()
