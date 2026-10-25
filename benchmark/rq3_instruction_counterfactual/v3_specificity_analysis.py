#!/usr/bin/env python3
"""
v3_specificity_analysis.py — Score v3 instructions only; ori scores loaded from existing CSV.

ori and v2 are frozen. This script:
  1. Loads ori scores + ground-truth labels from outputs/expanded_specificity/per_doc_deltas.csv
  2. Scores each document with the v3 instruction only
  3. Computes delta (v3 - ori) and confusion-matrix shift (ori → v3)
  4. Writes outputs/expanded_specificity_v3/

Only the 3 SRs that have a v3 entry are processed (27802478, 27802505, 30409774).

Usage (from project root):
  .venv/bin/python benchmark/rq3_instruction_counterfactual/v3_specificity_analysis.py \\
      --model models/soces_pico_laya_e2e --batch-size 64

  # Single SR smoke-test:
  .venv/bin/python benchmark/rq3_instruction_counterfactual/v3_specificity_analysis.py \\
      --model models/soces_pico_laya_e2e --sr 27802505 --batch-size 32
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_LAYA_DIR   = os.path.join(_ROOT_DIR, "laya")

if _LAYA_DIR not in sys.path:
    sys.path.insert(0, _LAYA_DIR)

import laya  # noqa: E402

THRESHOLD       = 0.028
SPEC_JSON       = os.path.join(_SCRIPT_DIR, "expanded_specifity.json")
DOCS_PARQUET    = os.path.join(_ROOT_DIR, "benchmark", "pubmed_documents.parquet")
MAPPINGS_PARQUET= os.path.join(_ROOT_DIR, "benchmark", "sr_document_mappings.parquet")
ORI_SCORES_CSV  = os.path.join(_ROOT_DIR, "outputs", "expanded_specificity", "per_doc_deltas.csv")
DEFAULT_OUTPUT  = os.path.join(_ROOT_DIR, "outputs", "expanded_specificity_v3")
DEFAULT_MODEL   = os.path.join(_ROOT_DIR, "models", "soces_pico_laya_e2e")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_batch(agent, instruction: str, states: list[str]) -> list[float]:
    questions = {"relevance": {"type": "noul", "instructions": instruction.strip()}}
    results = agent.predict_batch(states, questions)
    return [float(r["answers"]["relevance"].get("noul", 0.0)) for r in results]


def build_states(df: pd.DataFrame) -> list[str]:
    states = []
    for _, row in df.iterrows():
        title    = str(row["title"])    if pd.notna(row.get("title"))    else ""
        abstract = str(row["abstract"]) if pd.notna(row.get("abstract")) else ""
        states.append(f"Title: {title.strip()}\nAbstract: {abstract.strip()}")
    return states


def cm_metrics(inc: pd.Series, pos: pd.Series) -> dict:
    return {
        "tp": int(( inc &  pos).sum()),
        "fp": int(( inc & ~pos).sum()),
        "fn": int((~inc &  pos).sum()),
        "tn": int((~inc & ~pos).sum()),
    }


CHANGE_CONDITIONS = [
    (lambda o, v, p:  o &  v &  p,  "TP→TP"),
    (lambda o, v, p:  o &  v & ~p,  "FP→FP"),
    (lambda o, v, p: ~o & ~v &  p,  "FN→FN"),
    (lambda o, v, p: ~o & ~v & ~p,  "TN→TN"),
    (lambda o, v, p: ~o &  v &  p,  "FN→TP (recovered)"),
    (lambda o, v, p: ~o &  v & ~p,  "TN→FP (new_fp)"),
    (lambda o, v, p:  o & ~v &  p,  "TP→FN (lost)"),
    (lambda o, v, p:  o & ~v & ~p,  "FP→TN (narrowed_out)"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="v3 instruction scoring — ori scores loaded from existing CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold",  type=float, default=THRESHOLD)
    parser.add_argument("--sr",         type=str, default=None,
                        help="Restrict to one SR PMID")
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--ori-csv",    type=str, default=ORI_SCORES_CSV,
                        help="Path to existing per_doc_deltas.csv with ori scores")
    args = parser.parse_args()

    # ── Load spec — only entries that have v3 ────────────────────────────────
    with open(SPEC_JSON) as f:
        spec_entries = json.load(f)

    entries = []
    for e in spec_entries:
        sr = str(e.get("pmid") or e.get("sr_pmid", "")).split(".")[0]
        v3 = e.get("v3")
        if not sr or not v3:
            continue
        entries.append({"sr_pmid": sr, "ori_instruction": e["ori"], "v3": v3})

    if args.sr:
        target = args.sr.strip().split(".")[0]
        entries = [e for e in entries if e["sr_pmid"] == target]
        if not entries:
            sys.exit(f"ERROR: SR {args.sr} has no v3 entry in spec file.")

    print(f"SRs with v3: {[e['sr_pmid'] for e in entries]}")

    # ── Load frozen ori scores ───────────────────────────────────────────────
    if not os.path.exists(args.ori_csv):
        sys.exit(f"ERROR: ori scores CSV not found at {args.ori_csv}\n"
                 f"Run expanded_specificity_analysis.py first.")

    df_ori = pd.read_csv(args.ori_csv)
    df_ori["pmid"]    = df_ori["pmid"].astype(str).str.split(".").str[0]
    df_ori["sr_pmid"] = df_ori["sr_pmid"].astype(str).str.split(".").str[0]

    # ── Load corpus text (needed for v3 scoring) ─────────────────────────────
    print(f"Loading corpus: {DOCS_PARQUET}")
    df_docs = pd.read_parquet(DOCS_PARQUET)
    df_docs["pmid"] = df_docs["pmid"].astype(str).str.split(".").str[0]

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    agent = laya.load(args.model, device=args.device)
    print(f"Device: {agent.device}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    all_rows:      list[pd.DataFrame] = []
    all_summaries: list[dict]         = []

    for entry in tqdm(entries, desc="SRs", unit="SR"):
        sr_pmid = entry["sr_pmid"]
        v3      = entry["v3"]

        # Pull ori scores + labels for this SR
        df_sr_ori = df_ori[df_ori["sr_pmid"] == sr_pmid][["pmid", "ori_score", "is_inclusion", "ori_include"]].copy()
        df_sr_ori["ori_include"]  = df_sr_ori["ori_include"].astype(bool)
        df_sr_ori["is_inclusion"] = df_sr_ori["is_inclusion"].astype(bool)

        # Merge corpus text
        df_c = df_sr_ori.merge(df_docs[["pmid", "title", "abstract"]], on="pmid", how="left")

        if args.limit:
            df_c = df_c.head(args.limit).copy()

        tqdm.write(f"  {sr_pmid}: {len(df_c):,} docs | {df_c['is_inclusion'].sum()} inclusions")

        # Score v3
        states = build_states(df_c)
        v3_scores: list[float] = []
        t0 = time.time()
        with tqdm(total=len(states), desc=f"    v3 {sr_pmid}", unit="doc", leave=False) as pbar:
            for i in range(0, len(states), args.batch_size):
                batch = states[i:i + args.batch_size]
                v3_scores.extend(score_batch(agent, v3, batch))
                pbar.update(len(batch))
        elapsed = time.time() - t0

        df_c = df_c.copy()
        df_c["v3_score"]   = v3_scores
        df_c["delta_v3"]   = df_c["v3_score"] - df_c["ori_score"]
        df_c["v3_include"] = df_c["v3_score"] >= args.threshold
        df_c["sr_pmid"]    = sr_pmid

        # Change categories (ori → v3)
        o = df_c["ori_include"]
        v = df_c["v3_include"]
        p = df_c["is_inclusion"]
        df_c["change_v3"] = "other"
        for fn, label in CHANGE_CONDITIONS:
            df_c.loc[fn(o, v, p), "change_v3"] = label

        # Confusion matrices
        ori_cm = cm_metrics(o, p)
        v3_cm  = cm_metrics(v, p)

        def recall(m):    return m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0.0
        def precision(m): return m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0.0

        fn_mask = ~o & p
        summary = {
            "sr_pmid": sr_pmid,
            "n_corpus": len(df_c),
            "n_inclusions": int(p.sum()),
            "threshold": args.threshold,
            "ori_recall":    round(recall(ori_cm), 4),
            "ori_precision": round(precision(ori_cm), 4),
            "v3_tp": v3_cm["tp"],  "v3_fp": v3_cm["fp"],
            "v3_fn": v3_cm["fn"],  "v3_tn": v3_cm["tn"],
            "v3_recall":    round(recall(v3_cm), 4),
            "v3_precision": round(precision(v3_cm), 4),
            "v3_fn_recovered":    int((df_c["change_v3"] == "FN→TP (recovered)").sum()),
            "v3_new_fp":          int((df_c["change_v3"] == "TN→FP (new_fp)").sum()),
            "v3_tp_lost":         int((df_c["change_v3"] == "TP→FN (lost)").sum()),
            "v3_fp_narrowed_out": int((df_c["change_v3"] == "FP→TN (narrowed_out)").sum()),
            "v3_delta_mean":      round(float(df_c["delta_v3"].mean()), 6),
            "v3_delta_median":    round(float(df_c["delta_v3"].median()), 6),
            "v3_fn_delta_mean":   round(float(df_c.loc[fn_mask, "delta_v3"].mean()), 6) if fn_mask.sum() > 0 else None,
            "v3_fn_delta_median": round(float(df_c.loc[fn_mask, "delta_v3"].median()), 6) if fn_mask.sum() > 0 else None,
            "elapsed_s": round(elapsed, 1),
            "v3": v3,
        }
        all_summaries.append(summary)
        all_rows.append(df_c[["sr_pmid", "pmid", "is_inclusion", "ori_score", "ori_include",
                               "v3_score", "delta_v3", "v3_include", "change_v3"]])

        tqdm.write(
            f"  {sr_pmid}: ori recall={summary['ori_recall']:.3f} → v3 recall={summary['v3_recall']:.3f} | "
            f"FNs recovered={summary['v3_fn_recovered']} | new FPs={summary['v3_new_fp']} | "
            f"elapsed={elapsed:.0f}s"
        )

    # ── Write outputs ────────────────────────────────────────────────────────
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            os.path.join(args.output_dir, "per_doc_deltas_v3.csv"), index=False)

    if all_summaries:
        df_s = pd.DataFrame(all_summaries)
        df_s.to_csv(os.path.join(args.output_dir, "summary_v3.csv"), index=False)
        with open(os.path.join(args.output_dir, "summary_v3.json"), "w") as f:
            json.dump(all_summaries, f, indent=2)

        print(f"\n{'='*70}")
        print("V3 RESULTS")
        print(f"{'SR PMID':<12} {'oriRec':>7} {'v3Rec':>7} {'RecΔ':>7}  {'NewFP':>7}  {'FNRec':>7}")
        print(f"{'-'*70}")
        for s in all_summaries:
            print(f"{s['sr_pmid']:<12} {s['ori_recall']:>7.3f} {s['v3_recall']:>7.3f} "
                  f"{s['v3_recall']-s['ori_recall']:>+7.3f}  {s['v3_new_fp']:>7,}  {s['v3_fn_recovered']:>7,}")
        print(f"{'='*70}")
        print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
