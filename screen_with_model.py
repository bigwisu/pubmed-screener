#!/usr/bin/env python3
"""
screen_with_model.py — Standalone benchmark screening engine for local fine-tuned models.

Screens candidate PubMed documents against systematic reviews across the 66-review
benchmark (Bentegeac et al. corpus) using local Laya decision models
(e.g., BioClinical-ModernBERT-large or ModernBERT-large).

Evaluates 100% of the candidate corpus (15,530 PubMed documents per review, ~1.02M
evaluations) without heuristic early stopping to isolate true decision boundaries,
calculating confusion matrix metrics (TP, FP, FN, TN, Recall, Precision, Specificity,
F1, and WSS@95/WSS@100).

Usage examples:
  # Screen single review with proposed BioClinical end-to-end model
  python benchmark/screen_with_model.py --model models/soces_pubmed_e2e --sr 22422870

  # Screen generic ModernBERT control on calibration reviews (5 SRs)
  python benchmark/screen_with_model.py --model models/soces_generic_e2e --test

  # Full 66 systematic review benchmark on local GPU/MPS
  python benchmark/screen_with_model.py --model models/soces_pubmed_e2e --batch-size 128
"""

import os
import sys
import time
import json
import sqlite3
import argparse
import threading
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add local Laya submodule to python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYA_DIR = os.path.join(ROOT_DIR, "laya")
if LAYA_DIR not in sys.path:
    sys.path.insert(0, LAYA_DIR)

from laya import load  # noqa: E402

# Standardized Noul instruction prompt (exact 1:1 parity with jev_client.py)
NOUL_INSTRUCTIONS = (
    "Given the systematic review objective in review_objective, "
    "does this paper contribute relevant evidence — even indirectly?\n"
    "True if: The paper reports empirical data on the topic or population described "
    "in review_objective, regardless of whether its study design or subpopulation "
    "exactly matches the review's primary methodology.\n"
    "False if: The paper's topic is entirely unrelated to review_objective, "
    "or it contains no data (e.g. editorial, commentary, opinion piece)."
)

# 5 calibration systematic reviews (PICO variety)
CALIBRATION_SR_PMIDS = [
    "22226047",  # Group B strep in infants (pediatric / infectious disease)
    "22323502",  # Cervical spine clearance in trauma (orthopedics / emergency)
    "22422870",  # Statin-induced myopathy (cardiology / pharmacology)
    "24157497",  # Direct oral anticoagulants in atrial fib (cardiology / hematology)
    "30326495",  # Bariatric surgery long-term outcomes (metabolic / surgery)
]

MODEL_SHORTCUTS = {
    "laya": "convaiinnovations/laya",
    "pubmed_e2e": "models/soces_pubmed_e2e",
    "generic_e2e": "models/soces_generic_e2e",
    "pubmed_frozen": "models/soces_pubmed_frozen",
    "generic_frozen": "models/soces_generic_frozen",
}


# ---------------------------------------------------------------------------
# Storage & Database Helpers
# ---------------------------------------------------------------------------

class ModelResultsDatabase:
    """Thread-safe SQLite storage for local model screening results."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS screening_results (
                    sr_pmid TEXT NOT NULL,
                    pmid TEXT NOT NULL,
                    relevance_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    predicted_include INTEGER NOT NULL,
                    is_inclusion INTEGER NOT NULL,
                    PRIMARY KEY (sr_pmid, pmid)
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_sr ON screening_results (sr_pmid);")
            conn.commit()
            conn.close()

    def save_screening_results(self, df_scored: pd.DataFrame):
        """Upsert scored candidates into SQLite."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            records = [
                (
                    str(r["sr_pmid"]),
                    str(r["pmid"]),
                    float(r["relevance_score"]),
                    float(r.get("confidence", 1.0)),
                    int(bool(r["predicted_include"])),
                    int(bool(r["is_inclusion"]))
                )
                for _, r in df_scored.iterrows()
            ]
            conn.executemany("""
                INSERT OR REPLACE INTO screening_results 
                (sr_pmid, pmid, relevance_score, confidence, predicted_include, is_inclusion)
                VALUES (?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
            conn.close()


def load_completed_sr_pmids(summary_path: str, target_count: int, force: bool = False) -> set[str]:
    """Return set of sr_pmid strings that already completed full screening."""
    if force or not os.path.exists(summary_path):
        return set()
    try:
        df = pd.read_csv(summary_path)
        if df.empty or "sr_pmid" not in df.columns or "n_candidates" not in df.columns:
            return set()
        df["sr_pmid"] = df["sr_pmid"].astype(str).str.split(".").str[0]
        completed = df[df["n_candidates"] >= target_count]["sr_pmid"].tolist()
        return set(completed)
    except Exception:
        return set()


def append_summary_row(summary_path: str, row_dict: dict):
    """Atomically append or update a review's summary row in CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    df_new = pd.DataFrame([row_dict])
    if os.path.exists(summary_path):
        try:
            df_existing = pd.read_csv(summary_path)
            df_existing["sr_pmid"] = df_existing["sr_pmid"].astype(str).str.split(".").str[0]
            df_filtered = df_existing[df_existing["sr_pmid"] != str(row_dict["sr_pmid"])]
            df_combined = pd.concat([df_filtered, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    tmp_path = summary_path + ".tmp"
    df_combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, summary_path)


def append_results_rows(results_path: str, df_scored: pd.DataFrame, db: ModelResultsDatabase | None = None):
    """
    Persist per-candidate decisions across formats:
    1. Primary: Save to indexed 'screening_results' table in SQLite.
    2. Streaming CSV: Append rows in O(1) time without loading 1M-line CSV into RAM.
    3. Parquet Export: Keep columnar parquet table in sync.
    """
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    result_cols = ["sr_pmid", "pmid", "relevance_score", "confidence", "predicted_include", "is_inclusion"]
    for c in result_cols:
        if c not in df_scored.columns:
            df_scored[c] = 1.0 if c == "confidence" else 0
    df_to_save = df_scored[result_cols].copy()
    sr_pmid = str(df_to_save["sr_pmid"].iloc[0])

    if db is not None:
        db.save_screening_results(df_to_save)

    # 2. Streaming CSV Append (O(1) memory)
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        df_to_save.to_csv(results_path, index=False, mode="w")
    else:
        df_to_save.to_csv(results_path, index=False, header=False, mode="a")

    # 3. Columnar Parquet Sync
    parquet_path = results_path.replace(".csv", ".parquet")
    try:
        if os.path.exists(parquet_path):
            df_pq = pd.read_parquet(parquet_path)
            df_pq = df_pq[df_pq["sr_pmid"].astype(str) != sr_pmid]
            df_combined = pd.concat([df_pq, df_to_save], ignore_index=True)
        else:
            df_combined = df_to_save
        df_combined.to_parquet(parquet_path, index=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """Compute complete confusion matrix and clinical screening metrics."""
    df = df.copy()
    df["predicted_include"] = df["relevance_score"] >= threshold

    tp = int(((df["predicted_include"]) & (df["is_inclusion"])).sum())
    fp = int(((df["predicted_include"]) & (~df["is_inclusion"])).sum())
    fn = int(((~df["predicted_include"]) & (df["is_inclusion"])).sum())
    tn = int(((~df["predicted_include"]) & (~df["is_inclusion"])).sum())
    total = len(df)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    window = int(df["predicted_include"].sum())

    # Work Saved over Sampling (Cohen et al. 2006)
    # WSS@95 = (TN + FN) / N - (1 - 0.95) if recall >= 0.95 else 0.0
    wss95 = ((tn + fn) / total - (1.0 - 0.95)) if (total > 0 and recall >= 0.95) else 0.0
    wss100 = ((tn + fn) / total - (1.0 - 1.00)) if (total > 0 and recall >= 1.00) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "f1": f1,
        "window": window,
        "wss95": wss95,
        "wss100": wss100,
        "n_gt": int(df["is_inclusion"].sum()),
    }


# ---------------------------------------------------------------------------
# Model Screening Loop
# ---------------------------------------------------------------------------

def screen_sr_with_model(
    agent,
    sr_pmid: str,
    sr_query_text: str,
    df_candidates: pd.DataFrame,
    threshold: float = 0.5,
    batch_size: int = 64,
    model_name: str = "model",
    instruction: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Screen candidates for one SR using local batched model inference."""
    if df_candidates.empty:
        return df_candidates, {}

    df = df_candidates.copy()

    # Prioritize curated direct PICO instruction; fallback to natural question
    if instruction:
        inst_text = instruction
    else:
        inst_text = f"Does this study report empirical findings on {sr_query_text}?"

    questions = {
        "relevance": {
            "type": "noul",
            "instructions": inst_text,
        }
    }

    # Construct clean text states for maximum encoder token headroom
    states = []
    for _, row in df.iterrows():
        p_title = str(row["title"]) if pd.notna(row["title"]) else ""
        p_abstract = str(row["abstract"]) if pd.notna(row["abstract"]) else ""
        states.append(f"Title: {p_title}\nAbstract: {p_abstract}")

    n_items = len(states)
    scores = []
    confidences = []

    # Run in batches with inner tqdm progress bar
    with tqdm(total=n_items, desc=f"  Screening {sr_pmid} ({model_name})", unit="doc", leave=False) as pbar:
        for i in range(0, n_items, batch_size):
            batch_states = states[i:i + batch_size]
            results = agent.predict_batch(batch_states, questions)
            for res in results:
                ans = res["answers"]["relevance"]
                scores.append(float(ans.get("noul", 0.0)))
                confidences.append(float(ans.get("confidence", 1.0)))
            pbar.update(len(batch_states))

    df["relevance_score"] = scores
    df["confidence"] = confidences
    df["sr_pmid"] = sr_pmid
    df["predicted_include"] = df["relevance_score"] >= threshold

    metrics = compute_metrics(df, threshold)
    return df, metrics


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark screening engine for local fine-tuned System-1 models (e.g. BioClinical-ModernBERT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="models/soces_pubmed_e2e",
        help="Path or shortcut to model checkpoint (e.g. models/soces_pubmed_e2e, pubmed_e2e, generic_e2e)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to load model on ('cuda', 'mps', 'cpu', or auto if None)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Inference batch size for agent.predict_batch (e.g. 64 or 128)",
    )
    parser.add_argument(
        "--sr", type=str, default=None,
        help="Screen a single systematic review PMID (e.g. 22422870)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Screen the 5 calibration systematic reviews only",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of candidate documents evaluated per SR (useful for testing)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for inclusion",
    )
    parser.add_argument(
        "--prefiltered", action="store_true",
        help="Screen only the pre-filtered candidate mappings instead of the full 15,530 PubMed corpus",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-screening even if marked completed in summary CSV",
    )
    parser.add_argument(
        "--use-abstract", action="store_true",
        help="Use raw published abstract instead of curated prospective review_objective",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to benchmark_candidates.parquet (defaults to benchmark/benchmark_candidates.parquet)",
    )
    parser.add_argument(
        "--include-excluded", action="store_true",
        help="Include the 4 flagged/excluded reviews (screens all 70 instead of 66)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory to save summary and result outputs",
    )
    parser.add_argument(
        "--objectives", type=str, default=None,
        help=(
            "Path to an alternative sr_review_objectives JSON file "
            "(e.g. benchmark/rq3_instruction_counterfactual/alt_objectives.json). "
            "Entries in this file override instruction_inclusive for matching sr_pmid values; "
            "all other SRs fall back to the default objectives file. "
            "Original instructions are preserved in the alt file under 'instruction_inclusive_ORIGINAL'. "
            "The primary sr_review_objectives.json is never modified."
        ),
    )

    args = parser.parse_args()

    # Resolve model path
    model_path = MODEL_SHORTCUTS.get(args.model, args.model)
    if not os.path.exists(model_path) and ("/" not in model_path or model_path.startswith("models/")):
        parser.error(f"Model path '{model_path}' not found! Check directory or pass valid path.")

    # Derive model slug for output files
    model_slug = Path(model_path).name
    if not model_slug or model_slug == ".":
        model_slug = Path(model_path).parent.name

    # Load Model
    print(f"Loading model: {model_path} ...")
    agent = load(model_path, device=args.device)
    print(f"Model loaded successfully on device: {agent.device}")

    # Load Datasets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data or os.path.join(script_dir, "benchmark_candidates.parquet")
    docs_path = os.path.join(script_dir, "pubmed_documents.parquet")
    mappings_path = os.path.join(script_dir, "sr_document_mappings.parquet")
    objectives_path = os.path.join(script_dir, "sr_review_objectives.json")

    full_corpus = not args.prefiltered

    if full_corpus and os.path.exists(docs_path) and os.path.exists(mappings_path):
        print(f"Loading standalone PubMed documents from: {docs_path}")
        df_docs = pd.read_parquet(docs_path)
        df_docs["pmid"] = df_docs["pmid"].astype(str).str.split(".").str[0]
        df_mappings = pd.read_parquet(mappings_path)
        df_mappings["sr_pmid"] = df_mappings["sr_pmid"].astype(str).str.split(".").str[0]
        df_mappings["pmid"] = df_mappings["pmid"].astype(str).str.split(".").str[0]
        df_candidates_all = None
    elif os.path.exists(data_path):
        print(f"Loading candidates from: {data_path}")
        if data_path.endswith(".parquet"):
            df_candidates_all = pd.read_parquet(data_path)
        else:
            df_candidates_all = pd.read_json(data_path, lines=True)
        df_candidates_all["sr_pmid"] = df_candidates_all["sr_pmid"].astype(str).str.split(".").str[0]
        df_candidates_all["pmid"] = df_candidates_all["pmid"].astype(str).str.split(".").str[0]
        df_docs = None
        df_mappings = None
    else:
        raise FileNotFoundError(f"Candidate dataset not found at {data_path} or {docs_path}!")

    # Load Review Objectives & Systematic Reviews
    with open(objectives_path, "r") as f:
        sr_objectives_data = json.load(f)

    # Apply alternative objectives override (RQ3 instruction counterfactual).
    # Entries in --objectives override instruction_inclusive for matching sr_pmid only.
    # The primary objectives file is never modified; originals are in instruction_inclusive_ORIGINAL.
    if args.objectives:
        alt_path = os.path.abspath(args.objectives)
        if not os.path.exists(alt_path):
            parser.error(f"--objectives file not found: {alt_path}")
        with open(alt_path, "r") as f:
            alt_data = json.load(f)
        alt_map = {str(entry["sr_pmid"]).split(".")[0]: entry for entry in alt_data}
        patched = 0
        for item in sr_objectives_data:
            pmid = str(item["sr_pmid"]).split(".")[0]
            if pmid in alt_map:
                alt = alt_map[pmid]
                item["instruction_inclusive"] = alt["instruction_inclusive"]
                patched += 1
        print(f"[--objectives] Patched {patched} SR instruction(s) from: {alt_path}")
        print(f"               Original instructions preserved in alt file under 'instruction_inclusive_ORIGINAL'.")

    # Filter systematic reviews
    target_pmids = None
    if args.sr:
        target_pmids = {p.strip().split(".")[0] for p in str(args.sr).replace(",", " ").split() if p.strip()}
    elif args.test:
        target_pmids = set(CALIBRATION_SR_PMIDS)

    sr_rows = []
    for item in sr_objectives_data:
        p = str(item["sr_pmid"]).split(".")[0]
        is_ex = bool(item.get("excluded_from_study", False))
        if not args.include_excluded and is_ex:
            continue
        if target_pmids and p not in target_pmids:
            continue
        sr_rows.append({
            "sr_pmid": p,
            "title": item.get("title", ""),
            "review_objective": item.get("review_objective", ""),
            "abstract": item.get("stripped_abstract", item.get("title", "")),
            "instruction": item.get("instruction_inclusive") or item.get("instruction", ""),
        })

    df_srs = pd.DataFrame(sr_rows)
    if df_srs.empty:
        print("No systematic reviews matched criteria.")
        sys.exit(0)

    # Output paths
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, f"{model_slug}_summary.csv")
    results_path = os.path.join(args.output_dir, f"{model_slug}_results.csv")
    sqlite_path = os.path.join(args.output_dir, f"{model_slug}_results.sqlite")
    db = ModelResultsDatabase(sqlite_path)

    corpus_desc = "Full Corpus (15,530 PubMed documents per SR)" if full_corpus else "Pre-filtered search mappings"
    objective_desc = "Raw SR abstract (--use-abstract)" if args.use_abstract else "Curated PICO review_objective"

    print("=" * 70)
    print("BENCHMARK SCREENING ENGINE: LOCAL MODEL")
    print(f"Model       : {model_slug} ({model_path})")
    print(f"Device      : {agent.device}")
    print(f"Batch Size  : {args.batch_size}")
    print(f"Corpus      : {corpus_desc}")
    print(f"Objective   : {objective_desc}")
    print(f"Threshold   : {args.threshold}")
    print(f"Reviews     : {len(df_srs)} systematic reviews")
    if args.limit:
        print(f"Limit       : {args.limit} candidates per SR")
    print(f"Results CSV : {results_path}")
    print(f"Summary CSV : {summary_path}")
    print(f"SQLite DB   : {sqlite_path}")
    print("=" * 70)
    print()

    target_docs = args.limit if args.limit is not None else (len(df_docs) if (full_corpus and df_docs is not None) else 15530)
    completed_pmids = load_completed_sr_pmids(summary_path, target_docs, force=args.force)

    if completed_pmids and not args.force:
        print(f"Resume State : {len(completed_pmids)} SRs already completed ({target_docs:,} candidates).")
        print(f"             : {len(df_srs) - len(completed_pmids)} systematic reviews remaining to screen.")
        print()

    try:
        for _, sr_row in tqdm(df_srs.iterrows(), total=len(df_srs), desc=f"Benchmarking {model_slug}"):
            sr_pmid = str(sr_row["sr_pmid"])
            if sr_pmid in completed_pmids and not args.force:
                tqdm.write(f"  {sr_pmid}: already completed ({target_docs} candidates evaluated). Skipping.")
                continue

            if args.use_abstract:
                sr_query_text = sr_row["abstract"] or sr_row["title"]
            else:
                sr_query_text = sr_row["review_objective"] or sr_row["title"]

            # Load candidates for this SR
            if full_corpus:
                if df_docs is not None and df_mappings is not None:
                    sr_map = df_mappings[df_mappings["sr_pmid"] == sr_pmid][["pmid", "is_inclusion"]]
                    df_c = df_docs[["pmid", "title", "abstract"]].merge(sr_map, on="pmid", how="left")
                    df_c["is_inclusion"] = df_c["is_inclusion"].fillna(False).astype(bool)
                elif df_candidates_all is not None:
                    unique_docs = df_candidates_all[["pmid", "title", "abstract"]].drop_duplicates("pmid")
                    sr_map = df_candidates_all[df_candidates_all["sr_pmid"] == sr_pmid][["pmid", "is_inclusion"]]
                    df_c = unique_docs.merge(sr_map, on="pmid", how="left")
                    df_c["is_inclusion"] = df_c["is_inclusion"].fillna(False).astype(bool)
            else:
                if df_candidates_all is not None:
                    df_c = df_candidates_all[df_candidates_all["sr_pmid"] == sr_pmid][["pmid", "title", "abstract", "is_inclusion"]].copy()
                elif df_mappings is not None and df_docs is not None:
                    sr_map = df_mappings[df_mappings["sr_pmid"] == sr_pmid]
                    df_c = sr_map.merge(df_docs[["pmid", "title", "abstract"]], on="pmid", how="inner")
                    df_c["is_inclusion"] = df_c["is_inclusion"].fillna(False).astype(bool)

            if args.limit is not None and args.limit > 0:
                df_c = df_c.head(args.limit).copy()

            if df_c.empty:
                tqdm.write(f"  {sr_pmid}: no candidates found, skipping.")
                continue

            t0 = time.time()
            df_scored, metrics = screen_sr_with_model(
                agent=agent,
                sr_pmid=sr_pmid,
                sr_query_text=sr_query_text,
                df_candidates=df_c,
                threshold=args.threshold,
                batch_size=args.batch_size,
                model_name=model_slug,
                instruction=sr_row.get("instruction") or None,
            )
            elapsed = time.time() - t0
            throughput = len(df_scored) / elapsed if elapsed > 0 else 0.0

            summary_row = {
                "model": model_slug,
                "sr_pmid": sr_pmid,
                "sr_title": sr_row["title"],
                "research_objective": sr_query_text,
                "n_candidates": len(df_scored),
                "n_gt": metrics["n_gt"],
                "window": metrics["window"],
                "recall": round(metrics["recall"], 4),
                "precision": round(metrics["precision"], 4),
                "specificity": round(metrics["specificity"], 4),
                "f1": round(metrics["f1"], 4),
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
                "wss95": round(metrics["wss95"], 4),
                "wss100": round(metrics["wss100"], 4),
                "elapsed_s": round(elapsed, 1),
                "docs_per_s": round(throughput, 1),
            }

            append_summary_row(summary_path, summary_row)
            append_results_rows(results_path, df_scored, db=db)

            tqdm.write(
                f"  {sr_pmid}: recall={metrics['recall']:.1%}  "
                f"precision={metrics['precision']:.1%}  "
                f"specificity={metrics['specificity']:.1%}  "
                f"f1={metrics['f1']:.3f}  "
                f"WSS@95={metrics['wss95']:.1%}  "
                f"window={metrics['window']}/{metrics['n_gt']} GT  "
                f"({elapsed:.1f}s, {throughput:.1f} doc/s)"
            )

    except KeyboardInterrupt:
        print("\n[Screening paused by user (Ctrl+C)]")
        print(f"All completed reviews were saved to:")
        print(f"  Summary : {summary_path}")
        print(f"  Results : {results_path}")
        print(f"  SQLite  : {sqlite_path}")
        print("To resume later, simply rerun the same command (completed SRs will auto-skip).")
        sys.exit(0)

    print()
    print("=" * 70)
    print("BENCHMARK SCREENING COMPLETE")
    print(f"Summary table saved to: {summary_path}")
    print(f"Results table saved to: {results_path}")
    print(f"Parquet table saved to: {results_path.replace('.csv', '.parquet')}")
    print(f"SQLite table saved to : {sqlite_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
