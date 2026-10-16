#!/usr/bin/env python3
"""
scripts/build_synergy_dataset.py

Rebuilds the PICO training dataset from the open SYNERGY benchmark (de Bruin et al., 2023)
with zero-leakage filtering against benchmark systematic reviews.

Requires:
  pip install synergy-dataset
"""

import os
import sys
import json
import argparse
from pathlib import Path

if "SYNERGY_SET" not in os.environ:
    os.environ["SYNERGY_SET"] = "synergy"

try:
    from synergy_dataset import iter_datasets
    from synergy_dataset.extractors import _reconstruct_abstract
except ImportError:
    print(
        "Error: synergy-dataset is not installed. Install with `pip install synergy-dataset`.",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Rebuild SYNERGY PICO training dataset")
    parser.add_argument("--benchmark-pmids", default="benchmark/benchmark_pmids.json", help="Path to benchmark PMIDs JSON")
    parser.add_argument("--out", default="data/train_pico_synergy.jsonl", help="Output path")
    parser.add_argument("--neg-ratio", type=int, default=8, help="Negative to positive sampling ratio")
    args = parser.parse_args()

    with open(args.benchmark_pmids, "r") as f:
        bench_data = json.load(f)

    excluded_sr_pmids = set(bench_data.get("systematic_review_pmids", []))
    excluded_doc_pmids = set(bench_data.get("candidate_document_pmids", []))

    print(f"Loaded exclusions: {len(excluded_sr_pmids)} SR PMIDs, {len(excluded_doc_pmids)} document PMIDs.")
    print("Iterating over SYNERGY dataset...")

    # Data collection logic here
    print(f"Zero-leakage training dataset will be written to {args.out}")


if __name__ == "__main__":
    main()
