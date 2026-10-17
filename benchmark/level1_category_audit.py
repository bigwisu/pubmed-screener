#!/usr/bin/env python3
"""
level1_category_audit.py -- descriptive checks supporting the Level 1 text and Appendix C.

Reads the archived Parquet benchmark results and the Level 1 miss table.
It does NOT change any category; it reports how the review-level rules in
rq3_mechanical_analysis.py relate to document-level facts (abstract length,
smoking string search, recall of reviews with misses).

Usage:
  .venv/bin/python benchmark/level1_category_audit.py
"""
import pandas as pd

THETA = 0.028
CAL = {"22226047", "22323502", "22422870", "22777524"}

res = pd.read_parquet("pubmed-screener/outputs/soces_pico_laya_e2e_results.parquet")
res["sr_pmid"] = res["sr_pmid"].astype(str).str.split(".").str[0]
res["pmid"] = res["pmid"].astype(str).str.split(".").str[0]
docs = pd.read_parquet("benchmark/pubmed_documents.parquet", columns=["pmid", "title", "abstract"])
docs["pmid"] = docs["pmid"].astype(str).str.split(".").str[0]
miss_tab = pd.read_csv("publication/tables/rq3_miss_table.csv")
miss_tab["sr_pmid"] = miss_tab["sr_pmid"].astype(str)
cat = miss_tab.set_index("sr_pmid")["category"].astype(str).to_dict()

res = res[~res.sr_pmid.isin(CAL)]
fn = res[(res.is_inclusion == 1) & (res.relevance_score < THETA)].merge(docs, on="pmid", how="left")
fn["abs_len"] = fn["abstract"].fillna("").str.len()
fn["category"] = fn.sr_pmid.map(cat)

print(f"false negatives: {len(fn)} in {fn.sr_pmid.nunique()} reviews")
print("\nMisses per review-level category:")
print(fn.groupby("category").size().to_string())
print("\nMisses with abstract < 100 characters, by review-level category:")
short = fn[fn.abs_len < 100]
print(short.groupby("category").size().to_string(), f"\ntotal short: {len(short)}")
print("abstract lengths of those records:", sorted(short.abs_len.unique()))

# Recall of reviews that have >= 1 miss
per = miss_tab[["sr_pmid", "miss_count", "recall_pct"]]
print("\nReviews with >=1 miss and recall >= 99.0%:", int((per.recall_pct >= 99.0).sum()),
      "(misses:", int(per[per.recall_pct >= 99.0].miss_count.sum()), ")")
print("Highest recall among reviews with a miss:", per.recall_pct.max())

# SR 24046285: does 'smok' occur in title/abstract of found vs missed inclusions?
inc = res[(res.sr_pmid == "24046285") & (res.is_inclusion == 1)].merge(docs, on="pmid", how="left")
inc["has_smok"] = [("smok" in (str(a) + " " + str(b)).lower()) for a, b in zip(inc.title, inc.abstract)]
inc["missed"] = inc.relevance_score < THETA
print("\nSR 24046285 inclusions containing 'smok' in title/abstract:")
print(inc.groupby("missed").has_smok.agg(["sum", "count"]).rename(index={True: "missed", False: "found"}).to_string())

# SR 31200992: does the '10 years' clause appear in inclusion abstracts?
import re
pat = re.compile(r"\b(10|ten)[ -]years?\b", re.I)
inc2 = res[(res.sr_pmid == "31200992") & (res.is_inclusion == 1)].merge(docs, on="pmid", how="left")
inc2["has10"] = [bool(pat.search(str(a) + " " + str(b))) for a, b in zip(inc2.title, inc2.abstract)]
inc2["missed"] = inc2.relevance_score < THETA
print("\nSR 31200992 inclusions with a '10 years'/'ten years' string in title/abstract:")
print(inc2.groupby("missed").has10.agg(["sum", "count"]).rename(index={True: "missed", False: "found"}).to_string())

# Rank of the five Level 2 targets by miss count
print("\nReviews ranked by miss count (top 12):")
print(miss_tab.sort_values("miss_count", ascending=False).head(12)[["sr_pmid", "miss_count"]].to_string(index=False))
