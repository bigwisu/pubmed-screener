#!/usr/bin/env python3
"""
level1_category_audit.py -- descriptive checks supporting the Level 1 text and Appendix C.

Reads the archived Parquet benchmark results and the Level 1 miss table.
It reports document-level facts about the misses (abstract length, a
'10 years' string search for SR 31200992, recall of reviews with misses).

Usage:
  .venv/bin/python benchmark/level1_category_audit.py
"""
import pandas as pd

THETA = 0.028
CAL = {"22226047", "22323502", "22422870", "22777524"}

res = pd.read_parquet("outputs/soces_pico_laya_e2e_results.parquet")
res["sr_pmid"] = res["sr_pmid"].astype(str).str.split(".").str[0]
res["pmid"] = res["pmid"].astype(str).str.split(".").str[0]
docs = pd.read_parquet("benchmark/pubmed_documents.parquet", columns=["pmid", "title", "abstract"])
docs["pmid"] = docs["pmid"].astype(str).str.split(".").str[0]
miss_tab = pd.read_csv("publication/tables/rq3_miss_table.csv")
miss_tab["sr_pmid"] = miss_tab["sr_pmid"].astype(str)

res = res[~res.sr_pmid.isin(CAL)]
docs = docs.drop_duplicates("pmid")
docs["abs_len"] = docs["abstract"].fillna("").str.len()
inc_all = res[res.is_inclusion == 1].merge(docs[["pmid", "abs_len", "abstract"]], on="pmid", how="left")
inc_all["short"] = inc_all.abs_len < 100
inc_all["miss"] = inc_all.relevance_score < THETA
fn = inc_all[inc_all.miss]

print(f"false negatives: {len(fn)} in {fn.sr_pmid.nunique()} reviews")
print("\nMisses by abstract length (< 100 characters = short):")
print(fn.groupby("short").size().rename("misses").to_string())
short = fn[fn.short]
print(f"\nShort-abstract misses: {len(short)} in {short.sr_pmid.nunique()} reviews")
print("literal '[No Abstract]' placeholders among misses:", int((fn.abstract == "[No Abstract]").sum()))
print("abstract lengths of short-abstract misses:", sorted(short.abs_len.unique()))

# Recall of reviews that have >= 1 miss
per = miss_tab[["sr_pmid", "miss_count", "recall_pct"]]
print("\nReviews with >=1 miss and recall >= 99.0%:", int((per.recall_pct >= 99.0).sum()),
      "(misses:", int(per[per.recall_pct >= 99.0].miss_count.sum()), ")")
print("Highest recall among reviews with a miss:", per.recall_pct.max())

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
