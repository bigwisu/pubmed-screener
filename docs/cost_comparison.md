# Hosted LLM Cost Comparison for the 1,024,980-Evaluation Benchmark

Token counts are derived from actual billing records for `typesafe/jev-1.13` (mean 841 input / 24 output tokens per call); all other models assume the same input profile. GCP compute cost uses spot instance pricing on the Jakarta region ($107.61/month, 10.24 hours at 27.8 docs/s). Hosted model prices are as listed on OpenRouter (retrieved October 18, 2026); batch pricing applies a 50% discount where available and is subject to change.

| Model | $/M input | $/M output | Non-batch | Batch | Optimal (Jakarta spot) |
|---|---|---|---|---|---|
| `openai/gpt-6-luna` | $0.10 | $0.50 | ~$99 | ~$49 | ~$49 |
| `anthropic/claude-haiku-5.5` | $0.10 | $0.50 | ~$99 | ~$49 | ~$49 |
| `typesafe/jev-1.13` | $0.037 | $0.18 | ~$36 | — | ~$36 |
| `google/gemini-3.8-flash` | $0.83 | $4.13 | ~$813 | ~$369 | ~$369 |
| PubMed-Screener-420M (GCP Jakarta spot) | compute only | | — | — | **$1.51** |

> **Note.** These estimates cover the benchmark evaluation run only. `openai/gpt-6-luna`, `anthropic/claude-haiku-5.5`, and `google/gemini-3.8-flash` are the representative hosted models suitable for high-volume screening workloads. Hosted LLM costs recur on every rerun and require transmitting review abstracts and screening objectives to external servers; open-weights compute costs scale with hardware time and involve no data egress. The Jakarta deployment achieves Indonesian data residency — review protocols and abstracts never leave Indonesian infrastructure. No direct accuracy comparison between PubMed-Screener-420M and any hosted model was conducted in this study.

## Sovereignty and Cost

Beyond achieving data sovereignty, PubMed-Screener-420M runs the full 1,024,980-evaluation benchmark at **$1.51 on a GCP spot instance in Jakarta** — approximately **24× cheaper than the lowest-cost hosted alternative** (jev-1.13 at ~$36) and **32× cheaper than batch-priced frontier models** (Luna or Haiku 5.5 at ~$49), with no data leaving Indonesian infrastructure.

Two caveats apply. First, spot instances can be preempted mid-run, making them suitable for reproducible benchmark reruns but less reliable for time-critical screening deadlines; an on-demand instance at $400.66/month brings the benchmark cost to $5.62, still 6× cheaper than jev-1.13. Second, the comparison is cost-per-benchmark-run: the proportional gap holds per-inclusion as well, but the absolute per-inclusion cost is in the sub-cent range for all models. The meaningful advantage is the cost of **reproducibility** — every rerun of the full benchmark by any team costs $1.51 on open weights versus $36–$369 on hosted APIs, with no accumulating API dependency.

## The Coffee Test

We ran the entire 1,024,980-evaluation benchmark — every abstract, every threshold decision, every missed recall calculation — for **$1.51 on a Jakarta spot VM.**

That is less than an Americano Double Shot at Kopi Kenangan.

GPT Luna and Claude Haiku? ~$49. Jev? ~$36. Us...? One coffee... no sugar, no data egress, no recurring API bill... and the coffee is optional.
