# Article-level relevance re-annotation — go/no-go memo

Status: **scaffold only, not applied to the paper.** This memo + the two
files below let you decide whether to run the full experiment before the
deadline. Nothing in `practicum_paper.tex` depends on any of it yet.

Companion files:
- `scripts/reannotate_article_level.py` — re-runs all six systems to get
  ranked article IDs and re-scores them against article-level qrels.
- `data/evaluation/gold_standard_test_article.SAMPLE.json` — the target
  schema, with 8 worked examples and 45 stubs to fill.

---

## 1. What the experiment is

Today relevance is judged at the **CELEX instrument** level: a hit is any
article of the right Regulation/Directive appearing in the top 5
(`src/evaluation/evaluator.py`, `retrieved_celex = [r.article.celex_id ...]`).
The re-annotation moves the bar to the **exact article**: relevance becomes a
set of `celex_id::Article N` doc IDs, and a hit requires one of *those* in the
top 5.

This is stricter and it is the paper's own stated limitation
(Conclusion: "lenient toward systems that find the right act but the wrong
provision"; Future Work: "extend relevance judgements ... to the article
level"). Doing it converts that limitation into a result and directly feeds
the methodology framing: *the choice of relevance granularity is a design
knob; here is what changes and what does not.*

## 2. Why it is not "no new inference"

The saved per-query reports (`data/indices/*_baseline_report.json`,
`test_report.json`) store only metrics per query
(`hit_at_k`, `reciprocal_rank`, `ndcg_at_k`) — **not the ranked lists**.
Re-scoring against a finer qrel set needs the actual ranked `doc_id`s, so
every system has to be re-run:

| system | how | approx cost (CPU, M1/Ryzen) |
|---|---|---|
| BM25 | `rank-bm25`, in-memory | seconds |
| nomic dense | FAISS index already built (`data/indices/dense.faiss`) | ~1 min (query encode only) |
| hybrid | RRF over the two above | seconds |
| SPLADE | sparse `.npz` already built | ~1–2 min |
| E5-large-v2 | FAISS index already built | ~1 min |
| BGE-small-v1.5 | FAISS index already built | <1 min |

Corpus embeddings do **not** need recomputing — all five indices exist and
all five models are in the local HF cache. Total re-run: **well under 15
minutes**. Then bootstrap (5,000 resamples) + Wilcoxon + exact McNemar:
another 2–3 minutes. `scripts/reannotate_article_level.py` does all of this;
it is written and runs, it just needs the qrel file.

## 3. The real cost: the annotation itself

Article-level relevance is a human legal-judgment call. AI can *propose*
`(celex_id, article)` sets by reading each currently-relevant instrument's
article texts (`data/corpus/eu_climate_articles.jsonl` has full
`article_text`), but the instrument-level qrels were "produced with
generative-AI assistance and **verified by the authors**" and the same
standard has to hold here or the significance tests rest on unverified
machine labels.

Observed difficulty from the 8 worked examples in the SAMPLE file:

- **Well-anchored (~60% of queries):** one obvious article. e.g. q024 →
  `32019L1161::Article 5` ("Minimum procurement targets"); q026 →
  `32023L1791::Article 3` ("Energy efficiency first principle"). ~1–2 min
  each to confirm.
- **Small-set / judgment (~30%):** 2–3 candidate articles, need a call on
  whether the query wants the definition, the obligation, or both. e.g.
  q006 F-gas reporting → Article 19 alone, or 19 + 20? ~3–5 min each.
- **Hard (~10%):** colloquial queries far from statutory register (q009
  "suppliers wrecking the environment" → which CSDDD article?), multi-
  instrument queries (q011, q014, q023 each map to 2 CELEX). ~5–10 min each.

**Estimate for the 53 test queries:** ~2.5–4 hours of author time to verify
an AI-proposed set to a defensible standard. All 102 (adds the 49-query
validation split): ~5–7 hours.

## 4. Recommendation (given the deadline)

Pick one:

1. **Full, 53 queries** — if ~3–4 hours of verification is available before
   the deadline. Highest payoff: the instrument-vs-article comparison
   becomes a real subsection and the framing is fully earned. Risk: if
   verification slips, you have unverified labels under headline numbers.

2. **EP-derived 31 only** — the Commission's written answers frequently name
   the operative provision, so article mapping is better anchored and
   faster (~1.5–2 hours). Report as "article-level results on the
   externally-sourced subset"; still enough for the methodology point.

3. **Illustrative subset (~15–20)** — the 8 worked examples plus ~10 more,
   reported as a sensitivity probe, not a re-scoring of the paper. ~1 hour.
   Lowest risk, weakest claim.

4. **Defer** — keep it in Future Work (it is already there) and ship the
   reframe + Table II/III work now.

My suggestion: **option 2** if you want the result in this version, **option
4** if verification time is not there. Option 1 only if you are confident of
the 3–4 hours.

## 5. If you run it — exact steps

```bash
# 1. Fill data/evaluation/gold_standard_test_article.json
#    (copy the SAMPLE, replace every "relevant_article_ids": null)
cp data/evaluation/gold_standard_test_article.SAMPLE.json \
   data/evaluation/gold_standard_test_article.json
#    ... edit: for each query, list the operative article doc_ids ...

# 2. Re-run all six systems + re-score + stats
python scripts/reannotate_article_level.py \
    --article-qrels data/evaluation/gold_standard_test_article.json \
    --out data/evaluation/article_level_results.json

# 3. Inspect data/evaluation/article_level_results.json:
#    per-system HR@5/MRR@5/NDCG@5 at article level, deltas vs instrument
#    level, Wilcoxon + McNemar p-values, bootstrap 95% CIs.
```

## 6. What would change in the paper (only after step 3)

- **Section V-A / new subsection:** instrument-level vs article-level table
  for all six systems; discussion of whether "dense ≈ hybrid, both beat
  BM25" survives.
- **Fig. 3:** regenerate under article qrels, or add the compact
  comparison table (page budget is currently zero — the table is the
  safer option; see below).
- **Abstract, Intro preview, Conclusion:** reconcile the headline numbers
  (0.925 / 0.906 / 0.698 etc.) with the article-level values, honestly
  reporting any shift.
- **Section IV-A + Section I aside:** reframe instrument-level as "the
  coarser of two standards we report".
- **Future Work:** drop "extend relevance judgements ... to the article
  level" (done).
- **GenAI disclosure:** add the re-annotation as AI-assisted, author-verified.

## 7. Page-budget warning

The paper is at **exactly 10 pages** (hard cap; unevaluated if exceeded).
The reframe + Table II/III work already consumed all slack and was clawed
back by trimming redundant prose and making Table II single-column. Item 4's
new subsection + comparison table needs a further ~15–20 lines cut
elsewhere. Candidates not yet touched: Fig. 1 (token-length histogram; its
content is fully stated in text), Section V-F, the Section VI "Small samples
mislead" paragraph.

## 8. Known modelling caveats for the annotation

- Five amendment articles are split into sub-chunks that all inherit the
  parent `Article 1` label, so `32023R0839::Article 1` maps to 2 retrieval
  units; a hit on either counts. The scorer handles this (matches on
  `doc_id` prefix set).
- Some instruments are amendment acts that renumber host articles
  (`Article 13a`, `13b`...). Use the doc IDs exactly as they appear in
  `eu_climate_articles.jsonl` — the SAMPLE file lists the available
  article numbers per relevant CELEX to make this mechanical.
