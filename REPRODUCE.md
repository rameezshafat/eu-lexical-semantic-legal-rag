# Reproducing "Evaluating Hybrid Sparse-Dense Retrieval for EU Climate Legal Documents"

This walks through regenerating every table and statistic in
`docs/paper/practicum_paper.tex` (the canonical merged paper; the tables
originated in `eu_climate_hybrid_retrieval.tex` and appear in the same
order). For general system usage (the demo UI, generation queries, running the
test suite) see `README.md`; this file is scoped to paper reproduction only.

All commands run from the repository root. Python 3.14, dependencies pinned
in `pyproject.toml`. All experiments in the paper ran on an Apple M1 Pro,
CPU-only; wall-clock times below are from that machine.

## 0. Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # if present; no API keys are required for retrieval
```

Dense models download automatically on first use via `sentence-transformers`
(no `HF_TOKEN` required, but set one to avoid Hugging Face Hub rate limits —
without it, `intfloat/e5-large-v2` and `BAAI/bge-small-en-v1.5` downloads in
this reproduction took several minutes each on unauthenticated requests).

## 1. Build the primary corpus and indices

The corpus itself (`data/corpus/eu_climate_articles.jsonl`, 1,156 raw
articles from 72 CELEX instruments) is extracted via
`notebooks/cellar_etl.ipynb` (SPARQL against the EU CELLAR endpoint, Formex
XML parsing). This is a long-running, network-dependent notebook and is not
re-run as part of routine reproduction; the extracted JSONL is checked in.

```bash
python main.py --mode index
# Applies hierarchical chunking (1,156 articles -> 1,166 chunks; see
#   src/ingestion/chunker.py for the amendment-point splitting logic)
# Embeds chunks with nomic-embed-text-v1.5 (~5-10 min on CPU)
# Saves: data/indices/dense.faiss, data/indices/dense_article_map.pkl,
#        data/indices/sparse.bm25.pkl, data/indices/sparse_article_map.pkl
```

## 2. RRF hyperparameter tuning (validation set only, Section III-D)

```bash
python scripts/tune_rrf.py
```

Grid search over `dense_weight in {1,2,3,5,8,12,20}`, `k in {10,20,30,60}`,
`top_k_retrieval in {20,50,100}` against the full 49-query
`data/evaluation/gold_standard_val.json`. The sealed 53-query test set is
never touched at this stage. Expected best: `dense_weight=5.0, k=20,
top_k_retrieval=100` — already set as the defaults in `config.py`.

## 3. Table I — main effectiveness comparison (Section V-A)

```bash
python scripts/eval_test.py
```

Evaluates BM25, nomic-embed (dense-only), and Hybrid RRF once each on the
sealed 53-query test set (`data/evaluation/gold_standard_test.json`), saving
per-query scores to `data/indices/test_report.json`. Expected output:

```
System                          HR@5   MRR@5   NDCG@5    HN@5
BM25 (sparse-only)             0.6981  0.5280  0.4935  0.2000
nomic-embed (dense-only)       0.9245  0.7425  0.7094  0.2500
Hybrid RRF (ours)              0.9057  0.7730  0.7208  0.2500
```

Then compute bootstrap 95% CIs and Wilcoxon significance:

```bash
python scripts/bootstrap_ci.py
python scripts/significance_test.py
```

`bootstrap_ci.py` writes `data/evaluation/confidence_intervals.json`
(5,000-resample percentile bootstrap; this is what Table I's parenthetical
CIs come from — despite an earlier draft mislabeling it "Wilson", it has
always been a bootstrap CI). `significance_test.py` writes
`data/evaluation/significance_results.json`; expect `p=0.000475` for
Hybrid-vs-BM25 and `p=0.613661` for Hybrid-vs-Dense (rounded to `p=0.0005`
and `p=0.614` in the paper).

## 4. Table I — embedding-model ablation rows (Section V-D)

```bash
python scripts/e5_baseline.py --split test
python scripts/bge_baseline.py --split test
```

Each script builds its own FAISS index (`dense_e5.faiss` /
`dense_bge.faiss`, both checked in) if one does not already exist, then
evaluates on the same sealed test set and prints a comparison table against
the cached nomic/E5 results. Expected:

```
E5-large-v2 (dense)          0.9245  0.8057  0.7945  0.2500
BGE-small-v1.5 (dense)       0.9434  0.7314  0.6962  0.2000
```

Both scripts save per-query results (`data/indices/e5_baseline_report.json`,
`data/indices/bge_baseline_report.json`) so a bootstrap CI can be computed
the same way as Table I's other rows — this is not wired into
`bootstrap_ci.py` (which is scoped to bm25/dense/hybrid) but is a direct
percentile-bootstrap over each report's `per_query` list; see the inline
computation used when the paper's CIs were derived, reproducible as:

```bash
python3 -c "
import json, random
def ci(values, n=5000, seed=42):
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k))/k for _ in range(n))
    return sum(values)/k, means[int(.025*n)], means[int(.975*n)]
for path, key in [('data/indices/e5_baseline_report.json','e5_large_v2'),
                  ('data/indices/bge_baseline_report.json','bge_small_v1_5')]:
    pq = json.load(open(path))[key]['per_query']
    for metric, field in [('HR@5','hit_at_k'),('MRR@5','reciprocal_rank'),('NDCG@5','ndcg_at_k')]:
        print(path, metric, ci([float(q[field]) for q in pq]))
"
```

**Do not stop at the point estimates and CIs above.** An earlier internal
draft of this paper reported E5-large-v2 and BGE-small-v1.5 as "beating" or
"exceeding" nomic-embed-text-v1.5 based on these point estimates alone. Run
the Wilcoxon signed-rank test before writing any comparative claim:

```bash
python3 -c "
import json
from scipy.stats import wilcoxon
nomic = {q['query_id']: q for q in json.load(open('data/indices/test_report.json'))['results']['dense']['per_query']}
e5    = {q['query_id']: q for q in json.load(open('data/indices/e5_baseline_report.json'))['e5_large_v2']['per_query']}
bge   = {q['query_id']: q for q in json.load(open('data/indices/bge_baseline_report.json'))['bge_small_v1_5']['per_query']}
qids = sorted(nomic)
for model_name, model_pq in [('E5-large-v2', e5), ('BGE-small-v1.5', bge)]:
    for metric, field in [('HR@5','hit_at_k'), ('MRR@5','reciprocal_rank'), ('NDCG@5','ndcg_at_k')]:
        a = [float(model_pq[q][field]) for q in qids]
        b = [float(nomic[q][field]) for q in qids]
        w, p = wilcoxon(a, b)
        print(f'{model_name} vs nomic  {metric}: p={p:.4f}  {\"SIGNIFICANT\" if p < 0.05 else \"not significant\"}')
"
```

This should print six comparisons, **none significant** (`p` ranging from
about 0.055 to 1.0 at the time this file was written). The full results are
also saved at `data/evaluation/ablation_significance_results.json`. The
paper reports these as statistical parity with nomic, not superiority — if
you regenerate this file and get different, significant results, the paper
text needs to change accordingly, not just the numbers in the tables.

## 5. Token-length distribution stat (Section V-D, abstract)

The "21.1% of chunks exceed 512 tokens" figure:

```bash
python3 -c "
import pickle
from transformers import AutoTokenizer
chunks = pickle.load(open('data/indices/sparse_article_map.pkl','rb'))
tok = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
lengths = sorted(len(tok.encode(c.article_text)) for c in chunks)
n = len(lengths)
print(f'median={lengths[n//2]} mean={sum(lengths)/n:.1f} max={lengths[-1]}')
print(f'over 512: {sum(1 for l in lengths if l > 512)}/{n}')
"
```

## 6. Table II — per-query hit pattern breakdown (Section V-B)

Derived directly from `data/indices/test_report.json` (produced in step 3),
no separate script:

```bash
python3 -c "
import json
r = json.load(open('data/indices/test_report.json'))
bm25   = {q['query_id']: q['hit_at_k'] for q in r['results']['bm25']['per_query']}
dense  = {q['query_id']: q['hit_at_k'] for q in r['results']['dense']['per_query']}
hybrid = {q['query_id']: q['hit_at_k'] for q in r['results']['hybrid']['per_query']}
from collections import Counter
c = Counter((bm25[q], dense[q], hybrid[q]) for q in bm25)
for k, v in sorted(c.items(), key=lambda x: -x[1]):
    print(k, v)
"
```

## 7. Table III — failure root-cause table (Section V-B)

Not scripted end-to-end (it is a qualitative table backed by targeted
investigation). To re-verify the underlying claims:

- Corpus coverage of the 5 gold CELEX IDs: check membership and article
  count in `data/indices/sparse_article_map.pkl` (see the paper text for the
  exact `celex_id` values: 32024L1760, 32021R1229, 32023L0959 x2,
  32020R0852).
- q080's actual BM25/dense rankings (used to confirm the cross-instrument
  discrimination diagnosis, not the amendment-diff-chunking hypothesis in an
  earlier draft): load `DenseRetriever`/`SparseRetriever` per
  `scripts/eval_test.py`'s setup and call `.retrieve(query, 15)` directly
  with q080's query text.
- q083's content-absence claim: the raw Formex XML for 32023L0959 is not
  checked in (`data/etl/raw_cache/` is gitignored); re-fetch it from the
  manifest URL in `data/etl/manifest_cache.json` with
  `curl -H "Accept: application/zip" <url>` and grep the extracted XML for
  "fishing", "vessel", "inland navigation".

## 8. RRF parameter sensitivity (Section VI-A)

```bash
python scripts/plot_rrf_sweep.py --run-sweep   # validation-set dense-weight sweep
GOLD_STANDARD_PATH=data/evaluation/gold_standard_test.json python scripts/rrf_sweep.py
```

The second command is a post-hoc robustness check on the sealed test set
(84-combination grid); it is not used to alter the reported configuration.

## 8a. Figures (Fig. 1-3)

```bash
pip install -e ".[figures]"
python scripts/make_figures.py
```

Regenerates all three paper figures into `docs/paper/figures/` as vector
PDFs: the token-length histogram (step 5's stat, visualised), the
per-difficulty breakdown bar chart (joins `test_report.json` against
`gold_standard_test.json`'s `difficulty` field — no new evaluation), and
the RRF heatmap (re-runs the `k`x`w_d` grid at top-`k_r`=100 on the sealed
test set and saves both the figure and `data/indices/rrf_heatmap_data.json`).
The heatmap sweep re-embeds the 53 test queries once; expect a few minutes
on CPU.

## 9. IAA (Section VI-C, Threats to Validity)

```bash
python scripts/compute_iaa.py --a1 data/evaluation/iaa_annotator1.json --a2 data/evaluation/iaa_annotator2.json
```

`iaa_annotator1.json` is checked in. `iaa_annotator2.json` does not exist —
this requires a genuine second, independent human annotator following
`docs/annotator_instructions.md`; it is intentionally not simulated (see
Threats to Validity for why). The 58.5% MEP-corroboration figure quoted
alongside it is reproducible directly from the gold standard:

```bash
python3 -c "
import json, re
qs = json.load(open('data/evaluation/gold_standard_test.json'))['queries']
n = sum(1 for q in qs if re.search(r'E-\d{6}/\d{4}|Commission answer', q.get('notes','')))
print(f'{n}/{len(qs)} = {n/len(qs)*100:.1f}%')
"
```

## 10. Known gaps in this reproduction package

- `README.md`'s own "Reproducing Results" section predates several fixes
  made to the paper (it still shows `p=0.0036` for Hybrid-vs-BM25 where the
  current, correct value is `p=0.0005`) and was not updated as part of this
  pass — treat this file, not the README, as authoritative for the paper's
  numbers.
- The E5/BGE ablation bootstrap CIs are computed via the ad hoc script in
  step 4 above rather than a checked-in script; if this ablation is extended
  further, that computation should be promoted into `bootstrap_ci.py`
  properly rather than re-copy-pasted.
- Table III is not push-button reproducible end-to-end; the investigative
  commands in step 7 reproduce the underlying evidence but were run
  interactively, not via a saved script.
