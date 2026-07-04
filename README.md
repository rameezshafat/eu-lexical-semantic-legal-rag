# Lexico-Semantic Fusion for EU Climate Law Retrieval

A hybrid retrieval-augmented generation (RAG) system for EU climate legislation.
Combines BM25 lexical search with nomic-embed-text-v1.5 dense retrieval, fused via
Reciprocal Rank Fusion (RRF), and grounded answer generation via Llama 3.3 70B (Ollama).

---

## What This System Does

Given a natural-language legal question, the system:

1. Retrieves relevant legislative articles from 72 EU climate law documents
   using both keyword search (BM25) and semantic search (nomic-embed-text-v1.5 + FAISS)
2. Fuses both ranked lists into a single ranking using Reciprocal Rank Fusion
3. Generates a strictly grounded answer via Llama 3.3 70B (Ollama), with every claim cited
   to a specific `[CELEX_ID — Article N]`

---

## Dataset

| Property | Value |
|---|---|
| Source | EU CELLAR database (publications.europa.eu) |
| Documents | 72 EU legislative acts |
| Articles | 1,156 article-level provisions |
| Indexed chunks | 1,166 (5 oversized articles split via hierarchical chunking) |
| Instruments | ETS, CBAM, Taxonomy, LULUCF, F-Gas, MRV, ESR, European Climate Law |
| Format | JSONL — one line per article |
| Fields | `celex_id`, `doc_type`, `article_number`, `article_text`, `cross_references`, `concept_ids` |

The ETL pipeline is in `notebooks/cellar_etl.ipynb`.

---

## Project Structure

```
├── src/
│   ├── models/schemas.py           # Pydantic data contracts
│   ├── ingestion/
│   │   ├── loader.py               # JSONL loader with validation
│   │   └── chunker.py              # Hierarchical chunking for oversized articles
│   ├── retrieval/
│   │   ├── base.py                 # BaseRetriever ABC
│   │   ├── dense.py                # DenseRetriever (nomic-embed-text-v1.5 + FAISS)
│   │   └── sparse.py               # SparseRetriever (BM25Okapi)
│   ├── fusion/controller.py        # RankFusionController (concurrent + RRF)
│   ├── generation/generator.py     # LegalGenerator (Ollama/Llama, strict grounding)
│   └── evaluation/evaluator.py     # Evaluator (HR@k, MRR@k, NDCG@k, HN_Rate@k)
├── data/
│   ├── corpus/eu_climate_articles.jsonl
│   ├── evaluation/gold_standard.json
│   ├── indices/                    # built by --mode index
│   └── etl/                        # ETL cache (manifest, raw zips)
├── notebooks/
│   └── cellar_etl.ipynb            # EU CELLAR ETL pipeline
├── docs/
│   ├── TECHNICAL_DOCUMENTATION.md
│   └── evaluation_upgrade_notes.md
├── scripts/
│   ├── sweep_weights.py            # Weighted-RRF ablation grid search
│   └── breakdown_by_difficulty.py  # Per-difficulty retrieval breakdown
├── tests/
│   ├── test_retrieval.py
│   ├── test_fusion.py
│   ├── test_evaluation.py
│   └── test_chunker.py
├── main.py        # CLI entry point
├── app.py         # Gradio demo UI
├── config.py      # Pydantic BaseSettings
└── requirements.txt
```

---

## Setup

**Python 3.10+ required.**

```bash
git clone <repo>
cd eu-lexical-semantic-legal-rag

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Pull the generation model (one-time, ~40GB)
ollama pull llama3.3:70b
# nomic-embed-text-v1.5 (~280MB) downloads automatically on first run via sentence-transformers
```

---

## Reproducing Results

### Step 1 — Build indices

```bash
python main.py --mode index
# Applies hierarchical chunking (1,156 articles → 1,166 chunks)
# Embeds chunks with nomic-embed-text-v1.5 (downloads model on first run)
# Saves: data/indices/dense.faiss + data/indices/sparse.bm25.pkl
```

### Step 2 — Tune RRF hyperparameters (validation set only)

```bash
python scripts/tune_rrf.py
```

Sweeps `dense_weight ∈ {1,2,3,5,8,12,20}`, `k ∈ {10,20,30,60}`,
`top_k_retrieval ∈ {20,50,100}` against the 49-query **validation set**.
Test set is never touched. Best found: `dense_weight=5.0, k=20, top_k_retrieval=100`.

### Step 3 — Evaluate on held-out test set

```bash
python scripts/eval_test.py
```

**Held-out test results** (53 queries: original 22-query split, seed=42, later
expanded with 31 queries derived from European Parliament written questions —
E-number provenance recorded per query):

```
System                   HR@5      MRR@5     NDCG@5
------------------------------------------------------
BM25 (sparse-only)       0.6981    0.5280    0.4935
nomic-embed (dense-only) 0.9245    0.7425    0.7094
Hybrid RRF (ours)        0.9057    0.7730    0.7208
```

Both dense-only and hybrid significantly outperform BM25 (Wilcoxon signed-rank
on per-query MRR: p=0.0005 and p=0.0036 respectively). Hybrid and dense-only
are statistically indistinguishable (p=0.61, negligible effect size): hybrid is
slightly ahead on MRR/NDCG and slightly behind on HR@5, all within noise.
Fusion parameters were selected on the validation set only. HN_Rate is omitted
here pending completion of hard-negative annotation on the expanded test set
(currently only 4 of 53 test queries are annotated).

### Step 4 — Full-corpus smoke test (all 102 queries)

```bash
python main.py --mode baselines
```

Runs all queries (validation + test combined) as a pipeline smoke test. Full
per-query results saved to `data/indices/baseline_report.json`.

> Do not cite these numbers: the full set includes the 49 validation queries
> the fusion hyperparameters were tuned on, so results are optimistically
> biased. The held-out test table above is the reportable one.

### Step 3 — Run a query with generation (requires Ollama running with llama3.3:70b)

```bash
python main.py --mode query \
  --query "What are the obligations for monitoring GHG emissions from maritime transport?"
```

### Step 4 — Run the demo UI

```bash
pip install gradio==6.15.2
python app.py
# Open http://localhost:7860
```

---

## Running Tests

No API keys required. All tests are fully offline.

```bash
pip install pytest==9.0.3
python -m pytest tests/ -v
# Expected: 71 passed
```

---

## Evaluation Methodology

**Gold standard:** 102 queries in `data/evaluation/gold_standard.json`, split into
49 validation queries (`gold_standard_val.json`, used for hyperparameter tuning)
and 53 held-out test queries (`gold_standard_test.json`). The 31 most recent test
queries are derived from real European Parliament written questions (parliamentary
E-numbers recorded in each query's notes field), paraphrased into natural user
language to avoid query–corpus vocabulary leakage. 72 of 102 queries carry
annotated hard-negative CELEX IDs (documents that share surface vocabulary but
are legally incorrect answers); syncing these annotations to the test split is
in progress.

**Metrics:**
- `HR@5` — fraction of queries where at least one relevant document appears in top-5
- `MRR@5` — mean reciprocal rank of the first relevant result
- `NDCG@5` — normalised discounted cumulative gain; rewards finding *all* relevant
  documents at high ranks (many queries map to 2+ relevant instruments). Each
  relevant document is credited once, at its first occurrence — the retrieved list
  is article-level, so one CELEX document can occupy several top-k slots
- `HN_Rate@5` — mean fraction of top-5 slots occupied by hard-negative documents
  (lower is better; computed only over queries with hard-negative annotations)

**Matching:** CELEX-level (document, not article). Corrigenda suffixes stripped
before comparison (e.g. `32003L0087R(02)` matches gold `32003L0087`). Sub-chunks
produced by hierarchical chunking (e.g. `Article 1 §2`) match on the parent CELEX ID.

---

## Design Decisions

**Why RRF over learned fusion weights?**
RRF requires no training data and is robust to score distribution mismatches
between BM25 (unbounded positive reals) and cosine similarity (−1 to 1).
The k=60 constant is empirically validated (Cormack et al., 2009) and
appropriate for small candidate pools (top-20 per retriever).

**Why article-level chunking?**
EU legislative articles are semantically self-contained — each article
addresses one obligation, definition, or procedure. Article-level chunking
avoids the arbitrary window-size decisions of token-based chunking and
produces units that align with how lawyers reason about law. Chunking strategy
is held constant across BM25, dense, and hybrid to avoid confounding the comparison.

**Why hierarchical chunking for oversized articles?**
Five articles (all `Article 1` of amending instruments) exceed the 8,192-token
context window. Rather than excluding them, we split at EU legal paragraph
boundaries (the `;  (N)` amendment-point separators) and fall back to sentence
boundaries where needed. Each sub-chunk carries the parent CELEX ID, so
evaluation and citation are unaffected.

**Why nomic-embed-text-v1.5?**
nomic-embed-text-v1.5 is a pure bi-encoder trained exclusively for dense
retrieval, with an 8,192-token context window. This gives a clean three-way
comparison — BM25 (pure lexical) vs nomic-embed (pure semantic) vs Hybrid RRF
(both) — without the contamination introduced by multi-functional models that
jointly encode sparse and dense signals. The model is fully reproducible (no API
key), downloaded automatically via sentence-transformers, and requires asymmetric
task prefixes: `"search_query: "` on queries, `"search_document: "` on documents.

**Why strict grounding in the system prompt?**
Legal analysis requires traceable claims. A response that cites
`[32003L0087 — Article 12]` can be verified against the statute.
Hallucinated legal conclusions cannot.

---

## Configuration

All hyperparameters are in `config.py` / `.env`. Key settings:

| Parameter | Default | Effect |
|---|---|---|
| `dense_model` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model |
| `dense_embed_dim` | `768` | Must match model output dimension |
| `chunk_token_limit` | `7000` | Max tokens per chunk (hierarchical splitting) |
| `bm25_k1` | `1.5` | Term frequency saturation |
| `bm25_b` | `0.75` | Document length normalization |
| `rrf_k` | `20` | RRF smoothing constant (val-set tuned) |
| `top_k_retrieval` | `100` | Candidates per retriever before fusion (val-set tuned) |
| `top_k_fused` | `5` | Final results returned to generator |
| `rrf_dense_weight` | `5.0` | Dense retriever weight in RRF (val-set tuned; 5:1 ratio) |
| `llm_model` | `llama3.3:70b` | Generation model (Ollama tag; use `llama3.2:latest` for local dev) |
| `llm_max_tokens` | `2048` | Max generated tokens |

---

## Limitations

1. **Corpus completeness** — 72 documents covers the core EU climate acquis
   but excludes delegated acts, implementing regulations, and some amendments.
2. **Gold standard size** — 71 queries provides reasonable coverage but
   independent validation of a subset is recommended before publication.
3. **No temporal filtering** — the system treats all corpus documents as
   equally current; consolidated versions are not tracked.
4. **Single-language** — English text only. EU law is official in 24 languages.
5. **No reranking** — a cross-encoder reranker on top of the top-5 would
   likely improve precision further.
