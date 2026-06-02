# Lexico-Semantic Fusion for EU Climate Law Retrieval

A hybrid retrieval-augmented generation (RAG) system for EU climate legislation.
Combines BM25 lexical search with Voyage-law-2 dense retrieval, fused via
Reciprocal Rank Fusion (RRF), and grounded answer generation via Claude.

---

## What This System Does

Given a natural-language legal question, the system:

1. Retrieves relevant legislative articles from 57 EU climate law documents
   using both keyword search (BM25) and semantic search (Voyage-law-2 + FAISS)
2. Fuses both ranked lists into a single ranking using Reciprocal Rank Fusion
3. Generates a strictly grounded answer via Claude, with every claim cited
   to a specific `[CELEX_ID — Article N]`

---

## Dataset

| Property | Value |
|---|---|
| Source | EU CELLAR database (publications.europa.eu) |
| Documents | 57 EU legislative acts |
| Articles | 825 article-level provisions |
| Instruments | ETS, CBAM, Taxonomy, LULUCF, F-Gas, MRV, ESR, European Climate Law |
| Format | JSONL — one line per article |
| Fields | `celex_id`, `doc_type`, `article_number`, `article_text`, `cross_references` |

The ETL pipeline is in `notebooks/cellar_etl.ipynb`.

---

## Project Structure

```
├── src/
│   ├── models/schemas.py        # Pydantic data contracts
│   ├── ingestion/loader.py      # JSONL loader with validation
│   ├── retrieval/
│   │   ├── base.py              # BaseRetriever ABC
│   │   ├── dense.py             # DenseRetriever (Voyage-law-2 + FAISS)
│   │   └── sparse.py            # SparseRetriever (BM25Okapi)
│   ├── fusion/controller.py     # RankFusionController (concurrent + RRF)
│   ├── generation/generator.py  # LegalGenerator (Anthropic, strict grounding)
│   └── evaluation/evaluator.py  # Evaluator (Hit_Rate@k, MRR, baselines)
├── data/
│   ├── corpus/eu_climate_articles.jsonl
│   ├── evaluation/gold_standard.json
│   ├── indices/                 # built by --mode index
│   └── etl/                     # ETL cache (manifest, raw zips)
├── notebooks/
│   └── cellar_etl.ipynb         # EU CELLAR ETL pipeline
├── docs/
│   └── first_principles_audit.docx
├── scripts/
│   └── generate_audit.py
├── tests/
│   ├── test_retrieval.py
│   ├── test_fusion.py
│   └── test_evaluation.py
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

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
# Edit .env:
#   VOYAGE_API_KEY=your_voyage_key
#   ANTHROPIC_API_KEY=your_anthropic_key
```

---

## Reproducing Results

### Step 1 — Build indices (requires VOYAGE_API_KEY)

```bash
python main.py --mode index
# Embeds 825 articles with voyage-law-2 (~3 API calls at batch_size=64)
# Saves: data/indices/dense.faiss + data/indices/sparse.bm25.pkl
```

### Step 2 — Run the ablation study

```bash
python main.py --mode baselines
```

This evaluates three systems side by side on the 12-query gold standard:

```
System               Hit_Rate@5       MRR@5
----------------------------------------------
BM25 (sparse-only)   x.xxxx           x.xxxx
Voyage (dense-only)  x.xxxx           x.xxxx
Hybrid RRF (ours)    x.xxxx           x.xxxx
```

Full per-query results saved to `data/indices/baseline_report.json`.

### Step 3 — Run a query with generation (requires ANTHROPIC_API_KEY)

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
# Expected: 35 passed
```

---

## Evaluation Methodology

**Gold standard:** 12 manually curated queries in `data/evaluation/gold_standard.json`.
Each query maps to one or more relevant CELEX document IDs.

**Metrics:**
- `Hit_Rate@5` — fraction of queries where at least one relevant document
  appears in the top-5 results
- `MRR@5` — mean reciprocal rank of the first relevant result

**Matching:** CELEX-level (document, not article). Corrigenda suffixes stripped
before comparison (e.g. `32003L0087R(02)` matches gold `32003L0087`).

**Known limitations:**
- 12 queries produces wide confidence intervals (~±0.14 for Hit_Rate at 95%).
  Interpret aggregate numbers as indicative, not statistically conclusive.
- 2 of 12 gold queries reference documents not in the corpus
  (`32019R2088` SFDR, `32023R0851` CO2 car standards). These will always
  score MRR=0, creating a lower bound on reported metrics.

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
produces units that align with how lawyers reason about law.

**Why voyage-law-2?**
Legal text has a distributional shift from general text: dense terminology,
Latin phrases, cross-references. Domain fine-tuning on legal corpora
produces meaningfully better embeddings than `text-embedding-3-large`
for legal retrieval tasks.

**Why strict grounding in the system prompt?**
Legal analysis requires traceable claims. A response that cites
`[32003L0087 — Article 12]` can be verified against the statute.
Hallucinated legal conclusions cannot.

---

## Configuration

All hyperparameters are in `config.py` / `.env`. Key settings:

| Parameter | Default | Effect |
|---|---|---|
| `voyage_model` | `voyage-law-2` | Embedding model |
| `voyage_embed_dim` | `1024` | Must match model output |
| `bm25_k1` | `1.5` | Term frequency saturation |
| `bm25_b` | `0.75` | Document length normalization |
| `rrf_k` | `60` | RRF smoothing constant |
| `top_k_retrieval` | `20` | Candidates per retriever before fusion |
| `top_k_fused` | `5` | Final results returned to generator |
| `llm_model` | `claude-opus-4-7` | Generation model |
| `llm_max_tokens` | `2048` | Max generated tokens |

---

## Limitations

1. **Corpus completeness** — 57 documents covers the core EU climate acquis
   but excludes delegated acts, implementing regulations, and amendments.
2. **Gold standard size** — 12 queries is too small for statistically
   robust conclusions. Expanding to 50+ queries is recommended before
   publication.
3. **No temporal filtering** — the system treats all corpus documents as
   equally current; consolidated versions are not tracked.
4. **Single-language** — English text only. EU law is official in 24 languages.
5. **No reranking** — a cross-encoder reranker on top of the top-5 would
   likely improve precision further.
