# Technical Documentation — EU Lexical-Semantic Legal RAG

**Authors:** Rameez Wani, Ananya Warior — School of Computing, Dublin City University
**Branch:** `feature/ananya` | **Last updated:** 2026-06-26

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Question](#2-research-question)
3. [Repository Layout](#3-repository-layout)
4. [System Architecture](#4-system-architecture)
5. [Data Pipeline — EU CELLAR ETL](#5-data-pipeline--eu-cellar-etl)
6. [Corpus Format](#6-corpus-format)
7. [Configuration System](#7-configuration-system)
8. [Data Schemas (src/models/schemas.py)](#8-data-schemas)
9. [Ingestion Layer (src/ingestion/loader.py)](#9-ingestion-layer)
10. [Retrieval Layer](#10-retrieval-layer)
    - 10.1 [BaseRetriever ABC](#101-baseretriever-abc)
    - 10.2 [SparseRetriever — BM25](#102-sparseretriever--bm25)
    - 10.3 [DenseRetriever — bge-large-en-v1.5 + FAISS](#103-denseretriever--bge-large-en-v15--faiss)
11. [Rank Fusion Controller (src/fusion/controller.py)](#11-rank-fusion-controller)
12. [Generation Layer (src/generation/generator.py)](#12-generation-layer)
13. [Evaluation Layer (src/evaluation/evaluator.py)](#13-evaluation-layer)
    - 13.1 [Gold Standard Benchmark](#131-gold-standard-benchmark)
    - 13.2 [Metrics](#132-metrics)
    - 13.3 [Hard Negatives](#133-hard-negatives)
    - 13.4 [Baseline Ablation](#134-baseline-ablation)
14. [CLI Entry Point (main.py)](#14-cli-entry-point)
15. [Demo UI (app.py)](#15-demo-ui)
16. [Analysis Scripts](#16-analysis-scripts)
17. [Test Suite](#17-test-suite)
18. [Setup and First Run](#18-setup-and-first-run)
19. [End-to-End Workflow](#19-end-to-end-workflow)
20. [Key Design Decisions and Trade-offs](#20-key-design-decisions-and-trade-offs)
21. [Known Limitations and Open Issues](#21-known-limitations-and-open-issues)
22. [Glossary](#22-glossary)

---

## 1. Project Overview

This is a Retrieval-Augmented Generation (RAG) system for EU climate legislation. Given a plain-English legal question, the system:

1. Retrieves the most relevant legislative article provisions from a corpus of EU climate law instruments using two independent retrieval methods.
2. Merges both ranked result lists using Reciprocal Rank Fusion (RRF).
3. Sends the fused context to a local LLM (Llama 3.3 70B via Ollama) to produce a strictly-grounded, citation-backed answer.

The project is simultaneously a working tool and a research experiment. The core investigation is whether combining sparse (lexical) and dense (semantic) retrieval improves grounding accuracy for legal RAG systems compared to either method used alone.

---

## 2. Research Question

> Does combining sparse and dense retrieval via Reciprocal Rank Fusion improve retrieval accuracy for EU climate law queries compared to either BM25 or pure bi-encoder retrieval used alone?

**Why this is non-trivial in the legal domain:**

- Legal text is both technically precise (exact article numbers, instrument names, defined terms like "CELEX ID" or "ETS allowance" matter) and semantically complex (a query about "carbon price relief for poor households" must map to the Social Climate Fund Regulation, which never uses those words verbatim).
- BM25 (sparse) excels at exact-match queries but fails on paraphrase and synonym variation.
- Dense bi-encoder retrieval (semantic) handles paraphrase well but can conflate semantically similar but legally distinct concepts (e.g., "aviation allowances" vs "maritime allowances").
- The research hypothesis is that combining both signals via RRF produces a system that is robust to both failure modes.

**Why the dense model matters:**
The system originally used BAAI/bge-m3 as the "dense" retriever. This was incorrect: bge-m3 is a multi-functional model trained jointly for dense, sparse (SPLADE-style), and ColBERT retrieval. Using it as the "dense baseline" contaminated the comparison because its dense vectors already encode lexical signals. The system now uses `BAAI/bge-large-en-v1.5`, a pure bi-encoder, enabling a clean three-way comparison:

| System | Mechanism |
|---|---|
| BM25 (sparse-only) | Pure lexical — term frequency and inverse document frequency |
| bge-large (dense-only) | Pure semantic — cosine similarity in embedding space |
| RRF Hybrid | Both signals fused — the proposed system |

---

## 3. Repository Layout

```
eu-lexical-semantic-legal-rag/
│
├── main.py                          # CLI entry point (index / query / evaluate / baselines)
├── app.py                           # Gradio demo web UI
├── config.py                        # Central configuration (Pydantic BaseSettings)
├── requirements.txt                 # Python dependencies
│
├── src/                             # All application source code
│   ├── models/
│   │   └── schemas.py               # ALL Pydantic data contracts
│   ├── ingestion/
│   │   └── loader.py                # JSONL corpus loader + validator
│   ├── retrieval/
│   │   ├── base.py                  # BaseRetriever abstract class
│   │   ├── sparse.py                # BM25Okapi lexical retriever
│   │   └── dense.py                 # bge-large-en-v1.5 + FAISS dense retriever
│   ├── fusion/
│   │   └── controller.py            # RankFusionController — concurrent RRF
│   ├── generation/
│   │   └── generator.py             # LegalGenerator — Ollama/Llama, strict grounding
│   └── evaluation/
│       └── evaluator.py             # Evaluator — HR@k, MRR, NDCG, HN_Rate, baselines
│
├── data/
│   ├── corpus/
│   │   └── eu_climate_articles.jsonl  # Indexed corpus (1,156 article provisions)
│   ├── evaluation/
│   │   └── gold_standard.json         # 71-query benchmark with hard negatives
│   ├── indices/                        # Built at runtime by --mode index
│   │   ├── dense.faiss                 # FAISS vector index
│   │   ├── dense_article_map.pkl       # Maps FAISS row → LegalArticle
│   │   ├── sparse.bm25.pkl             # BM25 index object
│   │   ├── sparse_article_map.pkl      # Maps BM25 row → LegalArticle
│   │   ├── evaluation_report.json      # Last --mode evaluate output
│   │   └── baseline_report.json        # Last --mode baselines output
│   └── etl/
│       ├── manifest_cache.json         # Cached SPARQL → CELEX manifest
│       └── raw_cache/                  # Raw Formex XML zip files per instrument
│
├── notebooks/
│   └── cellar_etl.ipynb              # EU CELLAR ETL pipeline (data extraction)
│
├── scripts/
│   ├── sweep_weights.py             # RRF weight ablation grid search
│   └── breakdown_by_difficulty.py   # Per-difficulty metric breakdown
│
├── tests/
│   ├── test_retrieval.py            # Tokenizer + BM25 + dense (mocked) tests
│   ├── test_fusion.py               # RRF math + deduplication + provenance tests
│   └── test_evaluation.py           # HR, MRR, NDCG, HN_Rate tests
│
└── docs/
    ├── TECHNICAL_DOCUMENTATION.md   # This file
    ├── evaluation_upgrade_notes.md  # Hard negative + metric upgrade change log
    ├── project_review_notes.md      # Supervisor review notes
    └── paper/                       # JURIX 2025 paper draft
```

---

## 4. System Architecture

The system is a standard RAG pipeline with a parallel dual-retriever front end:

```
User Query
    │
    ▼
┌──────────────────────────────────────────────┐
│          RankFusionController                 │
│                                              │
│   ┌─────────────────┐  ┌──────────────────┐ │
│   │  SparseRetriever│  │  DenseRetriever  │ │
│   │  (BM25Okapi)    │  │  (bge-large-en + │ │
│   │                 │  │   FAISS IndexIP) │ │
│   └────────┬────────┘  └────────┬─────────┘ │
│            │   ThreadPoolExecutor│           │
│            └──────────┬──────────┘           │
│                       │                      │
│         Reciprocal Rank Fusion               │
│         RRF(d) = Σ w_r / (k + rank_r(d))    │
│                       │                      │
│              top_k_fused results             │
└──────────────────────────────────────────────┘
                    │
                    ▼
         ┌─────────────────┐
         │  LegalGenerator │
         │  (Llama 3.3 70B │
         │   via Ollama)   │
         └────────┬────────┘
                  │
                  ▼
          Grounded Answer
          with [CELEX — Article N] citations
```

**Key architectural properties:**

- **Decoupled layers:** The retrieval layer and generation layer are entirely independent. The evaluator measures retrieval quality without ever calling the LLM. This means retrieval can be benchmarked without Ollama running.
- **Dependency injection:** The fusion controller receives both retrievers as constructor arguments. Swapping in a new retriever (e.g., ColBERT, SPLADE) requires zero changes to the fusion or evaluation code.
- **Concurrent retrieval:** Both retrievers execute in parallel via `ThreadPoolExecutor`, reducing wall-clock latency.
- **Protocol-based evaluation:** The evaluator depends on a `Fusable` protocol (any object with a `fuse_results()` method), not on the concrete `RankFusionController`. This allows single-retriever wrappers to be scored through the same evaluation path.

---

## 5. Data Pipeline — EU CELLAR ETL

The corpus was extracted from the EU's CELLAR semantic repository (`publications.europa.eu`) using a two-phase pipeline implemented in `notebooks/cellar_etl.ipynb`.

### Phase 1 — SPARQL query for work URIs

The EUR-Lex CELLAR exposes a SPARQL endpoint. The notebook queries it with:

```sparql
SELECT ?work ?celex WHERE {
  ?work cdm:work_has_celex_number ?celex .
  FILTER(STRSTARTS(?celex, "3"))   -- sector 3 = EU secondary law
}
```

This returns a list of `(work_URI, CELEX_ID)` pairs for all sector-3 instruments (regulations, directives, decisions). A pre-defined manifest of target CELEX IDs (the EU climate acquis — ETS, CBAM, Taxonomy, LULUCF, etc.) is then used to filter to the relevant instruments.

### Phase 2 — Formex XML extraction

For each target CELEX ID, the pipeline:

1. Fetches the Formex 4 XML manifestation of the instrument (a structured XML format used by the EUR-Lex publication office).
2. Parses the `<ARTICLE>` elements, extracting `article_number` and `article_text`.
3. Extracts `cross_references` (other CELEX IDs cited in the article text).
4. Attaches EuroVoc `concept_ids` from the CELLAR metadata.
5. Writes each article as a JSON object to `data/corpus/eu_climate_articles.jsonl`.

### CELEX ID format

CELEX IDs encode the instrument type and year:

```
3  2003  L  0087
│   │    │    │
│   │    │    └── Sequential number within year
│   │    └─────── Type: L=Directive, R=Regulation, D=Decision
│   └──────────── Year of enactment
└──────────────── Sector: 3 = EU secondary legislation
```

Corrigenda (corrections) are identified by a suffix: `32003L0087R(02)` is the second corrigendum to directive 2003/87. The system strips these suffixes for evaluation matching because the underlying instrument is the same.

### ETL cache

`data/etl/manifest_cache.json` caches the SPARQL results to avoid re-querying CELLAR on reruns. `data/etl/raw_cache/` stores the raw downloaded zip files. If either cache exists, the notebook skips the corresponding download step.

---

## 6. Corpus Format

`data/corpus/eu_climate_articles.jsonl` — one JSON object per line, one object per article provision.

```json
{
  "celex_id": "32003L0087",
  "doc_type": "Directive",
  "article_number": "Article 12",
  "article_text": "Member States shall ensure that, by 30 April each year, the operator of each installation surrenders a number of allowances equal to the total emissions...",
  "cross_references": ["32003L0087", "32004D0156"],
  "concept_ids": ["1115", "2897"]
}
```

**Field definitions:**

| Field | Type | Description |
|---|---|---|
| `celex_id` | `str` | EUR-Lex CELEX identifier. Must start with "3" (sector 3). |
| `doc_type` | `"Directive"` \| `"Regulation"` | Legal instrument type. |
| `article_number` | `str` | Article label as it appears in the instrument (e.g., "Article 3", "Article 3a"). |
| `article_text` | `str` | Full verbatim text of the article. Minimum 10 characters. Placeholder articles ("Does not concern...") are rejected. |
| `cross_references` | `list[str]` | CELEX IDs explicitly cited within this article. May be empty. |
| `concept_ids` | `list[str]` | EuroVoc concept identifiers. May be empty. |

**Corpus statistics:** 1,156 articles from 72 EU legislative instruments.

---

## 7. Configuration System

**File:** `config.py`

All tunable parameters are centralised in a single `Settings` class using Pydantic BaseSettings. This means every parameter can be overridden at runtime via environment variables or a `.env` file without touching source code.

```python
from config import settings
settings.dense_model        # "BAAI/bge-large-en-v1.5"
settings.rrf_k              # 60
settings.top_k_fused        # 5
```

**All parameters with defaults:**

| Parameter | Default | Description |
|---|---|---|
| `dense_model` | `"BAAI/bge-large-en-v1.5"` | Sentence-transformers model name for dense retrieval. **Do not change to bge-m3** — see §10.3. |
| `dense_embed_dim` | `1024` | Embedding dimension. Must match the model's output. bge-large-en-v1.5 outputs 1024-dim. |
| `dense_batch_size` | `64` | Number of documents embedded per batch during indexing. |
| `bm25_k1` | `1.5` | BM25 term-frequency saturation. Higher = more influence of high-frequency terms. |
| `bm25_b` | `0.75` | BM25 document-length normalisation. 0 = no normalisation, 1 = full normalisation. |
| `rrf_k` | `60` | RRF smoothing constant. Controls how much rank-1 documents dominate. 60 is the standard validated default. |
| `top_k_retrieval` | `20` | Candidate documents fetched from each retriever before fusion. |
| `top_k_fused` | `5` | Final results returned after fusion (used as evaluation cut-off). |
| `rrf_dense_weight` | `1.0` | Multiplier on the dense retriever's RRF contribution. Equal weights = standard RRF. |
| `rrf_sparse_weight` | `1.0` | Multiplier on the sparse retriever's RRF contribution. |
| `llm_model` | `"llama3.3:70b"` | Ollama model tag for generation. |
| `llm_max_tokens` | `2048` | Maximum tokens in the generated response. |
| `llm_ollama_base_url` | `"http://localhost:11434/v1"` | Ollama API base URL. |
| `corpus_path` | `"data/corpus/eu_climate_articles.jsonl"` | Path to corpus JSONL. |
| `index_dir` | `"data/indices"` | Directory where built indices are saved and loaded from. |
| `gold_standard_path` | `"data/evaluation/gold_standard.json"` | Path to evaluation benchmark. |

**Overriding with environment variables:**

```bash
# Override the RRF k constant for an experiment
DENSE_MODEL="BAAI/bge-large-en-v1.5" RRF_K=30 python main.py --mode evaluate

# Or via .env file (copy from .env.example)
cp .env.example .env
echo "RRF_K=30" >> .env
```

**Derived path properties:**

`settings.faiss_index_path` → `Path("data/indices/dense.faiss")`
`settings.bm25_index_path` → `Path("data/indices/sparse.bm25.pkl")`

---

## 8. Data Schemas

**File:** `src/models/schemas.py`

All data contracts are Pydantic v2 models. Every object that crosses a module boundary is typed and validated here. This is the single source of truth for data shapes — if you need to understand what a function returns or accepts, look here first.

### `LegalArticle`

One article provision from the corpus.

```python
class LegalArticle(BaseModel):
    celex_id: str           # Must start with "3"
    doc_type: Literal["Directive", "Regulation"]
    article_number: str
    article_text: str       # min_length=10, strips placeholder text
    cross_references: list[str] = []
    concept_ids: list[str] = []

    @property
    def doc_id(self) -> str:  # "32003L0087::Article 12"
    @property
    def citation_label(self) -> str:  # "[32003L0087 — Article 12]"
```

### `RetrievedResult`

Output of a single retriever (before fusion).

```python
class RetrievedResult(BaseModel):
    article: LegalArticle
    score: float            # Raw score: cosine similarity or BM25 score
    rank: int               # 1-based rank within this retriever's list
    retriever_name: Literal["dense", "sparse"]
```

### `FusedResult`

Output after RRF fusion. Carries provenance from both retrievers.

```python
class FusedResult(BaseModel):
    article: LegalArticle
    rrf_score: float        # Aggregate RRF score
    rank: int               # Final rank in the fused list
    dense_rank: int | None  # Rank from the dense retriever (None if not retrieved)
    sparse_rank: int | None # Rank from the sparse retriever (None if not retrieved)

    @property
    def provenance_str(self) -> str:  # "dense=#3, sparse=#1"
```

### `GenerationOutput`

Full output from the LLM generation layer.

```python
class GenerationOutput(BaseModel):
    query: str
    answer: str
    cited_provisions: list[str]   # Citation labels actually used in the answer
    fused_results: list[FusedResult]
    model_used: str
    input_tokens: int
    output_tokens: int
```

### `GoldQuery`

One entry in the evaluation benchmark.

```python
class GoldQuery(BaseModel):
    query_id: str
    query: str
    relevant_celex_ids: list[str]           # min_length=1
    hard_negative_celex_ids: list[str] = [] # Documents that look relevant but are legally wrong
    difficulty: str = "standard"            # "standard" or "hard"
    notes: str = ""
```

### `EvaluatedQuery`

Per-query evaluation result.

```python
class EvaluatedQuery(BaseModel):
    query_id: str
    query: str
    relevant_celex_ids: list[str]
    retrieved_celex_ids: list[str]     # Ordered list from fused ranking
    hit_at_k: bool                     # Any relevant doc in top-k?
    reciprocal_rank: float             # 1/rank_of_first_relevant, 0 if none
    ndcg_at_k: float = 0.0            # NDCG at cut-off k
    hard_negatives_in_top_k: int = 0  # Count of hard-negative docs in top-k
    top_k: int
```

### `EvaluationReport`

Aggregate metrics over the full benchmark.

```python
class EvaluationReport(BaseModel):
    total_queries: int
    top_k: int
    hit_rate: float             # Mean hit_at_k
    mrr: float                  # Mean reciprocal rank
    ndcg: float = 0.0          # Mean NDCG@k
    hard_negative_rate: float = 0.0  # Mean HN_Rate over annotated queries only
    per_query: list[EvaluatedQuery]
```

---

## 9. Ingestion Layer

**File:** `src/ingestion/loader.py`

`CorpusLoader` reads the JSONL corpus, validates every row with Pydantic, and returns a clean `list[LegalArticle]`. Rows that fail validation are skipped and logged — they do not raise exceptions. This handles:

- Empty lines
- Malformed JSON
- Placeholder articles (`"(Does not concern..."` text)
- Non-object JSON values
- Articles with `celex_id` not starting with "3"

```python
loader = CorpusLoader("data/corpus/eu_climate_articles.jsonl")
articles = loader.load()   # Returns list[LegalArticle], skips invalid rows
```

The loader reads line-by-line (not `json.load()` on the whole file) so it scales to large corpora without loading everything into memory at once.

---

## 10. Retrieval Layer

### 10.1 BaseRetriever ABC

**File:** `src/retrieval/base.py`

All retrievers inherit from `BaseRetriever`. The interface has six abstract members:

```python
class BaseRetriever(ABC):
    def index(self, articles: list[LegalArticle]) -> None: ...
    def retrieve(self, query: str, top_k: int) -> list[RetrievedResult]: ...
    def save(self, directory: str) -> None: ...
    def load(self, directory: str) -> None: ...

    @property
    def name(self) -> str: ...       # "dense" or "sparse"
    @property
    def is_indexed(self) -> bool: ...
```

**Contract guarantees that every implementation must uphold:**
- `retrieve()` returns results in descending relevance order
- `rank` in each `RetrievedResult` is 1-based and contiguous
- `retrieve()` raises `RuntimeError` if called before `index()` or `load()`
- `save()` raises `RuntimeError` if called before `index()` or `load()`

The shared utility `_resolve_dir(directory)` resolves the path and creates the directory if it does not exist.

---

### 10.2 SparseRetriever — BM25

**File:** `src/retrieval/sparse.py`

**Algorithm:** BM25Okapi from the `rank-bm25` library.

**BM25 formula:**

```
BM25(q, d) = Σ_{t ∈ q}  IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))
```

Where:
- `IDF(t)` = inverse document frequency of term `t`
- `tf(t,d)` = term frequency in document `d`
- `k1 = 1.5` — controls term-frequency saturation (diminishing returns for repeated terms)
- `b = 0.75` — document-length normalisation (penalises very long documents)
- `|d|` = document length in tokens
- `avgdl` = average document length across corpus

**Tokenizer:**

The tokenizer is custom-built for legal text. It does NOT use a standard NLP tokenizer because:

- Legal terms must match exactly — stemming ("allowances" → "allow") would cause `32003L0087` (ETS allowances) to match unrelated documents.
- Hyphenated legal terms like "net-zero", "low-carbon", "non-ETS" must be preserved as single tokens.

```python
_TOKEN_RE = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?")

def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
```

This regex matches sequences of alphanumeric characters and internal hyphens. It lowercases everything and strips all punctuation except internal hyphens.

**Index persistence:**

Two pickle files are written:
- `data/indices/sparse.bm25.pkl` — the `BM25Okapi` object (contains the inverted index, IDF weights, document lengths)
- `data/indices/sparse_article_map.pkl` — list of `LegalArticle` objects in the same order as the BM25 corpus

The article map is needed because BM25 operates on position indices, not on article objects directly.

---

### 10.3 DenseRetriever — bge-large-en-v1.5 + FAISS

**File:** `src/retrieval/dense.py`

**Embedding model:** `BAAI/bge-large-en-v1.5` — a pure bi-encoder trained exclusively for dense retrieval. 1024-dimensional output vectors. Downloads automatically from HuggingFace Hub on first use (~1.3GB).

**Why NOT bge-m3:**
`BAAI/bge-m3` is "Multi-Functional" — it was trained jointly for dense retrieval, sparse (SPLADE-style) retrieval, and ColBERT multi-vector retrieval. Using it as the "dense" baseline contaminated the research because its dense vectors already encode lexical signals from the multi-task training. `bge-large-en-v1.5` is a pure bi-encoder — it produces only dense vectors and was trained only for dense retrieval. This gives a clean experimental comparison.

**Asymmetric encoding (query prefix):**

BGE bi-encoders use asymmetric encoding — queries and documents are encoded with different instructions to improve retrieval quality:

```python
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# At query time:
q_vec = model.encode([_QUERY_PREFIX + query], normalize_embeddings=True)

# At index time (documents encoded WITHOUT the prefix):
matrix = model.encode(texts, normalize_embeddings=True)
```

This is the correct usage per the bge-large-en-v1.5 documentation. Omitting the query prefix causes measurable quality degradation.

**FAISS index type:**

`faiss.IndexFlatIP` — exact inner-product search. Because all vectors are L2-normalised at encoding time, inner product equals cosine similarity. Flat index means no approximation — every query computes exact nearest neighbours over all 1,156 vectors, which is fast at this corpus size (sub-millisecond).

**Embedding dimension validation:**

The retriever validates that the model's actual output dimension matches `embed_dim` and raises a descriptive error if not. This catches the common mistake of changing the model without updating the dimension setting.

**Index persistence:**

Two files are written:
- `data/indices/dense.faiss` — the FAISS index (binary format via `faiss.write_index`)
- `data/indices/dense_article_map.pkl` — list of `LegalArticle` objects in the same order as FAISS row IDs

**Important:** After changing the dense model, the FAISS index MUST be rebuilt. The index stores model-specific vector representations — a bge-m3 index cannot be used with bge-large-en-v1.5. Run `python main.py --mode index` to rebuild.

---

## 11. Rank Fusion Controller

**File:** `src/fusion/controller.py`

`RankFusionController` orchestrates concurrent retrieval from both retrievers and merges their results using Weighted Reciprocal Rank Fusion.

**RRF formula:**

```
RRF(d) = Σ_r  w_r / (k + rank_r(d))
```

Where:
- `d` = a document
- `r` = retriever (dense or sparse)
- `rank_r(d)` = rank of document `d` in retriever `r`'s list (1-based)
- `k = 60` = smoothing constant (standard validated default from Cormack et al., 2009)
- `w_r` = per-retriever weight (1.0 by default)

**What the k=60 constant does:**

With k=60, the score contribution of rank-1 is `1/61 ≈ 0.0164` and rank-20 is `1/80 = 0.0125`. The difference between rank-1 and rank-20 is only 0.0039. This means that a document appearing in both lists at moderate ranks (e.g., rank 10 in both) will consistently outscore a document that ranks 1 in only one list. The formula rewards breadth of coverage across retrievers, not dominance in a single list.

**Concurrent execution:**

Both retrievers are executed in parallel using `ThreadPoolExecutor(max_workers=2)`. Results are collected via `as_completed()`. If one retriever fails, fusion continues with the surviving retriever's results (with a warning log). If both fail, a `RuntimeError` is raised.

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(self._dense.retrieve, query, self._top_k_r): "dense",
        executor.submit(self._sparse.retrieve, query, self._top_k_r): "sparse",
    }
```

**Deduplication:**

Documents are keyed by `doc_id` (`celex_id::article_number`). If the same article appears in both lists, their RRF scores are accumulated (not averaged) and a single `FusedResult` is produced. The `dense_rank` and `sparse_rank` fields in `FusedResult` preserve provenance.

**Weighted RRF:**

`rrf_dense_weight` and `rrf_sparse_weight` in `config.py` scale each retriever's contribution. Standard RRF uses equal weights (1.0 / 1.0). The `scripts/sweep_weights.py` script evaluates several ratios to find the optimal weighting for this corpus.

---

## 12. Generation Layer

**File:** `src/generation/generator.py`

`LegalGenerator` sends the fused provisions to Llama 3.3 70B running locally via Ollama.

**Ollama / OpenAI SDK compatibility:**

Ollama exposes an OpenAI-compatible REST API at `http://localhost:11434/v1`. The generator uses the `openai` Python SDK with a custom `base_url` and `api_key="ollama"` (the API key is ignored by Ollama but required by the SDK).

**System prompt design:**

The system prompt enforces strict grounding via six rules:
1. Only use information from the provided CONTEXT block
2. Every legal claim must cite `[CELEX_ID — Article N]`
3. Citations must appear immediately after the cited text
4. If context is insufficient, respond with `"INSUFFICIENT CONTEXT: ..."`
5. No speculation beyond what the text explicitly states
6. Direct answer first, supporting provisions second

This design prioritises traceability over fluency. The goal is a system whose outputs can be verified against the statute, not a conversational legal assistant.

**Context block format:**

The fused provisions are formatted as a numbered block for the LLM:

```
CONTEXT (fused legal provisions, ordered by relevance):
======================================================================

[1] [32003L0087 — Article 12]  (type: Directive, RRF rank: 1, dense=#2, sparse=#1)
------------------------------------------------------------
Member States shall ensure that, by 30 April each year, the operator of each
installation surrenders a number of allowances equal to the total emissions...

[2] [32023L0959 — Article 3]  (type: Directive, RRF rank: 2, dense=#1, sparse=#5)
------------------------------------------------------------
...
======================================================================
```

**Citation extraction:**

After generation, the system checks which `citation_label` strings (`[CELEX_ID — Article N]`) actually appear in the answer text. This produces the `cited_provisions` field — a list of provisions the LLM actually used, not just all provisions it was given.

**Truncation handling:**

If the response is cut off at `max_tokens`, a `[NOTE: This response was truncated...]` message is appended. This prevents silent truncation from producing incomplete legal analysis.

**Graceful empty-context handling:**

If `fused_results` is empty (e.g., retrievers returned nothing), the generator returns an `INSUFFICIENT CONTEXT` response without calling the LLM at all.

---

## 13. Evaluation Layer

**File:** `src/evaluation/evaluator.py`

The evaluator is deliberately decoupled from the generation layer — it only depends on a `Fusable` protocol:

```python
@runtime_checkable
class Fusable(Protocol):
    def fuse_results(self, query: str) -> list[FusedResult]: ...
```

Any object implementing `fuse_results()` can be evaluated — including the `_SingleRetrieverWrapper` adapter that wraps a single `BaseRetriever` for baseline evaluation.

### 13.1 Gold Standard Benchmark

**File:** `data/evaluation/gold_standard.json`

71 manually constructed queries. Each query:
- Is written in natural language, avoiding the exact vocabulary of the relevant instrument
- Maps to one or more relevant CELEX IDs (the correct legal instruments)
- Is tagged with a difficulty level (`standard` or `hard`)
- Optionally carries `hard_negative_celex_ids` (annotated for 20 queries)
- Has a `notes` field explaining the legal mapping reasoning

**Example entry:**

```json
{
  "query_id": "q005",
  "query": "When goods come into the EU, how do they figure out how much carbon went into making them?",
  "relevant_celex_ids": ["32023R0956"],
  "hard_negative_celex_ids": ["32003L0087"],
  "difficulty": "hard",
  "notes": "CBAM Regulation 2023/956. 'carbon that went into making them'=embedded emissions; CBAM unnamed."
}
```

**Difficulty levels:**

- `standard` — The query uses vocabulary that appears in the instrument text. BM25 can find it with reasonable confidence.
- `hard` — The query uses synonyms, analogies, or colloquialisms. BM25 will likely fail; semantic matching is required.

### 13.2 Metrics

**Hit Rate@k (`hit_rate`)**

Binary per-query: did any relevant CELEX ID appear in the top-k results?

```
HR@k = (number of queries with a hit in top-k) / (total queries)
```

Simple, interpretable, but insensitive to rank. A hit at rank 5 scores identically to a hit at rank 1.

**Mean Reciprocal Rank (`mrr`)**

Rewards finding the relevant document at a high rank.

```
MRR = (1/|Q|) × Σ_{q ∈ Q} 1 / rank_first_relevant(q)
```

A document at rank 1 contributes 1.0; rank 2 contributes 0.5; rank 5 contributes 0.2; no hit contributes 0.0.

Limitation: MRR only rewards the FIRST relevant document. For queries with multiple relevant instruments (28 of 71 queries), MRR ignores whether the second instrument was retrieved.

**NDCG@k (`ndcg`)**

Normalised Discounted Cumulative Gain — rewards finding ALL relevant documents at high ranks.

```
DCG@k  = Σ_{i=1}^{k}  rel_i / log₂(i + 1)
IDCG@k = DCG achieved if all relevant docs ranked first (ideal ordering)
NDCG@k = DCG@k / IDCG@k
```

Where `rel_i = 1` if the document at rank `i` is relevant, else `0`. NDCG@k = 1.0 only if all relevant documents are at the very top of the ranking.

NDCG is the most informative metric for this benchmark because many queries have multiple relevant instruments. It is also the most sensitive to hard-negative pollution — four hard negatives occupying ranks 1-4 with the correct document at rank 5 produces NDCG ≈ 0.17, vs HR@5 = 1.0 (which misleadingly treats this as a full success).

**HN Rate@k (`hard_negative_rate`)**

The fraction of top-k result slots occupied by annotated hard-negative documents.

```
HN_Rate@k = (count of hard-negative docs in top-k) / k
```

Computed only over the 20 queries that have `hard_negative_celex_ids` annotated. Lower is better. This is the direct diagnostic for whether the system is being fooled by surface-level keyword matching.

**CELEX-level matching:**

All metrics are computed at the CELEX-ID level (instrument level), not article level. Any article from a relevant instrument counts as a hit. This matches the legal research workflow where the task is to find the right law, not the right paragraph.

Corrigendum suffixes are stripped before matching: `32003L0087R(02)` matches gold entry `32003L0087`.

### 13.3 Hard Negatives

20 queries have annotated hard negatives — documents that share surface vocabulary with the query but are legally incorrect answers. Two categories:

**Cross-sector / cross-instrument** — same vocabulary, different legal mechanism:

| Example | Query intent | Relevant (correct) | Hard negative | Why it fools |
|---|---|---|---|---|
| q001 | Maritime CO2 reporting | `32015R0757` | `32008L0101` | Aviation ETS — same CO2/monitoring vocabulary |
| q005 | CBAM embedded carbon | `32023R0956` | `32003L0087` | ETS — same "carbon price" but domestic not border |
| q015 | Taxonomy technical criteria | `32021R2139` | `32021R2178` | Taxonomy disclosure — same vocabulary, wrong focus |
| q023 | Shipping carbon obligations | `32023L0959` | `32008L0101` | Aviation ETS — same "allowances"/"operator"/"surrender" |

**Temporal / obsolete versions** — superseded instruments:

| Example | Query intent | Relevant (current) | Hard negative (obsolete) |
|---|---|---|---|
| q022 | How aviation entered carbon market (2008) | `32008L0101` | `32023L0959` — wrong era |
| q025 | Current energy savings obligation | `32023L1791` | `32018L2002` — superseded EED amendment |
| q026 | Energy efficiency first principle | `32023L1791` | `32018L2002` — obsolete formulation |

### 13.4 Baseline Ablation

`run_baselines()` evaluates three conditions in one pass:

1. **sparse_only** — BM25 results wrapped in `_SingleRetrieverWrapper` and fed through the evaluator
2. **dense_only** — Same with the dense retriever
3. **hybrid** — Full `RankFusionController` with both retrievers

`_SingleRetrieverWrapper` adapts a `BaseRetriever` into the `Fusable` protocol by wrapping `retrieve()` → `fuse_results()`. This lets the same evaluation code path handle all three conditions.

Output is a `BaselineReport` with a `summary_table()` method:

```
System                 HR@5       MRR@5      NDCG@5     HN_Rate@5
----------------------------------------------------------------
BM25 (sparse-only)     x.xxxx     x.xxxx     x.xxxx     x.xxxx
bge-large (dense-only) x.xxxx     x.xxxx     x.xxxx     x.xxxx
Hybrid RRF (ours)      x.xxxx     x.xxxx     x.xxxx     x.xxxx
```

---

## 14. CLI Entry Point

**File:** `main.py`

Five execution modes controlled by the `--mode` flag:

### `--mode index`

Builds and saves both retrieval indices. Run this once before any other mode. Required after any change to the corpus or the dense model.

```bash
python main.py --mode index
```

Steps:
1. `CorpusLoader` loads and validates the corpus.
2. `SparseRetriever.index()` builds the BM25 inverted index.
3. `SparseRetriever.save()` writes `sparse.bm25.pkl` + `sparse_article_map.pkl`.
4. `DenseRetriever.index()` embeds all articles with bge-large-en-v1.5 (batched, ~10-20 minutes on CPU).
5. `DenseRetriever.save()` writes `dense.faiss` + `dense_article_map.pkl`.

### `--mode query`

Loads saved indices, runs a single query through retrieval + generation, prints results.

```bash
python main.py --mode query --query "What are the ETS Phase 4 auctioning rules?"
```

Requires Ollama running with `llama3.3:70b`. The retrieval step works without Ollama; generation will fail.

### `--mode evaluate`

Runs the full 71-query gold standard evaluation. Does NOT require Ollama (no LLM calls).

```bash
python main.py --mode evaluate
```

Saves the full report to `data/indices/evaluation_report.json`.

### `--mode baselines`

Runs the three-way ablation study (BM25-only, dense-only, hybrid). The primary experiment for the paper.

```bash
python main.py --mode baselines
```

Saves to `data/indices/baseline_report.json`.

### `--mode pipeline`

Runs index → query → evaluate in sequence. Useful for demos and fresh reproductions.

```bash
python main.py --mode pipeline --query "What are the carbon border adjustment rules?"
```

---

## 15. Demo UI

**File:** `app.py`

A Gradio web UI for interactive querying. Accessible at `http://localhost:7860` after:

```bash
pip install gradio==6.15.2
python app.py
```

**Graceful degradation:** The app detects which components are available at startup and operates in the best available mode:

| Condition | Mode |
|---|---|
| Dense index + Ollama running | Full RAG (retrieval + generation) |
| Dense index exists, Ollama not running | Retrieval-only (hybrid BM25 + dense, no answer) |
| No dense index | Sparse-only (BM25, fully offline) |

If no saved index exists, the BM25 index is built in-memory at startup.

**Features:**
- Free-text query input
- 8 domain-grouped sample queries (ETS, CBAM, Taxonomy, Maritime, LULUCF, Climate Law, F-Gas, ESR)
- Evidence cards showing each retrieved provision with provenance badges (keyword-only / semantic-only / both)
- Generated answer with citation support
- Stats bar (retrieval latency, generation latency, citation count)

---

## 16. Analysis Scripts

Both scripts must be run from the project root after building indices (`--mode index`).

### `scripts/sweep_weights.py`

Evaluates the hybrid retriever at multiple dense:sparse weight ratios to find the optimal RRF weighting.

```bash
python scripts/sweep_weights.py
```

Tests ratios: `1:1`, `2:1`, `3:1`, `5:1`, `10:1`. Reports HR@5 and MRR@5 for each, plus single-retriever baselines as reference lines. Use this to justify the weight choice in the paper or to tune `rrf_dense_weight` / `rrf_sparse_weight` in `config.py`.

### `scripts/breakdown_by_difficulty.py`

Splits the gold standard by `difficulty` tag and reports metrics per bucket.

```bash
python scripts/breakdown_by_difficulty.py
```

Answers: "Does BM25 still win on standard-difficulty (keyword-heavy) queries even though the hybrid wins overall?" If yes, that confirms each retriever has a distinct domain of competence and RRF is genuinely combining complementary signals.

---

## 17. Test Suite

**Files:** `tests/test_retrieval.py`, `tests/test_fusion.py`, `tests/test_evaluation.py`

46 tests total. All tests are fully offline — no API keys, no Ollama, no internet connection required.

```bash
pip install pytest==9.0.3
python -m pytest tests/ -v
# Expected: 46 passed
```

### `test_retrieval.py` — 19 tests

**`TestTokenizer` (3 tests):**
- Hyphenated terms preserved (`"net-zero"` → `["net-zero"]`)
- Lowercased correctly
- Punctuation stripped

**`TestSparseRetriever` (12 tests):**
- `is_indexed` is False before indexing
- `name` returns `"sparse"`
- `retrieve` raises if not indexed
- Best match on exact terms
- Results are ranked in descending order
- `retrieve_returns_top_k` respects the `top_k` parameter
- `scores_are_non_negative`
- `retriever_name_field` in `RetrievedResult`
- `save_and_load` round-trip

**`TestDenseRetrieverMocked` (4 tests):**
- Dense retriever is tested with a mocked `SentenceTransformer` to avoid downloading the model in CI
- `is_indexed`, `retriever_name`, retrieve returns results, ranks are contiguous

### `test_fusion.py` — 8 tests

**`TestRRFMath` (7 tests):**
- RRF score formula correctness (exact value)
- Document in both lists scores higher than document in one list
- Results only from dense (sparse empty) — fusion still works
- Provenance (dense_rank, sparse_rank) tracked correctly
- Ranks are 1-based and contiguous
- `top_k_fused` limits output length
- Empty retrievers return empty list

**`TestRRFDeduplication` (1 test):**
- Same `doc_id` appearing in both lists is deduplicated into one result

### `test_evaluation.py` — 19 tests

**`TestBaseCelex` (3 tests):** Corrigendum suffix stripping

**`TestHitRate` (3 tests):** Perfect / zero / partial hit rate

**`TestMRR` (5 tests):** Rank-1 / rank-2 / rank-3 / no-hit / corrigendum

**`TestNDCG` (7 tests):**
- Rank-1 → NDCG = 1.0
- Rank-2 → expected discounted value
- No relevant → 0.0
- Two relevant both at top → 1.0
- Two relevant, one missing → between 0 and 1
- Corrigendum handled correctly
- Empty retrieved → 0.0

**`TestHardNegativeCount` (4 tests):**
- HN in results counted correctly
- No HN → 0
- HN outside k → not counted
- Multiple HNs → all counted

---

## 18. Setup and First Run

**Requirements:** Python 3.10+, ~4GB disk space for model, ~40GB for Ollama generation model.

### Step 1 — Clone and create environment

```bash
git clone <repo-url>
cd eu-lexical-semantic-legal-rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Install Ollama (for generation only)

```bash
# macOS
brew install ollama
ollama pull llama3.3:70b          # ~40GB download — required for generation only
ollama serve                      # Start the Ollama server
```

The bge-large-en-v1.5 embedding model (~1.3GB) downloads automatically from HuggingFace on first `--mode index` run.

### Step 3 — Build indices

```bash
python main.py --mode index
```

Expected output:
```
Stage: Index
Loaded 1156 articles from data/corpus/eu_climate_articles.jsonl
Building sparse (BM25) index…
Building dense (bge-large-en-v1.5) index…
✓ Indices saved to data/indices
```

This takes 20–40 minutes on CPU (embedding 1,156 articles with bge-large-en-v1.5). It runs once; subsequent runs load from disk in seconds.

### Step 4 — Run the ablation study

```bash
python main.py --mode baselines
```

### Step 5 — Query the system

```bash
python main.py --mode query \
  --query "What are the monitoring obligations for maritime GHG emissions?"
```

### Step 6 — Run tests

```bash
python -m pytest tests/ -v
```

---

## 19. End-to-End Workflow

### Data flows through the pipeline as follows:

```
JSONL corpus
    │
    ▼ CorpusLoader.load()
list[LegalArticle]
    │
    ├─────────────────────────────────────────────┐
    ▼                                             ▼
SparseRetriever.index()              DenseRetriever.index()
BM25Okapi object                     FAISS IndexFlatIP
sparse.bm25.pkl                      dense.faiss
sparse_article_map.pkl               dense_article_map.pkl
    │                                             │
    └──────────────── loaded at query time ───────┘
                              │
                              ▼ Query: "What are ETS allowance rules?"
                    RankFusionController.fuse_results()
                         │              │
              sparse.retrieve()    dense.retrieve()
              (BM25 scores)        (cosine similarity)
                list[RetrievedResult]  list[RetrievedResult]
                         │              │
                    ThreadPoolExecutor (concurrent)
                         └──────┬───────┘
                                ▼
                    RRF: w / (k + rank) per retriever
                    deduplicate on doc_id
                    sort descending by RRF score
                                │
                         list[FusedResult]
                                │
                    ┌───────────┴──────────────┐
                    ▼                          ▼
            LegalGenerator               Evaluator
            (LLM generation)             (IR metrics)
                    │                          │
           GenerationOutput            EvaluationReport
           (answer + citations)        (HR, MRR, NDCG, HN_Rate)
```

---

## 20. Key Design Decisions and Trade-offs

### Article-level chunking vs. token-level chunking

**Decision:** Index at the article level (one article = one document).

**Rationale:** EU legislative articles are semantically self-contained — each article states one obligation, definition, or procedure. Article-level chunking avoids arbitrary window-size decisions, produces units that align with how lawyers cite law (`[32003L0087 — Article 12]` is a standard legal citation), and eliminates the boundary problem (a provision split across two chunks cannot be retrieved as a unit).

**Trade-off:** Some articles are very long (preambles, annexes), which could challenge the embedding model's context window. bge-large-en-v1.5 has a 512-token context window — articles longer than ~380 words will be truncated at embedding time. This is a known limitation.

### RRF vs. learned fusion weights

**Decision:** Reciprocal Rank Fusion with equal weights (1.0 / 1.0).

**Rationale:** RRF requires no training data, is parameter-free (k=60 is the standard validated default), and is robust to the score distribution mismatch between BM25 (unbounded positive reals) and cosine similarity ([-1, 1]). Learned fusion (e.g., a linear classifier over scores) would require labelled training data — data the project does not have.

**Trade-off:** Equal weights may not be optimal. The `sweep_weights.py` script can identify a better ratio. However, the gain from tuning weights is likely small compared to the gain from combining both retrievers at all.

### CELEX-level evaluation vs. article-level evaluation

**Decision:** Evaluation matches at the CELEX-ID level (instrument level).

**Rationale:** The gold standard maps queries to instruments, not to specific articles. Multiple articles from the same instrument may be relevant to the same query. Evaluating at the article level would require annotating which specific article within each instrument is most relevant — a much more labour-intensive task.

**Trade-off:** A system that retrieves a low-information article from the right instrument scores the same as one that retrieves the most relevant article. This means measured HR@5 may be an overestimate of practical usefulness.

### Strict grounding in the system prompt

**Decision:** The system prompt forbids the LLM from using any knowledge outside the provided context.

**Rationale:** Legal analysis requires traceable claims. Allowing the LLM to draw on pre-training knowledge produces unverifiable statements. The `INSUFFICIENT CONTEXT` fallback ensures the system does not hallucinate legal conclusions when the retrieved provisions are inadequate.

**Trade-off:** Answers may be less fluent or complete than a system with relaxed grounding. For research purposes, this is preferable — a conservative system whose failures are visible is more useful than a confident system that is wrong silently.

### Why Ollama / local LLM vs. OpenAI API

**Decision:** Llama 3.3 70B via Ollama rather than GPT-4 or Claude.

**Rationale:** Reproducibility. An academic system that depends on a commercial API cannot be fully reproduced by reviewers, and its behaviour may change between runs. Ollama runs locally with a fixed model version, producing deterministic results given the same inputs.

**Trade-off:** Requires ~40GB disk and a machine with enough RAM to run the model. Inference is slower than hosted APIs. The embedding model (bge-large-en-v1.5) is also local for the same reason.

---

## 21. Known Limitations and Open Issues

### 1. FAISS index must be rebuilt after model swap

The index at `data/indices/dense.faiss` was built with the original bge-m3 model and has not been rebuilt since the model swap to bge-large-en-v1.5. Until `python main.py --mode index` is run, the dense retriever will load a stale index and produce incorrect results.

**Action required:** `python main.py --mode index`

### 2. Corpus/benchmark size discrepancy

The paper states 1,156 articles / 72 instruments / 71 queries. The README states 825 / 57 / 50. One of these is stale. Must be reconciled before paper submission.

**Action required:** Run `wc -l data/corpus/eu_climate_articles.jsonl` and count distinct CELEX IDs to determine the correct corpus statistics.

### 3. Hard negatives annotated for 20 of 71 queries only

The `hard_negative_celex_ids` field is populated for 20 queries. The HN_Rate metric is only computed over annotated queries, which means the reported HN_Rate covers only 28% of the benchmark. Full annotation of all 71 queries is needed for a publishable evaluation.

### 4. No inter-annotator agreement score

The gold standard was constructed by one annotator. For JURIX submission, a subset of queries (recommended: 20%, ~14 queries) should be independently annotated by a second person, and Cohen's κ computed.

### 5. 512-token truncation in bge-large-en-v1.5

Articles longer than approximately 380 words will be silently truncated to 512 tokens during embedding. The system does not warn when this happens. Long articles (typically preambles or definition-heavy provisions) may have their tail text excluded from the vector representation.

### 6. No temporal filtering

The corpus contains both current and superseded versions of instruments. The system does not filter by temporal validity. A query about Phase 4 ETS rules may retrieve Phase 1 articles (original 2003/87) alongside current Phase 4 articles (2023 revision). The hard-negative annotations partially address this for evaluation; the retrieval system itself is still temporally unaware.

### 7. English-only

The corpus uses English text only. All 24 official EU languages are legally equivalent. Non-English queries will produce degraded results.

### 8. No reranking

The pipeline uses a single-stage retrieval architecture. Adding a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) on top of the fused top-20 results would likely improve precision at rank-1 with minimal latency cost.

### 9. Placeholder results in the paper

The paper at `docs/paper/shafat2025_jurix_standalone.tex` still contains `[PLACEHOLDER]` cells in the results tables. These will be filled once the FAISS index is rebuilt and `python main.py --mode baselines` is run.

---

## 22. Glossary

| Term | Definition |
|---|---|
| **CELEX ID** | Identifier used by EUR-Lex to uniquely identify EU legal acts. Format: `SECTOR + YEAR + TYPE + NUMBER`. E.g., `32003L0087` = sector 3, year 2003, Directive, number 0087. |
| **EUR-Lex** | The official database of EU law (`eur-lex.europa.eu`). |
| **CELLAR** | The EU's semantic repository and content management system behind EUR-Lex. Exposes a SPARQL endpoint. |
| **Formex 4** | XML format used by the EUR-Lex publication office for structured EU legislative text. |
| **ETS** | EU Emissions Trading System. The main carbon market mechanism (CELEX `32003L0087` + amendments). |
| **CBAM** | Carbon Border Adjustment Mechanism. Import carbon pricing mechanism (CELEX `32023R0956`). |
| **LULUCF** | Land Use, Land-Use Change and Forestry. Carbon accounting for land (CELEX `32018R0841`). |
| **ESR** | Effort Sharing Regulation. National emission targets for non-ETS sectors (CELEX `32018R0842`). |
| **Taxonomy** | EU Taxonomy Regulation for sustainable finance classification (CELEX `32020R0852`). |
| **BM25** | Best Match 25 — a ranking function based on TF-IDF with term frequency saturation and document length normalisation. Standard baseline for lexical search. |
| **bi-encoder** | A neural architecture where query and document are encoded independently into a shared vector space. Enables fast retrieval via pre-computed document embeddings. Contrast with cross-encoder. |
| **cross-encoder** | A neural architecture where query and document are encoded jointly. More accurate but requires encoding at query time (no pre-computation). |
| **FAISS** | Facebook AI Similarity Search — a library for fast nearest-neighbour search in vector spaces. `IndexFlatIP` = exact inner-product search. |
| **RRF** | Reciprocal Rank Fusion. A score-free rank combination method: `RRF(d) = Σ w/(k + rank)`. |
| **RAG** | Retrieval-Augmented Generation. Architecture where a retriever supplies context to an LLM before generation. |
| **Ollama** | Tool for running open-source LLMs locally. Exposes an OpenAI-compatible REST API. |
| **MRR** | Mean Reciprocal Rank. Average of 1/rank_of_first_relevant_document over all queries. |
| **NDCG** | Normalised Discounted Cumulative Gain. Position-aware metric that rewards multiple relevant results at high ranks. |
| **HR@k** | Hit Rate at cut-off k. Fraction of queries where any relevant document appears in top-k. |
| **HN_Rate@k** | Hard-Negative Rate at cut-off k. Fraction of top-k slots occupied by annotated hard-negative documents. |
| **Hard negative** | A document that shares vocabulary with a query but is legally incorrect. Used to test whether the retrieval system makes genuine legal distinctions rather than surface-level keyword matches. |
| **Corrigendum** | An official correction notice to an EU legislative act. Identified by `R(NN)` suffix in the CELEX ID. The system strips these for evaluation matching. |
| **EuroVoc** | The EU's multilingual thesaurus of legal and policy concepts. Concept IDs are attached to articles as `concept_ids`. |
