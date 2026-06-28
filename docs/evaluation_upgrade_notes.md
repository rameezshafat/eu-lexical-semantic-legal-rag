# Evaluation Upgrade: Hard Negatives and Ranking-Sensitive Metrics

**Branch:** `feature/ananya`
**Date:** 2026-06-26

---

## Background

The supervisor feedback identified two weaknesses in the original evaluation design:

1. The negative pool was implicit — every document not in `relevant_celex_ids` was treated as a negative. In a focused EU climate law corpus where most instruments share vocabulary (`emissions`, `allowances`, `monitoring`, `national targets`), a retrieval system could achieve high Hit Rate@5 purely through keyword or semantic overlap without making any legally meaningful distinction.

2. Hit Rate@5 was the primary metric. It gives equal credit to a system that returns the correct document at rank 1 and one that buries it at rank 5 behind four wrong results. In a legal setting, rank-1 is what the user reads first.

The changes below address both issues.

---

## Changes Made

### 1. Hard Negatives Added to Gold Standard

**File:** `data/evaluation/gold_standard.json`

Added a `hard_negative_celex_ids` field to 20 of the 71 queries. Each hard negative is a document that:
- Is present in the corpus
- Shares surface vocabulary or policy area with the query
- Is legally incorrect for that specific question

No hard negative overlaps with the relevant set for its query (validated programmatically).

**Two categories of hard negatives were used:**

**Cross-sector / cross-instrument** — documents from a different legal mechanism that share the same vocabulary:

| Query | Relevant | Hard Negative | Why it confuses BM25/dense |
|---|---|---|---|
| Maritime CO2 reporting (q001) | `32015R0757` | `32008L0101` | Aviation ETS — same CO2/monitoring/transport language |
| ETS cap + allowances (q002) | `32003L0087`, `32023L0959` | `32015R0757` | Maritime MRV reports emissions but doesn't set the cap |
| CBAM embedded carbon (q005) | `32023R0956` | `32003L0087` | ETS — same "carbon price" but domestic market, not border imports |
| LULUCF forests/farmland (q007) | `32018R0841`, `32023R0839` | `32018R0842` | ESR — same "national targets" but non-ETS sectors, not land use |
| ETS auctioning current phase (q012) | `32003L0087`, `32023L0959` | `32008L0101` | Aviation ETS add-on — same "auctioning" vocabulary |
| Taxonomy technical criteria (q015) | `32021R2139`, `32020R0852` | `32021R2178` | Taxonomy disclosure reg — same vocabulary, wrong focus |
| Fund disclosure about green products (q016) | `32021R2178`, `32020R0852` | `32021R2139` | Taxonomy criteria reg — same vocabulary, wrong focus |
| Airline monitoring plans (q021) | `32023L0959` | `32015R0757` | Maritime MRV — same "monitoring plan" language, wrong transport mode |
| Shipping carbon obligations (q023) | `32023L0959`, `32015R0757` | `32008L0101` | Aviation ETS — same "allowances"/"operator"/"surrender" but wrong mode |
| EU green bond requirements (q027) | `32023R2631` | `32020R0852` | Taxonomy Reg — same "green"/"sustainable" but covers activities not bonds |
| Free permits for heavy industry (q037) | `32003L0087`, `32023L0959` | `32023R0955` | Social Climate Fund — same carbon-price impact framing but compensation not exemption |
| Union Registry tracking (q039) | `32018R0208`, `32003L0087` | `32015R1844` | Kyoto credits reg — same "registry"/"transfer"/"credits" but Kyoto not ETS allowances |
| ESR flexibilities (q040) | `32018R0842` | `32018R0841` | LULUCF — same "national accounting" but land use not ESR carry/bank/borrow |
| LULUCF forest baseline (q041) | `32018R0841` | `32013R0525` | Monitoring Mechanism — same "GHG accounting" but Union-wide inventory not LULUCF baseline |
| CBAM free-allocation adjustment (q044) | `32025R2620`, `32023R0956` | `32003L0087` | ETS free allocation — same vocabulary but domestic producers not CBAM |
| Taxonomy DNSH test (q049) | `32021R2139`, `32020R0852` | `32020R1818` | Paris-aligned benchmarks — same sustainability criteria language but indices not activities |

**Temporal / obsolete-version** — an earlier or superseded instrument that shares the same text but contains outdated rules:

| Query | Relevant (current) | Hard Negative (obsolete) | Temporal distinction required |
|---|---|---|---|
| How aviation entered carbon market (q022) | `32008L0101`, `32003L0087` | `32023L0959` | Query asks about the 2008 inclusion — the 2023 revision post-dates the question |
| Energy savings obligation (q025) | `32023L1791` | `32018L2002` | 2018 EED amendment superseded by 2023 recast with higher targets |
| Energy efficiency first principle (q026) | `32023L1791` | `32018L2002` | Same principle exists in old EED but the operative formulation changed in the recast |
| 2026–2029 LULUCF annual limits (q067) | `32026R0893`, `32018R0841` | `32023R0839` | 2023 LULUCF revision — same family but the specific 2026–2029 limits are only in the implementing reg |

---

### 2. NDCG@5 and HN\_Rate@5 Added to Evaluator

**Files:** `src/evaluation/evaluator.py`, `src/models/schemas.py`

#### Why NDCG@5

MRR only considers the rank of the **first** relevant document. Many queries in this benchmark have two or more relevant instruments (28 of 71 queries). If the second relevant instrument is missing from the top-5, MRR does not penalise this. NDCG@5 does.

```
DCG@k  = Σ_{i=1}^{k}  rel_i / log₂(i + 1)
NDCG@k = DCG@k / IDCG@k
```

Where IDCG is the ideal DCG achieved if all relevant documents are placed at the top ranks. With binary relevance, NDCG@k = 1.0 only when every relevant document appears in the top-k in the best possible order.

Concretely: a system that returns four hard negatives then the correct document at rank 5 scores:
- HR@5 = 1.0 (misleading — masks the failure)
- MRR@5 = 0.20 (honest about rank)
- NDCG@5 ≈ 0.17 (also penalises the four wasted slots)

#### Why HN\_Rate@5

HN\_Rate@5 measures how often the system is fooled by hard negatives — the fraction of top-5 result slots occupied by annotated hard-negative documents. This is the direct diagnostic for whether the system is doing legal reasoning or surface matching.

Expected behaviour:
- **BM25** should have a high HN\_Rate: it matches keywords regardless of legal sector, so aviation and maritime ETS articles are interchangeable to it.
- **bge-large (dense)** should do better: semantic representations encode meaning, not just tokens.
- **Hybrid RRF** should do best: the dense signal can correct BM25's sector-blind retrieval.

If the hybrid HN\_Rate is not lower than BM25, it means the fusion is not adding genuine legal discrimination, only retrieval coverage.

HN\_Rate is computed **only over the 20 annotated queries** — unannotated queries contribute 0.0 unconditionally, which would dilute the aggregate unfairly.

#### Implementation

Two new module-level functions in `evaluator.py`:

```python
def _ndcg(relevant_set: set[str], retrieved: list[str], k: int) -> float:
    gains = [1.0 if _base_celex(c) in relevant_set else 0.0 for c in retrieved[:k]]
    dcg   = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg  = sum(1.0 / math.log2(j + 2) for j in range(min(len(relevant_set), k)))
    return dcg / idcg if idcg > 0.0 else 0.0

def _count_hard_negatives(hard_negative_set: set[str], retrieved: list[str], k: int) -> int:
    return sum(1 for c in retrieved[:k] if _base_celex(c) in hard_negative_set)
```

Both functions respect corrigendum normalisation via `_base_celex`.

---

### 3. Schema Updates

**File:** `src/models/schemas.py`

| Class | Field added | Default | Purpose |
|---|---|---|---|
| `GoldQuery` | `hard_negative_celex_ids` | `[]` | Stores annotated hard negatives per query |
| `GoldQuery` | `difficulty` | `"standard"` | Was in JSON but previously silently ignored |
| `EvaluatedQuery` | `ndcg_at_k` | `0.0` | Per-query NDCG score |
| `EvaluatedQuery` | `hard_negatives_in_top_k` | `0` | Raw count of HN docs in top-k for this query |
| `EvaluationReport` | `ndcg` | `0.0` | Aggregate mean NDCG@k |
| `EvaluationReport` | `hard_negative_rate` | `0.0` | Mean HN\_Rate over annotated queries |

All defaults ensure full backwards compatibility — existing tests and the main pipeline require no changes.

The `summary_table()` in `BaselineReport` now reports all four metrics side by side:

```
System                 HR@5       MRR@5      NDCG@5     HN_Rate@5
----------------------------------------------------------------
BM25 (sparse-only)     x.xxxx     x.xxxx     x.xxxx     x.xxxx
bge-large (dense-only) x.xxxx     x.xxxx     x.xxxx     x.xxxx
Hybrid RRF (ours)      x.xxxx     x.xxxx     x.xxxx     x.xxxx
```

---

### 4. New Tests

**File:** `tests/test_evaluation.py`

11 new tests added across two classes:

**`TestNDCG`** (7 tests):
- Single relevant doc at rank 1 → NDCG = 1.0
- Single relevant doc at rank 2 → expected discounted value
- No relevant doc returned → 0.0
- Two relevant docs both at top-2 → 1.0
- Two relevant docs, one missing → value between 0 and 1
- Corrigendum suffix handled correctly
- Empty retrieved list → 0.0

**`TestHardNegativeCount`** (4 tests):
- HN in results → counted correctly
- No HN in results → 0
- HN appears at rank 3 with k=2 → not counted (outside cutoff)
- Multiple HNs → all counted

Total test count: **35 → 46**, all passing.

---

## What Has Not Changed

- BM25 sparse retriever — no changes
- Dense retriever (`bge-large-en-v1.5`) — no changes
- FAISS index — still needs to be rebuilt after the earlier model swap (`python main.py --mode index`)
- RRF fusion controller — no changes
- Main pipeline entrypoint — no changes
- The 51 queries without hard negative annotations still evaluate correctly; they simply contribute 0.0 to HN\_Rate (which is excluded from the HN\_Rate aggregate)

---

## Remaining Work

| Task | Why |
|---|---|
| Rebuild FAISS dense index | Existing index was built with bge-m3 vectors — stale since model swap |
| Run `python main.py --mode baselines` | Fills all `[PLACEHOLDER]` cells in the paper results tables |
| Extend hard negative annotation to remaining 51 queries | 20/71 annotated is sufficient for initial signal; full annotation needed for submission |
| Inter-annotator agreement (Cohen's κ) | Hard negative judgements are subjective; a second annotator on 20% of the set defends benchmark quality |
| Reconcile corpus size discrepancy | Paper claims 1,156 articles / 72 instruments / 71 queries; README states 825 / 57 / 50 |
