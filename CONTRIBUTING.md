# Contributing

## Reproduction checklist

To reproduce the paper results from scratch:

```bash
# 1. Install dependencies
python -m pip install -e ".[etl,dev]"

# 2. Build corpus (requires network access to EU CELLAR SPARQL endpoint)
python main.py --mode index

# 3. Verify corpus size: should be 1,166 articles from 72 CELEX instruments
python -c "
import pickle
with open('data/indices/sparse_article_map.pkl', 'rb') as f:
    arts = pickle.load(f)
celex = {a.celex_id for a in arts}
print(f'{len(arts)} articles, {len(celex)} instruments')
"

# 4. Run full evaluation (all 71 queries, all 3 systems)
python scripts/eval_test.py

# 5. Run statistical significance tests
python scripts/significance_test.py

# 6. Run bootstrap confidence intervals
python scripts/bootstrap_ci.py

# 7. Run RRF k-ablation sweep
python scripts/plot_rrf_sweep.py
```

Expected results (71-query full set):

| System | HR@1 | HR@5 | MRR@5 |
|--------|------|------|-------|
| BM25-only | 0.465 | 0.704 | 0.556 |
| nomic-embed-only | 0.676 | 0.958 | 0.785 |
| Hybrid RRF k=20 dw=5 | 0.690 | 0.972 | 0.795 |

## Environment

- Python ≥ 3.10 (tested on 3.14.4)
- No GPU required; all retrieval runs on CPU
- Ollama with `llama3.3:70b` required only for `scripts/eval_generation.py`

## Key configuration

All parameters are in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RRF_K` | `20` | RRF smoothing constant |
| `RRF_DENSE_WEIGHT` | `5.0` | Dense retriever weight |
| `TOP_K_RETRIEVAL` | `100` | Candidates per retriever |
| `LLM_MODEL` | `llama3.3:70b` | Generation model |

## Running tests

```bash
python -m pytest tests/ -q
```

All 67 tests should pass.

## Inter-annotator agreement

To compute Cohen's κ after collecting a second annotator's judgements:

```bash
python scripts/compute_iaa.py \
  --a1 data/evaluation/iaa_annotator1.json \
  --a2 data/evaluation/iaa_annotator2.json
```

The annotator instructions and 14-query subset are in
`docs/annotator_instructions.md`.
