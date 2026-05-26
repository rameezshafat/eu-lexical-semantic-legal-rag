# eu-lexical-semantic-legal-rag

Article-level ETL pipeline for EU climate law, extracting structured legal text from EUR-Lex/CELLAR to support hybrid lexical-semantic legal RAG.

## Contents

- `eu_cellar_etl.ipynb` - notebook for CELLAR/EUR-Lex extraction and parsing
- `src/` - Python source modules
- `tests/` - test scaffolding
- `data/` - project data assets
- `eu_climate_articles.jsonl` - exported article-level dataset

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
