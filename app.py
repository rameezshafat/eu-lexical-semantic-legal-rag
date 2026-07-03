"""
app.py - Demo UI for the Lexico-Semantic Fusion Legal RAG system.

Tabs
----
  Search       - single query; mode selector (Full RAG / Hybrid / Dense Only / BM25)
  Compare      - same query run through BM25, Dense, and Hybrid side-by-side
  Evaluation   - baseline metrics table from the last evaluation run

Run:
    pip install gradio==6.15.2
    python app.py
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
load_dotenv()

import json
import re
import time
from pathlib import Path

import gradio as gr

from config import settings
from src.ingestion.chunker import apply_hierarchical_chunking
from src.ingestion.loader import CorpusLoader
from src.retrieval.sparse import SparseRetriever

# ── Capability detection ──────────────────────────────────────────────────────

_INDEX_DIR = Path(settings.index_dir)
_HAS_DENSE = (_INDEX_DIR / "dense.faiss").exists()

# ── Startup: load all available components ────────────────────────────────────

print("Initialising demo…")
_loader   = CorpusLoader(settings.corpus_path)
_articles = apply_hierarchical_chunking(
    _loader.load(), settings.chunk_token_limit
)
print(f"  {len(_articles)} chunks loaded")

_sparse = SparseRetriever(k1=settings.bm25_k1, b=settings.bm25_b)
if (_INDEX_DIR / "sparse.bm25.pkl").exists():
    _sparse.load(str(_INDEX_DIR))
    print("  Sparse (BM25) index loaded")
else:
    _sparse.index(_articles)
    print("  Sparse (BM25) index built in-memory")

_dense      = None
_controller = None
_generator  = None

if _HAS_DENSE:
    from src.retrieval.dense import DenseRetriever
    from src.fusion.controller import RankFusionController
    _dense = DenseRetriever(
        model=settings.dense_model,
        embed_dim=settings.dense_embed_dim,
        batch_size=settings.dense_batch_size,
        device=settings.dense_device,
    )
    _dense.load(str(_INDEX_DIR))
    _controller = RankFusionController(
        dense_retriever=_dense,
        sparse_retriever=_sparse,
        rrf_k=settings.rrf_k,
        top_k_retrieval=settings.top_k_retrieval,
        top_k_fused=settings.top_k_fused,
        dense_weight=settings.rrf_dense_weight,
        sparse_weight=settings.rrf_sparse_weight,
    )
    print("  Dense (nomic-embed-text-v1.5) index + RankFusionController loaded")

try:
    from src.generation.generator import LegalGenerator
    _generator = LegalGenerator(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        base_url=settings.llm_ollama_base_url,
    )
    print(f"  LegalGenerator ({settings.llm_model}) ready")
except Exception as _exc:
    print(f"  LegalGenerator unavailable: {_exc}")

if _generator and _controller:
    _MODE, _MODE_COLOR = "Full RAG", "#1d4ed8"
elif _controller:
    _MODE, _MODE_COLOR = "Retrieval-Only", "#0369a1"
else:
    _MODE, _MODE_COLOR = "Sparse-Only (BM25)", "#065f46"

print(f"\nDemo mode: {_MODE}\n")

# Build mode list in priority order (best available first)
_AVAILABLE_MODES: list[str] = ["Sparse Only (BM25)"]
if _dense:
    _AVAILABLE_MODES.insert(0, "Dense Only")
if _controller:
    _AVAILABLE_MODES.insert(0, "Retrieval Only (Hybrid)")
if _generator and _controller:
    _AVAILABLE_MODES.insert(0, "Full RAG")
_DEFAULT_MODE = _AVAILABLE_MODES[0]

# ── Load baseline report (for Evaluation tab) ─────────────────────────────────

_BASELINE_REPORT: dict | None = None
_baseline_path = _INDEX_DIR / "baseline_report.json"
if _baseline_path.exists():
    _BASELINE_REPORT = json.loads(_baseline_path.read_text(encoding="utf-8"))
    print("  Baseline report loaded")

# ── Load hard negative CELEX IDs (for flagging in evidence cards) ─────────────

_HARD_NEG_CELEX: set[str] = set()
_gold_path = Path(settings.gold_standard_path)
if _gold_path.exists():
    _gold_data = json.loads(_gold_path.read_text(encoding="utf-8"))
    for _q in _gold_data.get("queries", []):
        for _c in _q.get("hard_negative_celex_ids", []):
            _HARD_NEG_CELEX.add(_c)
    print(f"  Hard-negative index: {len(_HARD_NEG_CELEX)} CELEX IDs loaded")

# ── Sample queries (curated, domain-grouped) ──────────────────────────────────

_SAMPLES = {
    "ETS":        "How are emission allowances allocated to industrial installations in EU ETS Phase 4?",
    "CBAM":       "How does CBAM calculate the embedded emissions in imported steel and cement?",
    "Taxonomy":   "What criteria must an economic activity meet to be environmentally sustainable under the Taxonomy Regulation?",
    "Maritime":   "What are the monitoring and reporting obligations for greenhouse gas emissions from ships?",
    "LULUCF":     "What land use accounting rules apply to member states under the LULUCF Regulation?",
    "Climate Law":"What is the EU's legally binding 2050 climate neutrality target and governance framework?",
    "F-Gas":      "What are the reporting requirements for fluorinated greenhouse gases?",
    "ESR":        "What are the binding annual emission targets for member states under the Effort Sharing Regulation?",
}

_EURLEX_BASE = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:"
_CELEX_STRIP_RE = re.compile(r"R\(.*$")


def _celex_url(celex_id: str) -> str:
    return _EURLEX_BASE + _CELEX_STRIP_RE.sub("", celex_id)


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
*, *::before, *::after { box-sizing: border-box; }

.hero { padding: 20px 0 8px; }
.hero h1 { font-size: 1.6rem; font-weight: 700; margin: 0 0 4px;
           color: #0f172a; letter-spacing: -0.02em; }
.hero .subtitle { font-size: 0.9rem; color: #475569; margin: 0 0 10px; }

.dark .hero h1       { color: #f1f5f9; }
.dark .hero .subtitle{ color: #94a3b8; }
.dark .corpus-stat   { color: #94a3b8; }
.dark .corpus-stat b { color: #e2e8f0; }
.dark .sample-label, .dark .query-label { color: #94a3b8; }
.dark .evidence-header { color: #cbd5e1; }
.dark .compare-col-header { color: #cbd5e1; border-color: #334155; background: #1e293b; }

.mode-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
    font-weight: 600; color: white; margin-bottom: 16px;
}
.corpus-stat { font-size: 0.8rem; color: #64748b; margin-top: 4px; }
.corpus-stat b { color: #1e293b; }

.query-label { font-size: 0.8rem; font-weight: 600; color: #374151;
               text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
.sample-label { font-size: 0.75rem; font-weight: 600; color: #6b7280;
                text-transform: uppercase; letter-spacing: 0.05em; margin: 12px 0 6px; }

.stats-bar {
    display: flex; gap: 16px; padding: 8px 14px;
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; margin-bottom: 12px;
    font-size: 0.78rem; color: #475569 !important; flex-wrap: wrap;
}
.stats-bar .stat     { display: flex; align-items: center; gap: 5px; color: #475569 !important; }
.stats-bar .stat-val { font-weight: 700; color: #0f172a !important; }
.stats-bar .sep      { color: #cbd5e1 !important; }

.answer-box {
    border-left: 3px solid #2563eb; padding: 14px 18px;
    background: #f0f6ff; border-radius: 0 8px 8px 0;
    font-size: 0.92rem; line-height: 1.7; color: #1e293b !important; margin-bottom: 20px;
}
.answer-box p { margin: 0 0 10px; color: #1e293b !important; }
.answer-box p:last-child { margin: 0; }

.evidence-header {
    font-size: 0.8rem; font-weight: 700; color: #374151;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 4px 0 10px; display: flex; align-items: center; gap: 8px;
}
.evidence-header .ev-count {
    background: #e2e8f0; color: #475569;
    padding: 1px 8px; border-radius: 10px; font-size: 0.72rem;
}

.ev-card {
    border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 12px 14px; margin-bottom: 10px; background: #ffffff;
    transition: box-shadow 0.15s;
}
.ev-card:hover { box-shadow: 0 2px 10px rgba(0,0,0,.07); }
.ev-card-hn { border-color: #fca5a5; background: #fff5f5; }

.ev-card-top {
    display: flex; align-items: flex-start;
    justify-content: space-between; gap: 10px; margin-bottom: 8px; flex-wrap: wrap;
}
.ev-rank { font-size: 0.75rem; font-weight: 700; color: #94a3b8; min-width: 22px; padding-top: 1px; }
.ev-meta { flex: 1; }
.ev-celex { font-size: 0.82rem; font-weight: 700; }
.ev-celex a { color: #0f172a; text-decoration: none; border-bottom: 1px dashed #94a3b8; }
.ev-celex a:hover { color: #2563eb; border-color: #2563eb; }
.ev-article { font-size: 0.78rem; color: #475569; margin-top: 1px; }

.prov-both   { background:#fef9c3; border:1px solid #fbbf24; color:#92400e; }
.prov-dense  { background:#eff6ff; border:1px solid #93c5fd; color:#1e40af; }
.prov-sparse { background:#f0fdf4; border:1px solid #86efac; color:#14532d; }
.prov-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600;
}
.prov-icon { font-size: 0.65rem; }

.hn-badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: #fee2e2; border: 1px solid #f87171; color: #991b1b;
    padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600;
    margin-top: 3px;
}

.ev-rrf { font-weight: 600; color: #6366f1; font-size: 0.7rem; }

.ev-text-preview {
    font-size: 0.83rem; line-height: 1.6; color: #374151;
    border-top: 1px solid #f1f5f9; padding-top: 8px; margin-top: 4px;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
details.ev-expand { margin-top: 6px; }
details.ev-expand summary {
    font-size: 0.75rem; color: #6366f1; cursor: pointer; user-select: none;
    list-style: none; display: inline-flex; align-items: center; gap: 4px;
}
details.ev-expand summary::-webkit-details-marker { display: none; }
details.ev-expand[open] summary { color: #4f46e5; }
.ev-text-full {
    font-size: 0.82rem; line-height: 1.65; color: #374151;
    margin-top: 6px; white-space: pre-wrap; word-break: break-word;
}

.placeholder {
    color: #94a3b8; font-size: 0.88rem; text-align: center;
    padding: 32px 16px; border: 1px dashed #e2e8f0; border-radius: 10px;
}
.insufficient {
    border-left: 3px solid #f59e0b; background: #fffbeb;
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    font-size: 0.88rem; color: #78350f !important;
}
.no-llm {
    border-left: 3px solid #94a3b8; background: #f8fafc;
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    font-size: 0.88rem; color: #475569 !important;
}
.no-llm code { background:#e2e8f0; padding:1px 5px; border-radius:4px; font-size:0.8rem; }

/* ── Compare tab ── */
.compare-col-header {
    font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #374151; padding: 8px 12px;
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    margin-bottom: 12px; text-align: center;
}

/* ── Metrics tab ── */
.metrics-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 12px; }
.metrics-table th {
    background: #f1f5f9; color: #374151; font-weight: 700;
    padding: 10px 14px; text-align: left; border-bottom: 2px solid #e2e8f0;
}
.metrics-table td { padding: 10px 14px; border-bottom: 1px solid #f1f5f9; color: #1e293b; }
.metrics-table tr:last-child td { border-bottom: none; }
.metrics-table .best { color: #16a34a; font-weight: 700; }
.metrics-table .system-name { font-weight: 600; }
.metrics-intro {
    font-size: 0.83rem; color: #475569; line-height: 1.6; margin-bottom: 16px;
    padding: 12px 16px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;
}
"""


# ── HTML rendering helpers ────────────────────────────────────────────────────

def _provenance_badge(dense_rank, sparse_rank) -> str:
    if dense_rank is not None and sparse_rank is not None:
        return (
            f'<span class="prov-badge prov-both">'
            f'<span class="prov-icon">●</span>'
            f'BM25 #{sparse_rank} · Dense #{dense_rank}'
            f'</span>'
        )
    if dense_rank is not None:
        return (
            f'<span class="prov-badge prov-dense">'
            f'<span class="prov-icon">◆</span>'
            f'Dense #{dense_rank}'
            f'</span>'
        )
    return (
        f'<span class="prov-badge prov-sparse">'
        f'<span class="prov-icon">◈</span>'
        f'BM25 #{sparse_rank}'
        f'</span>'
    )


def _render_evidence_cards(results: list, header: str = "Retrieved Evidence") -> str:
    if not results:
        return '<div class="placeholder">No results retrieved.</div>'

    cards = []
    for r in results:
        a          = r.article
        is_hn      = a.celex_id in _HARD_NEG_CELEX
        hn_cls     = " ev-card-hn" if is_hn else ""
        badge      = _provenance_badge(
            getattr(r, "dense_rank", None),
            getattr(r, "sparse_rank", None),
        )
        score_attr = "rrf_score" if hasattr(r, "rrf_score") else "score"
        score_val  = getattr(r, score_attr, 0.0)
        doc_bg     = "#dbeafe" if a.doc_type == "Directive" else "#dcfce7"
        doc_co     = "#1e40af" if a.doc_type == "Directive" else "#14532d"
        hn_html    = (
            '<div class="hn-badge">⚠ known hard negative</div>'
            if is_hn else ""
        )
        celex_link = (
            f'<a href="{_celex_url(a.celex_id)}" target="_blank" '
            f'title="Open in EUR-Lex">{a.celex_id} ↗</a>'
        )
        preview    = a.article_text[:300] + ("…" if len(a.article_text) > 300 else "")
        full_text  = a.article_text.replace("<", "&lt;").replace(">", "&gt;")

        cards.append(f"""
<div class="ev-card{hn_cls}">
  <div class="ev-card-top">
    <div class="ev-rank">#{r.rank}</div>
    <div class="ev-meta">
      <div class="ev-celex">{celex_link}</div>
      <div class="ev-article">
        {a.article_number}&nbsp;
        <span style="background:{doc_bg};color:{doc_co};
                     padding:1px 6px;border-radius:8px;font-size:0.67rem;
                     font-weight:600;">{a.doc_type}</span>
      </div>
      {hn_html}
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px;">
      {badge}
      <span class="ev-rrf">{score_val:.5f}</span>
    </div>
  </div>
  <div class="ev-text-preview">{preview}</div>
  <details class="ev-expand">
    <summary>▶ Show full text ({len(a.article_text):,} chars)</summary>
    <div class="ev-text-full">{full_text}</div>
  </details>
</div>""")

    return (
        f'<div class="evidence-header">'
        f'{header} <span class="ev-count">{len(results)}</span>'
        f'</div>'
        + "".join(cards)
    )


def _render_stats_bar(retrieval_ms: float, gen_ms: float | None,
                      n_citations: int | None, mode: str) -> str:
    parts = [
        f'<div class="stat">Retrieval <span class="stat-val">{retrieval_ms:.0f} ms</span></div>',
        '<span class="sep">|</span>',
        f'<div class="stat">Mode <span class="stat-val">{mode}</span></div>',
    ]
    if gen_ms is not None:
        parts += [
            '<span class="sep">|</span>',
            f'<div class="stat">Generation <span class="stat-val">{gen_ms:.0f} ms</span></div>',
        ]
    if n_citations is not None:
        parts += [
            '<span class="sep">|</span>',
            f'<div class="stat">Citations <span class="stat-val">{n_citations}</span></div>',
        ]
    return f'<div class="stats-bar">{"".join(parts)}</div>'


def _render_answer(answer: str) -> str:
    if answer.startswith("INSUFFICIENT CONTEXT"):
        return f'<div class="insufficient">{answer}</div>'
    paras = "".join(f"<p>{p.strip()}</p>" for p in answer.split("\n\n") if p.strip())
    return f'<div class="answer-box">{paras}</div>'


def _render_metrics_html() -> str:
    if not _BASELINE_REPORT:
        return (
            '<div class="placeholder">No baseline report found.<br>'
            'Run: <code>python main.py --mode baselines</code></div>'
        )

    top_k  = _BASELINE_REPORT.get("top_k", 5)
    rows   = [
        ("BM25 (sparse-only)",        _BASELINE_REPORT.get("sparse_only") or {}),
        ("nomic-embed (dense-only)",  _BASELINE_REPORT.get("dense_only")  or {}),
        ("Hybrid RRF (ours)",         _BASELINE_REPORT.get("hybrid")      or {}),
    ]

    # Find best value per metric for highlighting
    metric_keys = [f"hit_rate", f"mrr", f"ndcg", f"hn_rate"]
    best: dict[str, float] = {}
    for mk in metric_keys:
        vals = [r.get(mk, 0.0) for _, r in rows if r.get(mk) is not None]
        if vals:
            best[mk] = max(vals) if mk != "hn_rate" else min(vals)

    def _cell(val: float | None, mk: str) -> str:
        if val is None:
            return "<td>—</td>"
        is_best = best.get(mk) is not None and abs(val - best[mk]) < 1e-9
        cls = ' class="best"' if is_best else ""
        return f"<td{cls}>{val:.4f}</td>"

    header = (
        f"<tr>"
        f"<th>System</th>"
        f"<th>HR@{top_k}</th>"
        f"<th>MRR@{top_k}</th>"
        f"<th>NDCG@{top_k}</th>"
        f"<th>HN_Rate@{top_k} ↓</th>"
        f"</tr>"
    )
    body_rows = []
    for name, r in rows:
        body_rows.append(
            f"<tr>"
            f'<td class="system-name">{name}</td>'
            + _cell(r.get("hit_rate"), "hit_rate")
            + _cell(r.get("mrr"),      "mrr")
            + _cell(r.get("ndcg"),     "ndcg")
            + _cell(r.get("hn_rate"),  "hn_rate")
            + "</tr>"
        )

    intro = (
        '<div class="metrics-intro">'
        f'<b>Gold standard:</b> 71 manually curated queries · 4 metrics · '
        f'20 queries with hard-negative annotations<br>'
        f'<b>Matching:</b> CELEX-level (document, not article). '
        f'Sub-chunks produced by hierarchical chunking match on the parent CELEX ID.<br>'
        f'<b>HN_Rate@{top_k}:</b> mean fraction of top-{top_k} slots occupied by '
        f'annotated hard-negative documents (lower is better).'
        f'</div>'
    )
    table = (
        f'<table class="metrics-table">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table>"
    )
    return intro + table


# ── Core query handler ────────────────────────────────────────────────────────

def handle_query(query: str, mode_choice: str):
    from src.models.schemas import FusedResult

    query = query.strip()
    if not query:
        empty = '<div class="placeholder">Ask a legal question to see results here.</div>'
        return empty, empty, ""

    t0 = time.perf_counter()

    use_generation = mode_choice == "Full RAG" and _generator is not None and _controller is not None

    if mode_choice in ("Full RAG", "Retrieval Only (Hybrid)") and _controller:
        fused   = _controller.fuse_results(query)
        results = fused
        mode_label = "Hybrid (BM25 + nomic-embed + RRF)"
    elif mode_choice == "Dense Only" and _dense:
        raw = _dense.retrieve(query, top_k=settings.top_k_fused)
        results = [
            FusedResult(
                article=r.article, rrf_score=r.score,
                rank=r.rank, sparse_rank=None, dense_rank=r.rank,
            )
            for r in raw
        ]
        fused = None
        mode_label = "Dense Only (nomic-embed-text-v1.5)"
    else:
        raw = _sparse.retrieve(query, top_k=settings.top_k_fused)
        results = [
            FusedResult(
                article=r.article, rrf_score=r.score,
                rank=r.rank, sparse_rank=r.rank, dense_rank=None,
            )
            for r in raw
        ]
        fused = None
        mode_label = "BM25 Sparse-Only"

    retrieval_ms  = (time.perf_counter() - t0) * 1000
    evidence_html = _render_evidence_cards(results)

    gen_ms = n_citations = None
    if use_generation and fused:
        t1          = time.perf_counter()
        output      = _generator.generate(query, fused)
        gen_ms      = (time.perf_counter() - t1) * 1000
        n_citations = len(output.cited_provisions)
        answer_html = _render_answer(output.answer)
    elif mode_choice == "Full RAG" and not (_generator and _controller):
        answer_html = (
            '<div class="no-llm">Full RAG unavailable: Ollama is not running or indices are missing.<br><br>'
            'Run <code>python main.py --mode index</code> then <code>ollama serve</code>.</div>'
        )
    else:
        answer_html = (
            '<div class="no-llm">Retrieval complete. '
            'Switch to <b>Full RAG</b> mode to generate a cited answer.</div>'
        )

    stats_html = _render_stats_bar(retrieval_ms, gen_ms, n_citations, mode_label)
    return answer_html, evidence_html, stats_html


# ── Compare handler (all three systems, same query) ───────────────────────────

def handle_compare(query: str):
    from src.models.schemas import FusedResult

    query = query.strip()
    if not query:
        ph = '<div class="placeholder">Enter a query above to compare all systems.</div>'
        return ph, ph, ph

    # BM25
    bm25_raw = _sparse.retrieve(query, top_k=settings.top_k_fused)
    bm25_results = [
        FusedResult(article=r.article, rrf_score=r.score,
                    rank=r.rank, sparse_rank=r.rank, dense_rank=None)
        for r in bm25_raw
    ]

    # Dense Only
    if _dense:
        dense_raw = _dense.retrieve(query, top_k=settings.top_k_fused)
        dense_results = [
            FusedResult(article=r.article, rrf_score=r.score,
                        rank=r.rank, sparse_rank=None, dense_rank=r.rank)
            for r in dense_raw
        ]
    else:
        dense_results = []

    # Hybrid
    hybrid_results = _controller.fuse_results(query) if _controller else []

    bm25_html   = _render_evidence_cards(bm25_results,   "BM25 Results")
    dense_html  = (
        _render_evidence_cards(dense_results, "Dense Results")
        if dense_results
        else '<div class="placeholder">Dense index not built. Run <code>python main.py --mode index</code>.</div>'
    )
    hybrid_html = (
        _render_evidence_cards(hybrid_results, "Hybrid RRF Results")
        if hybrid_results
        else '<div class="placeholder">Hybrid requires dense index.</div>'
    )
    return bm25_html, dense_html, hybrid_html


# ── Gradio layout ─────────────────────────────────────────────────────────────

_CORPUS_COUNTS = f"{len(_articles):,} chunks · 72 EU legislative acts"

with gr.Blocks(title="EU Climate Law — Legal RAG") as demo:

    gr.HTML(f"""
    <div class="hero">
      <h1>EU Climate Law — Legal RAG</h1>
      <p class="subtitle">
        Lexico-semantic fusion: BM25 lexical search + nomic-embed-text-v1.5 semantic search
        → Reciprocal Rank Fusion → LLM cited answer.
      </p>
      <span class="mode-badge" style="background:{_MODE_COLOR};">{_MODE}</span>
      <p class="corpus-stat">
        Corpus: <b>{_CORPUS_COUNTS}</b> &nbsp;·&nbsp;
        Hard-negative annotations: <b>{len(_HARD_NEG_CELEX)} CELEX IDs</b> flagged in results
      </p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Search ─────────────────────────────────────────────────────
        with gr.Tab("Search"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=3, min_width=280):
                    query_box = gr.Textbox(
                        label="Legal question",
                        placeholder="e.g. What are the monitoring obligations for maritime GHG emissions?",
                        lines=4, max_lines=8,
                    )
                    mode_radio = gr.Radio(
                        choices=_AVAILABLE_MODES,
                        value=_DEFAULT_MODE,
                        label="Mode",
                        info=(
                            "Full RAG: retrieval + LLM cited answer  |  "
                            "Retrieval Only: hybrid ranked list, no generation  |  "
                            "Dense Only: nomic-embed-text-v1.5 standalone  |  "
                            "Sparse Only: BM25 keyword search"
                        ),
                    )
                    ask_btn = gr.Button("Search & Answer", variant="primary", size="lg")

                    gr.HTML('<div class="sample-label">Example queries</div>')
                    for topic, q in _SAMPLES.items():
                        gr.Button(f"{topic} — {q[:52]}…", size="sm", variant="secondary").click(
                            fn=lambda text=q: text, outputs=query_box,
                        )

                    gr.HTML(f"""
                    <div style="margin-top:20px;padding:12px;background:#f8fafc;
                                border-radius:8px;font-size:0.75rem;color:#64748b;line-height:1.7;">
                      <b style="color:#1e293b;">System parameters</b><br>
                      BM25: k1={settings.bm25_k1}, b={settings.bm25_b}<br>
                      Dense: {settings.dense_model}<br>
                      RRF: k={settings.rrf_k}, top-{settings.top_k_retrieval} per retriever<br>
                      Returning top-{settings.top_k_fused} fused results<br>
                      Generator: {settings.llm_model}
                    </div>
                    """)

                with gr.Column(scale=7, min_width=500):
                    stats_out   = gr.HTML('<div class="stats-bar" style="visibility:hidden;">…</div>')
                    answer_out  = gr.HTML('<div class="placeholder">Ask a question to see the grounded answer here.</div>')
                    evidence_out= gr.HTML('<div class="placeholder">Retrieved legislative provisions will appear here.</div>')

            ask_btn.click(fn=handle_query, inputs=[query_box, mode_radio],
                          outputs=[answer_out, evidence_out, stats_out])
            query_box.submit(fn=handle_query, inputs=[query_box, mode_radio],
                             outputs=[answer_out, evidence_out, stats_out])

            gr.HTML("""
            <div style="margin-top:8px;display:flex;gap:16px;flex-wrap:wrap;
                        font-size:0.73rem;color:#64748b;align-items:center;">
              <b style="color:#374151;">Provenance legend:</b>
              <span style="background:#fef9c3;border:1px solid #fbbf24;color:#92400e;
                           padding:2px 8px;border-radius:10px;font-weight:600;">
                ● BM25 #n · Dense #n — found by both retrievers
              </span>
              <span style="background:#eff6ff;border:1px solid #93c5fd;color:#1e40af;
                           padding:2px 8px;border-radius:10px;font-weight:600;">
                ◆ Dense #n — semantic match only
              </span>
              <span style="background:#f0fdf4;border:1px solid #86efac;color:#14532d;
                           padding:2px 8px;border-radius:10px;font-weight:600;">
                ◈ BM25 #n — keyword match only
              </span>
              <span style="background:#fee2e2;border:1px solid #f87171;color:#991b1b;
                           padding:2px 8px;border-radius:10px;font-weight:600;">
                ⚠ known hard negative — retrieved but annotated as legally incorrect
              </span>
            </div>
            """)

        # ── Tab 2: Compare Systems ────────────────────────────────────────────
        with gr.Tab("Compare Systems"):
            gr.HTML("""
            <div style="padding:12px 0 4px;font-size:0.85rem;color:#475569;line-height:1.6;">
              Run the same query through <b>BM25</b>, <b>Dense Only</b>, and <b>Hybrid RRF</b>
              simultaneously. This is the core ablation the paper is built on.
              CELEX IDs link to EUR-Lex. Red cards are annotated hard negatives.
            </div>
            """)
            with gr.Row():
                cmp_query = gr.Textbox(
                    label="Query",
                    placeholder="e.g. What are the Phase 4 auctioning rules for EU ETS?",
                    lines=2, scale=8,
                )
                cmp_btn = gr.Button("Compare", variant="primary", scale=1, min_width=100)

            gr.HTML('<div class="sample-label">Quick examples</div>')
            with gr.Row():
                for topic, q in list(_SAMPLES.items())[:4]:
                    gr.Button(f"{topic}", size="sm", variant="secondary").click(
                        fn=lambda text=q: text, outputs=cmp_query,
                    )
            with gr.Row():
                for topic, q in list(_SAMPLES.items())[4:]:
                    gr.Button(f"{topic}", size="sm", variant="secondary").click(
                        fn=lambda text=q: text, outputs=cmp_query,
                    )

            with gr.Row(equal_height=False):
                with gr.Column():
                    gr.HTML('<div class="compare-col-header">◈ BM25 Sparse-Only<br><small style="font-weight:400;font-size:0.7rem;">HR@5 = 0.7042 · MRR@5 = 0.5556</small></div>')
                    bm25_col = gr.HTML('<div class="placeholder">Run a query to see BM25 results.</div>')
                with gr.Column():
                    gr.HTML('<div class="compare-col-header">◆ Dense Only (nomic-embed-text-v1.5)<br><small style="font-weight:400;font-size:0.7rem;">HR@5 = 0.9577 · MRR@5 = 0.7854</small></div>')
                    dense_col = gr.HTML('<div class="placeholder">Run a query to see dense results.</div>')
                with gr.Column():
                    gr.HTML('<div class="compare-col-header">● Hybrid RRF (BM25 + Dense)<br><small style="font-weight:400;font-size:0.7rem;">HR@5 = 0.8592 · MRR@5 = 0.7162</small></div>')
                    hybrid_col = gr.HTML('<div class="placeholder">Run a query to see hybrid results.</div>')

            cmp_btn.click(fn=handle_compare, inputs=[cmp_query],
                          outputs=[bm25_col, dense_col, hybrid_col])
            cmp_query.submit(fn=handle_compare, inputs=[cmp_query],
                             outputs=[bm25_col, dense_col, hybrid_col])

        # ── Tab 3: Evaluation Metrics ─────────────────────────────────────────
        with gr.Tab("Evaluation"):
            gr.HTML("""
            <div style="padding:12px 0 4px;font-size:0.85rem;color:#475569;">
              Quantitative results from the last <code>python main.py --mode baselines</code> run.
              Green values are the best in each column. HN_Rate@5 is lower-is-better (↓).
            </div>
            """)
            gr.HTML(_render_metrics_html())

            gr.HTML("""
            <div style="margin-top:24px;padding:16px;background:#f8fafc;
                        border-radius:8px;border:1px solid #e2e8f0;
                        font-size:0.8rem;color:#475569;line-height:1.7;">
              <b style="color:#1e293b;">Key finding</b><br>
              Dense-only outperforms Hybrid RRF on all metrics. nomic-embed-text-v1.5's
              8 192-token context window already captures the lexical signal in long EU legislative
              articles, so adding BM25 via RRF introduces noise rather than complementary signal.
              Hybrid still dominates BM25-only by +22 pp HR@5, confirming that lexical search
              alone is insufficient for semantic legal retrieval.<br><br>
              <b style="color:#1e293b;">Chunking</b><br>
              1 156 articles → 1 166 chunks after hierarchical splitting of 5 oversized
              Article 1 amending instruments at EU legal paragraph boundaries (the "; (N)"
              amendment-point separators).<br><br>
              <b style="color:#1e293b;">Hard negatives</b><br>
              20 of 71 gold-standard queries carry annotated hard-negative CELEX IDs —
              documents that share surface vocabulary with the correct answer but are
              legally incorrect. HN_Rate@5 measures how often these appear in top-5 results.
            </div>
            """)


if __name__ == "__main__":
    _theme = gr.themes.Base(
        font=gr.themes.GoogleFont("Inter"),
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
    )
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=_theme,
        css=_CSS,
    )
