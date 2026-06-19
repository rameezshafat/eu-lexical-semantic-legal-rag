"""
app.py - Demo UI for the Lexico-Semantic Fusion Legal RAG system.

Run:
    pip install gradio==6.15.2
    python app.py

Graceful degradation by capability:
  Full RAG       - hybrid retrieval + LLM generation   (Ollama running with llama3.3:70b)
  Retrieval-only - hybrid BM25 + BGE-M3, no LLM        (dense index exists, Ollama not running)
  Sparse-only    - BM25 only, fully offline             (no index needed)
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from config import settings
from src.ingestion.loader import CorpusLoader
from src.retrieval.sparse import SparseRetriever

# ── Capability detection ──────────────────────────────────────────────────────

_INDEX_DIR = Path(settings.index_dir)
_HAS_DENSE = (_INDEX_DIR / "dense.faiss").exists()

# ── Startup: load all available components ────────────────────────────────────

print("Initialising demo…")
_loader   = CorpusLoader(settings.corpus_path)
_articles = _loader.load()
print(f"  {len(_articles)} articles loaded")

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
    print("  Dense (BGE-M3) index + RankFusionController loaded")

try:
    from src.generation.generator import LegalGenerator
    _generator = LegalGenerator(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        base_url=settings.llm_ollama_base_url,
    )
    print("  LegalGenerator (Ollama) ready")
except Exception as _exc:
    print(f"  LegalGenerator unavailable: {_exc}")

if _generator and _controller:
    _MODE, _MODE_COLOR = "Full RAG", "#1d4ed8"
elif _controller:
    _MODE, _MODE_COLOR = "Retrieval-Only", "#0369a1"
else:
    _MODE, _MODE_COLOR = "Sparse-Only (BM25)", "#065f46"

print(f"\nDemo mode: {_MODE}\n")

# ── Sample queries (curated, domain-grouped) ──────────────────────────────────

_SAMPLES = {
    "ETS": "How are emission allowances allocated to industrial installations in EU ETS Phase 4?",
    "CBAM": "How does CBAM calculate the embedded emissions in imported steel and cement?",
    "Taxonomy": "What criteria must an economic activity meet to be environmentally sustainable under the Taxonomy Regulation?",
    "Maritime": "What are the monitoring and reporting obligations for greenhouse gas emissions from ships?",
    "LULUCF": "What land use accounting rules apply to member states under the LULUCF Regulation?",
    "Climate Law": "What is the EU's legally binding 2050 climate neutrality target and governance framework?",
    "F-Gas": "What are the reporting requirements for fluorinated greenhouse gases?",
    "ESR": "What are the binding annual emission targets for member states under the Effort Sharing Regulation?",
}

# ── HTML rendering helpers ────────────────────────────────────────────────────

_CSS = """
/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── Page chrome ── */
.hero { padding: 20px 0 8px; }
.hero h1 { font-size: 1.6rem; font-weight: 700; margin: 0 0 4px;
           color: #0f172a; letter-spacing: -0.02em; }
.hero .subtitle { font-size: 0.9rem; color: #475569; margin: 0 0 10px; }

/* ── Dark-mode overrides (Gradio adds .dark to the page root) ──
   Custom cards keep light backgrounds in both themes, so text inside
   them must stay dark; page-level chrome must flip to light text. */
.dark .hero h1 { color: #f1f5f9; }
.dark .hero .subtitle { color: #94a3b8; }
.dark .corpus-stat { color: #94a3b8; }
.dark .corpus-stat b { color: #e2e8f0; }
.dark .sample-label, .dark .query-label { color: #94a3b8; }
.dark .evidence-header { color: #cbd5e1; }
.mode-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
    font-weight: 600; color: white; margin-bottom: 16px;
}
.corpus-stat { font-size: 0.8rem; color: #64748b; margin-top: 4px; }
.corpus-stat b { color: #1e293b; }

/* ── Query input area ── */
.query-label { font-size: 0.8rem; font-weight: 600; color: #374151;
               text-transform: uppercase; letter-spacing: 0.05em;
               margin-bottom: 6px; }
.sample-label { font-size: 0.75rem; font-weight: 600; color: #6b7280;
                text-transform: uppercase; letter-spacing: 0.05em;
                margin: 12px 0 6px; }

/* ── Stats bar ── */
.stats-bar {
    display: flex; gap: 16px; padding: 8px 14px;
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; margin-bottom: 12px;
    font-size: 0.78rem; color: #475569 !important; flex-wrap: wrap;
}
.stats-bar .stat { display: flex; align-items: center; gap: 5px; color: #475569 !important; }
.stats-bar .stat-val { font-weight: 700; color: #0f172a !important; }
.stats-bar .sep { color: #cbd5e1 !important; }

/* ── Answer box ── */
/* !important: the box keeps its light background in dark mode, where
   Gradio's theme would otherwise force near-white text onto it. */
.answer-box {
    border-left: 3px solid #2563eb;
    padding: 14px 18px;
    background: #f0f6ff;
    border-radius: 0 8px 8px 0;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #1e293b !important;
    margin-bottom: 20px;
}
.answer-box p { margin: 0 0 10px; color: #1e293b !important; }
.answer-box p:last-child { margin: 0; }

/* ── Evidence section ── */
.evidence-header {
    font-size: 0.8rem; font-weight: 700; color: #374151;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 4px 0 10px; display: flex; align-items: center; gap: 8px;
}
.evidence-header .ev-count {
    background: #e2e8f0; color: #475569;
    padding: 1px 8px; border-radius: 10px; font-size: 0.72rem;
}

/* ── Evidence card ── */
.ev-card {
    border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 12px 14px; margin-bottom: 10px;
    background: #ffffff;
    transition: box-shadow 0.15s;
}
.ev-card:hover { box-shadow: 0 2px 10px rgba(0,0,0,.07); }

.ev-card-top {
    display: flex; align-items: flex-start;
    justify-content: space-between; gap: 10px; margin-bottom: 8px;
    flex-wrap: wrap;
}
.ev-rank {
    font-size: 0.75rem; font-weight: 700; color: #94a3b8;
    min-width: 22px; padding-top: 1px;
}
.ev-meta { flex: 1; }
.ev-celex { font-size: 0.82rem; font-weight: 700; color: #0f172a; }
.ev-article { font-size: 0.78rem; color: #475569; margin-top: 1px; }

/* provenance badges */
.prov-both  { background:#fef9c3; border:1px solid #fbbf24; color:#92400e; }
.prov-dense { background:#eff6ff; border:1px solid #93c5fd; color:#1e40af; }
.prov-sparse{ background:#f0fdf4; border:1px solid #86efac; color:#14532d; }
.prov-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600;
}
.prov-icon { font-size: 0.65rem; }

.ev-scores {
    display: flex; gap: 8px; align-items: center;
    font-size: 0.7rem; color: #94a3b8; flex-wrap: wrap;
}
.ev-rrf { font-weight: 600; color: #6366f1; }

.ev-text {
    font-size: 0.83rem; line-height: 1.6; color: #374151;
    border-top: 1px solid #f1f5f9; padding-top: 8px;
    margin-top: 4px;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ── No-result / loading states ── */
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
.no-llm code { background:#e2e8f0; padding:1px 5px; border-radius:4px;
               font-size:0.8rem; }
"""


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


def _render_evidence_cards(results: list) -> str:
    if not results:
        return '<div class="placeholder">No results retrieved.</div>'

    cards = []
    for r in results:
        a           = r.article
        badge       = _provenance_badge(
            getattr(r, "dense_rank", None),
            getattr(r, "sparse_rank", None),
        )
        score_attr  = "rrf_score" if hasattr(r, "rrf_score") else "score"
        score_val   = getattr(r, score_attr, 0.0)
        doc_type_bg = "#dbeafe" if a.doc_type == "Directive" else "#dcfce7"
        doc_type_co = "#1e40af" if a.doc_type == "Directive" else "#14532d"

        cards.append(f"""
<div class="ev-card">
  <div class="ev-card-top">
    <div class="ev-rank">#{r.rank}</div>
    <div class="ev-meta">
      <div class="ev-celex">{a.celex_id}</div>
      <div class="ev-article">
        {a.article_number}
        &nbsp;
        <span style="background:{doc_type_bg};color:{doc_type_co};
                     padding:1px 6px;border-radius:8px;font-size:0.67rem;
                     font-weight:600;">{a.doc_type}</span>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px;">
      {badge}
      <span class="ev-rrf">{score_val:.5f}</span>
    </div>
  </div>
  <div class="ev-text">{a.article_text[:400]}{"…" if len(a.article_text) > 400 else ""}</div>
</div>""")

    return (
        f'<div class="evidence-header">'
        f'Retrieved Evidence <span class="ev-count">{len(results)}</span>'
        f'</div>'
        + "".join(cards)
    )


def _render_stats_bar(
    retrieval_ms: float,
    gen_ms: float | None,
    n_citations: int | None,
    mode: str,
) -> str:
    parts = [
        f'<div class="stat">Retrieval <span class="stat-val">{retrieval_ms:.0f} ms</span></div>',
        f'<span class="sep">|</span>',
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


# ── Core query handler ────────────────────────────────────────────────────────

def handle_query(query: str):
    """Execute retrieval (and optionally generation), return HTML for all panels."""
    query = query.strip()
    if not query:
        empty = '<div class="placeholder">Ask a legal question to see results here.</div>'
        return empty, empty, ""

    t0 = time.perf_counter()

    # Retrieval
    if _controller:
        fused   = _controller.fuse_results(query)
        results = fused
        mode    = "Hybrid (BM25 + BGE-M3 + RRF)"
    else:
        raw     = _sparse.retrieve(query, top_k=settings.top_k_fused)
        # Wrap as FusedResult-compatible objects for uniform rendering
        from src.models.schemas import FusedResult
        results = [
            FusedResult(
                article=r.article,
                rrf_score=r.score,
                rank=r.rank,
                sparse_rank=r.rank,
                dense_rank=None,
            )
            for r in raw
        ]
        fused = None
        mode  = "BM25 Sparse-Only"

    retrieval_ms = (time.perf_counter() - t0) * 1000
    evidence_html = _render_evidence_cards(results)

    # Generation
    gen_ms      = None
    n_citations = None
    if _generator and fused:
        t1 = time.perf_counter()
        output      = _generator.generate(query, fused)
        gen_ms      = (time.perf_counter() - t1) * 1000
        n_citations = len(output.cited_provisions)
        answer_html = _render_answer(output.answer)
    else:
        msg = (
            "LLM generation is not active in this demo instance.<br><br>"
            "To enable: build indices with "
            "<code>python main.py --mode index</code>, then start Ollama with "
            "<code>ollama serve</code> and pull <code>llama3.3:70b</code>."
        )
        answer_html = f'<div class="no-llm">{msg}</div>'

    stats_html = _render_stats_bar(retrieval_ms, gen_ms, n_citations, mode)

    return answer_html, evidence_html, stats_html


# ── Gradio layout ─────────────────────────────────────────────────────────────

_CORPUS_COUNTS = f"{len(_articles):,} articles · 72 EU legislative acts"
_PIPELINE_DESC = (
    "BM25 lexical search &nbsp;+&nbsp; BGE-M3 semantic search "
    "&nbsp;→&nbsp; Reciprocal Rank Fusion &nbsp;→&nbsp; LLM (Ollama cited answer)"
)

with gr.Blocks(title="EU Climate Law - Legal RAG") as demo:

    # ── Hero ─────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="hero">
      <h1>EU Climate Law - Legal RAG</h1>
      <p class="subtitle">
        Ask questions in plain English. Get answers grounded in EU legislation,
        with every claim cited to a specific article.
      </p>
      <span class="mode-badge" style="background:{_MODE_COLOR};">{_MODE}</span>
      <p class="corpus-stat">
        Corpus: <b>{_CORPUS_COUNTS}</b> &nbsp;·&nbsp;
        {_PIPELINE_DESC}
      </p>
    </div>
    """)

    # ── Main layout: left query panel + right results panel ──────────────────
    with gr.Row(equal_height=False):

        # Left: query + examples
        with gr.Column(scale=3, min_width=300):
            query_box = gr.Textbox(
                label="Your legal question",
                placeholder=(
                    "e.g. What are the monitoring obligations for maritime "
                    "GHG emissions under EU law?"
                ),
                lines=4,
                max_lines=8,
            )
            ask_btn = gr.Button("Search & Answer", variant="primary", size="lg")

            gr.HTML('<div class="sample-label">Example queries</div>')
            for topic, q in _SAMPLES.items():
                gr.Button(f"{topic} - {q[:55]}…", size="sm", variant="secondary").click(
                    fn=lambda text=q: text,
                    outputs=query_box,
                )

            gr.HTML(f"""
            <div style="margin-top:20px; padding:12px; background:#f8fafc;
                        border-radius:8px; font-size:0.75rem; color:#64748b;
                        line-height:1.6;">
              <b style="color:#1e293b;">System</b><br>
              Retriever: BM25 k1={settings.bm25_k1}, b={settings.bm25_b}<br>
              Dense: {settings.dense_model}<br>
              RRF k={settings.rrf_k} · top-{settings.top_k_retrieval} per retriever<br>
              Returning top-{settings.top_k_fused} fused results<br>
              Generator: {settings.llm_model}
            </div>
            """)

        # Right: stats + answer + evidence
        with gr.Column(scale=7, min_width=500):
            stats_box    = gr.HTML(
                '<div class="stats-bar" style="visibility:hidden;">placeholder</div>'
            )
            answer_box   = gr.HTML(
                '<div class="placeholder">Ask a question to see the grounded answer here.</div>'
            )
            evidence_box = gr.HTML(
                '<div class="placeholder">Retrieved legislative provisions will appear here.</div>'
            )

    # ── Wire interactions ─────────────────────────────────────────────────────
    ask_btn.click(
        fn=handle_query,
        inputs=[query_box],
        outputs=[answer_box, evidence_box, stats_box],
    )
    query_box.submit(
        fn=handle_query,
        inputs=[query_box],
        outputs=[answer_box, evidence_box, stats_box],
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="margin-top:8px; display:flex; gap:16px; flex-wrap:wrap;
                font-size:0.73rem; color:#64748b; align-items:center;">
      <b style="color:#374151;">Provenance legend:</b>
      <span style="background:#fef9c3;border:1px solid #fbbf24;color:#92400e;
                   padding:2px 8px;border-radius:10px;font-weight:600;">
        ● BM25 #n · Dense #n - found by both retrievers (strongest signal)
      </span>
      <span style="background:#eff6ff;border:1px solid #93c5fd;color:#1e40af;
                   padding:2px 8px;border-radius:10px;font-weight:600;">
        ◆ Dense #n - semantic match only
      </span>
      <span style="background:#f0fdf4;border:1px solid #86efac;color:#14532d;
                   padding:2px 8px;border-radius:10px;font-weight:600;">
        ◈ BM25 #n - keyword match only
      </span>
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
