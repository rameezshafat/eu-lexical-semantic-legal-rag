# CFP Alignment Memo — "Evaluating Hybrid Sparse-Dense Retrieval for EU Climate Legal Documents"

Internal memo. Not for submission. Written at the end of the `paper-hardening` pass
(branch `paper-hardening`, off `feature/ananya`) to record venue fit and what is still
missing before each submission. Supersedes the venue-status notes that motivated this
branch — several "still missing" items below are now closed.

---

## NLLP Workshop (Natural Legal Language Processing)

**CFP asks for:** real-world legal NLP, resource papers, system descriptions, empirical
evaluations, honest failure analysis.

**How this paper qualifies:**
- A manually constructed, CELEX-level gold standard (102 queries; sealed 53-query test
  set) grounded in real legislative research tasks and MEP-written parliamentary
  questions, not synthetic queries.
- A rigorous three-way empirical comparison (BM25 / dense / hybrid RRF) with bootstrap
  95% CIs on all three metrics and Wilcoxon signed-rank significance testing.
- An honest null result: hybrid does not significantly outperform dense alone
  (p=0.614), reported plainly rather than reframed as a win. NLLP reviewers are legal
  NLP specialists who read this kind of transparency as a strength, not a weakness.
- A root-cause failure table (Table III) built from actually inspecting the corpus
  index rather than speculating — e.g. confirming that all five hybrid failures have
  their gold instrument indexed, and tracing two of them to amendment-diff chunking
  and one to a genuine lexical/semantic register gap.

**Still missing:**
- Cohen's kappa from a second, independent annotator. The 14-query overlap subset,
  annotator instructions, `iaa_annotator1.json`, and `scripts/compute_iaa.py` are all
  ready; a second human pass has not come back. This is disclosed in Threats to
  Validity rather than papered over with an invented number.
- (Resolved this pass: Table II's per-query crosstab and the "instruments outside the
  corpus" hard-failure explanation were both wrong in the prior draft and are now
  verified against `test_report.json` and `sparse_article_map.pkl`.)

---

## RegNLP Workshop (Regulatory and Legal NLP)

**CFP asks for:** statutory/regulatory parsing, cross-instrument referencing, climate
and environmental compliance applications.

**How this paper qualifies:**
- A documented Formex XML pipeline (`notebooks/cellar_etl.ipynb`): the real
  ACT/ARTICLE/TI.ART/ALINEA element hierarchy, the leaf-vs-nested ALINEA
  deduplication logic, and the HTML fallback path, now described in the paper with the
  actual element names rather than a generic "we parsed XML" gloss.
- EuroVoc-scoped corpus construction with named descriptors (climate change, emission
  trading, renewable energy, energy efficiency, etc.) rather than an opaque concept-ID
  list.
- Test queries derived from actual Commission answers to MEP questions on compliance
  topics (CSDDD due diligence, ETS2 opt-in, Taxonomy disclosure vs. alignment
  obligations), independently corroborating the relevance judgements.
- The amendment-diff chunking finding (q080/q083) is directly a statutory-parsing
  contribution: it identifies *why* amending-instrument text is hard to retrieve
  against natural-language compliance questions, which is exactly RegNLP's territory.

**Still missing:**
- Generation-layer evaluation (`scripts/eval_generation.py`) has not been run; it
  requires a local Ollama instance and is out of scope for this retrieval-hardening
  pass. No citation-accuracy metric is reported. If RegNLP is the target, budget a
  separate pass for the generation layer before submission.

---

## JURIX (short paper track)

**CFP asks for:** AI and law, knowledge-based systems, formal and empirical methods in
legal informatics, European civil law systems.

**How this paper qualifies:**
- Article-level CELEX indexing with explicit provenance: every retrieved passage is
  citable as `[CELEX_ID — Article N]`, and evaluation is deliberately CELEX-level
  (matching the legal-research task of finding the right law, not the right
  paragraph) — a point JURIX reviewers ask about explicitly and which is now stated
  directly in Corpus Construction rather than left implicit.
- The Formex parsing methodology (the specific element hierarchy, the amendment-point
  splitting strategy, the corpus statistics) is now concrete enough to satisfy a
  reviewer checking "did they actually engineer this corpus or just download a
  dataset."
- A sealed train/val/test protocol with statistical rigor (bootstrap CIs, Wilcoxon
  tests, effect sizes) that meets JURIX's empirical-methods bar.

**Still missing:**
- Nothing structural for this pass. The main remaining risk for JURIX specifically is
  the same IAA gap flagged for NLLP — JURIX reviewers with an annotation background
  may ask the same question.

---

## ECIR Reproducibility Track (fallback)

**CFP asks for:** reproducible systems, parameter sensitivity analysis, empirical
comparisons with accessible code and data.

**How this paper qualifies:**
- A full RRF parameter sensitivity analysis: a clean validation-set sweep showing an
  unambiguous peak at the deployed `dense_weight=5`, plus an 84-combination
  `k x dense_weight x top_k_retrieval` grid re-run on the sealed test set purely as a
  post-hoc robustness check (explicitly not used to re-select parameters). This is
  exactly the kind of sensitivity reporting the track asks for, including the honest
  admission that the deployed configuration is not the test-set optimum.
- Pinned dependency versions (Hardware and Software subsection) and an open,
  documented codebase.
- A sealed test-set protocol described precisely enough to reproduce (tune on
  49-query val, evaluate once on 53-query test).

**Still missing:**
- Public data release of `data/evaluation/gold_standard_test.json` and a formal
  reproducibility checklist (artifact evaluation appendix). Both are mechanical to
  produce from what already exists but have not been done in this pass.

---

## Cross-cutting items affecting all four venues

- **Second IAA annotator.** Cannot be simulated — the entire point of Cohen's kappa
  here is to catch a gold standard that was written to look artificially consistent.
  This is a real external dependency, not a code or writing gap. Blocking for NLLP and
  JURIX in particular.
- **E5-large-v2 baseline.** `scripts/e5_baseline.py` had a broken import
  (`ArticleLoader` does not exist; fixed to `CorpusLoader`) and had never been run.
  Fixed and executed as part of this pass. The result was not either of the two
  outcomes the task brief anticipated ("similar to nomic" or "worse due to
  truncation"): E5-large-v2 ties nomic-embed-text-v1.5 exactly on HR@5 (0.925) but on
  a different 4-query subset, and clearly beats it on MRR@5 (0.806 vs. 0.743) and
  NDCG@5 (0.794 vs. 0.709), despite a 512-token context window vs. nomic's 8,192 and
  no hierarchical sub-chunking. This undercuts the paper's stated long-context
  rationale for choosing nomic specifically (the pure-bi-encoder and open-weight
  rationale still holds) and is now reported plainly as a new subsection
  (Section V-D) rather than smoothed over. Any reviewer at any of the four venues who
  reads the "why nomic" methodology paragraph closely enough to ask "did you check
  this" now has an answer in the paper itself, which is a strictly better position
  than not having run the ablation at all.
- **Generation evaluation.** Out of scope for every venue in this pass except as a
  "still missing" item for RegNLP specifically, since NLLP/JURIX/ECIR framing here is
  retrieval-only.
