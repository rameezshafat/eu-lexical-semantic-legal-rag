# PR Memo: `strong-accept-pass` (from `v1-weak-accept-draft`)

Internal memo for the human authors before this goes to a supervisor or
reviewer. Not for submission.

## What changed, and why

**Added a third embedding model (BGE-small-v1.5) to triangulate the E5
finding along the capacity axis**, not just context window. This was the
task: same corpus, same eval script, one more row.

**That work surfaced a real error in the weak-accept draft.** Investigating
why BGE alone resolved q080 led to re-checking the "amendment-diff chunking
dilution" root cause the previous pass had written for that query. It was
wrong. `sparse_article_map.pkl` contains clean, well-titled standalone
articles for the ETS2 chapter (Article 30a–30m, including Article 30k,
"Postponement of emissions trading for buildings, road transport..."), not
just messy amendment-diff sub-chunks. Checking the actual retrieval rankings
directly showed BM25's best rank for any of the instrument's 43 chunks is
13, and nomic ranks none in its top 15 — both instead rank thematically
adjacent decoy instruments (Social Climate Fund, Energy Efficiency Directive
recast, CBAM) higher. This is a cross-instrument discrimination failure, not
a text-quality problem, and re-chunking would not have fixed it. Table III,
the Discussion, and the future-work list were all corrected. q083 was
separately re-verified against the actual raw Formex XML (re-fetched
directly from Cellar since the local cache was incomplete): confirmed
"fishing" and "inland navigation" appear nowhere in the document and
"vessel" appears only in recitals, which are out of scope for article-level
indexing by design.

**A second, more serious error was caught late, while drafting this memo.**
The abstract, intro, ablation section, and conclusion all described E5 and
BGE as "beating" or "exceeding" nomic on various metrics, based on point
estimates alone. Running the Wilcoxon test that should have accompanied
those claims from the start showed none of the six comparisons (each model
vs. nomic, on HR@5/MRR@5/NDCG@5) are significant at n=53 (p ranges from
0.055 to 1.0). This is now fixed throughout: the paper reports statistical
parity between all three dense models, not a ranking among them. The
underlying finding survives and is arguably cleaner as a result — "a
512-token model is not distinguishably worse than an 8,192-token one" is a
real, useful, and more defensible claim than "the smaller model wins."

**Other changes:** added an External Validity subsection (citation-based,
using published BM25-vs-dense results already in Related Work — explicitly
not a new benchmark run, per the deliberate scope decision to skip Phase 4's
full external-benchmark option); rewrote the intro's RQs and contributions
list and reordered the abstract to lead with the embedding-model finding;
added the 58.5% MEP-corroboration figure as a non-LLM triangulation angle
for the still-open IAA gap; wrote `REPRODUCE.md`.

**Deliberately not done**, per explicit scope decisions made before this
pass started:

- No LLM-generated second annotator. The IAA gap is still open. This was a
  direct choice, not an oversight — see below.
- No re-chunking / consolidated-instrument-text experiment. Initially
  planned as a scoped proof-of-concept, but investigating q080 showed the
  premise was wrong (clean text already existed and still lost to decoys),
  so the experiment would have tested the wrong hypothesis. The
  investigation that replaced it produced a better-evidenced result than
  the planned experiment would have.
- No external benchmark run. Chosen deliberately over the citation-paragraph
  alternative to avoid an apples-to-oranges comparison a sharp reviewer
  would flag as weak methodology.

**A later, explicit request expanded the paper from 6-7 to 9 pages.** This
was done with real content, not padding: three vector figures generated
from data already in the repository (`scripts/make_figures.py` — a
token-length histogram, a per-difficulty grouped bar chart, and an RRF
sensitivity heatmap that visually confirms the "broad ridge, not narrow
spike" claim already in the text); a new Results-by-Query-Difficulty
subsection built by joining `test_report.json` against the gold standard's
existing `difficulty` tags (no new experiment, just a join that had not
been done before — it shows the hybrid's coverage deficit vs. dense is
concentrated entirely in the 18-query hard bucket); a worked qualitative
example (q006) showing the hybrid's actual fused output with per-retriever
provenance ranks, including a decoy from a different instrument that dense
ranked highly; a Related Work subsection on embedding-model selection that
incidentally fixed two orphaned citations ([9], [11]) that were in the
bibliography but never cited in the text; and corpus size-distribution
statistics. Every number added in this pass was verified against the
underlying data before being written down, same as everything else in this
memo — none of it is filler inserted to hit a page count.

## Load-bearing claims (what the paper's headline arguments actually stand on)

1. **Dense beats BM25, hybrid ties dense.** Rests on `test_report.json` +
   `significance_results.json`, reproduced byte-for-byte multiple times in
   this session. Solid.
2. **Context-window length does not predict retrieval quality on this
   corpus.** Rests on the six-comparison Wilcoxon suite in
   `ablation_significance_results.json` — all non-significant. This is a
   null-result claim (failure to show a difference), which is weaker in
   kind than a positive claim. See objection #2 below.
3. **q080 is a cross-instrument discrimination failure, verified by direct
   inspection of retrieval rankings, not the originally-diagnosed chunking
   problem.** This is a single-query, directly-verified factual claim, not
   a statistical one. It does not need — and is not weakened by — small
   sample size. It is the strongest individual piece of evidence in the
   paper precisely because it is not aggregate.
4. **58.5% of the test set has independent corroboration via Commission
   answers.** A verified count (`31/53`), not an inference. Solid, but
   narrow: it corroborates the *relevance judgments*, not inter-annotator
   *agreement* — it is a partial mitigation for the IAA gap, not a
   substitute for it, and the paper should not be read as implying otherwise.

## Strongest remaining objections (self-assessment, not modesty)

**1. The IAA gap is still open, and it is the objection every reviewer will
raise first.** The paper's own Threats to Validity section says relevance
judgments come from a single annotator and a second pass "had not been
returned at the time of writing." A reviewer will ask: is this benchmark
just one person's opinion? We chose not to paper over this with an
LLM-generated stand-in, because using an LLM to judge against a gold
standard that an LLM-assisted process helped produce risks circular,
inflated agreement — exactly the failure mode kappa exists to catch. That
is the right call scientifically, but it does not make the gap go away. If
a second human annotator does not come back before submission, this paper
should be pitched as "resource + rigorous null result," not oversold on
benchmark validity, and reviewers should be told the IAA protocol is ready
but incomplete, not that it is done.

**2. Several of this pass's own new claims are underpowered, not just
non-significant.** n=53 is small. "None of the six ablation comparisons
reached significance" is honestly reported, but a hostile reviewer could
reasonably say this shows the study lacks the power to distinguish these
models, not that the models are actually equivalent — failing to reject the
null is not the same as confirming it. We believe the framing in the paper
("statistical parity," "not distinguishably worse," explicit p-values
throughout) is careful about this distinction, but the paper's rhetorical
energy (leading the abstract with this finding, promoting it to a
first-class contribution) is arguably still doing more work than a
non-significant result, properly understood, can support. This is the
single most likely place a statistically literate reviewer pushes back, and
we do not have a good answer beyond "a larger test set would settle it,"
which is already in the paper.

**3. Two real errors were caught and fixed in this pass alone** (the q080
root cause, and the ablation significance overclaiming). Both were caught
before submission, which is the system working as intended — but a
reasonable reviewer, or a skeptical co-author, should ask how many more
exist that were not caught. We do not have a clean answer to this beyond
"we checked harder the second time." Every remaining specific factual claim
in the paper (article counts, CELEX presence, retrieval rankings) has been
directly verified against the actual data at least once in this session,
but we have not had a second, independent pair of eyes re-verify the
verification.

**4. q083's "content absent from source" conclusion depends on a keyword
search of the raw XML being exhaustive.** We searched for "fishing,"
"vessel," and "inland navigation" specifically, because those are the terms
the query and gold-standard notes use. EU legislative drafting sometimes
uses different terminology for the same concept (e.g., referring to a
vessel category by IMO classification or Annex reference rather than the
word "vessel"). We are confident in the negative result for those specific
terms, less confident that we have ruled out every possible phrasing of the
same underlying provision.

**5. The external validity paragraph is citation-based, not empirical**, by
deliberate choice (see scope decisions above). This is a legitimate and
disclosed choice, but it means the paper still has zero results generated
on any corpus other than its own. If a reviewer wants to see the pipeline
run on external data, this paragraph will not satisfy them, and we should
not present it as if it does more than triangulate against prior published
numbers.

## Recommendation

This is a stronger paper than the weak-accept draft: it corrects two errors
the previous pass introduced or missed, adds a second experimental
dimension (embedding-model capacity, not just context window), and is more
honest about statistical power throughout, including in places where that
honesty costs the paper a punchier claim. It is not yet at a point where we
would be surprised by anything less than accept — objection #1 (IAA) is
still capable of sinking it with the wrong reviewer, and objection #2 is a
real, not cosmetic, limitation on the paper's most-promoted new finding.
Before submission: chase the second annotator harder than anything else in
this list, since it is the one objection that is fully within the authors'
power to close.
