# Logic/Circularity Pass — Changelog

Scope: audit the argumentative chain (evidence → claim → conclusion) in
`docs/paper/eu_climate_hybrid_retrieval.tex` for circularity, missing
inferential steps, and confidence/support mismatches. No new experiments,
no numbers/citations changed, no style pass. All edits were made in a copy,
`docs/paper/eu_climate_hybrid_retrieval_logic_pass.tex`; the original file
was not touched. This document is for human review, not part of the paper
itself. Full diff: `docs/logic_pass.diff`.

Six changes, in the order they appear in the paper:

1. **Abstract — "match" → "show no significant difference."**
   The abstract said the two alternative embedding models "match" nomic's
   retrieval quality, citing only non-significant Wilcoxon tests as
   support. Non-significance means "no difference detected," not
   "confirmed equivalence" — those are different claims, and "match"
   asserts the stronger one. Reworded to "show no significant difference
   in retrieval quality."

2. **Introduction, contribution (iii) — same fix, for consistency.**
   The contributions list independently described the same ablation
   result as the two alternatives "statistically match" nomic. Same
   conflation as #1, in a different sentence the first pass missed.
   Reworded to "show no significant difference from," so the finding
   isn't asserted at two different confidence levels in the same paper.

3. **Discussion, §VI — grounded the "BM25 has no unique hits" claim.**
   The claim that BM25 contributes no hits that dense misses was stated
   as a bare assertion. Added a citation to `Table~\ref{tab:reliability}`
   and the exact figure (0 queries) so the claim points to the evidence
   it's read from instead of standing on its own.

4. **Discussion, §VI — grounded the "dense is already at rank 1" claim.**
   This claim was doing explanatory work (explaining why BM25's rank
   signal adds nothing on top of dense) but was stated as an unsupported
   background assumption. Replaced with the actual MRR@5 figure (0.743)
   and its table citation, so the explanation rests on a stated number
   instead of an implicit, uncited premise.

5. **Discussion, §VI — reframed the q080 correction as a case, not a rate.**
   This was the pass's main find. The original text used the q080
   correction to "reinforce" the ablation's claim that embedding-model
   choice is "the more actionable lever" for closing failure gaps — but
   q080 is the specific case that revealed the embedding-model effect in
   the first place, so citing it again to support a general claim about
   which lever is "more actionable" double-counts one data point as if it
   were independent confirmation. Reworded to state plainly that this is
   a single demonstrated case, combined it explicitly with the ablation's
   aggregate parity result (the actual basis for a general claim), and
   added an honest caveat that how often "check the embedding model
   first" would resolve a given failure is unmeasured.

6. **Conclusion — removed an unsupported three-way comparison, tied the
   recommendation back to the evidence that actually supports it.**
   The conclusion claimed that benchmarking embedding models "is likely
   to close more failure cases, faster, than either corpus expansion or
   retrieval-strategy changes such as RRF re-tuning" — a comparison
   between three remediation strategies, none of which was measured
   against the others; only embedding benchmarking was ever tested, and
   only on one case. Replaced with a version that traces the
   recommendation to exactly the two things that support it (the q080
   case and the ablation's parity result), repeats the "not a measured
   comparison of success rates" caveat from #5, and grounds the
   recommendation in the paper's own experience: the first hypothesis for
   q080 was a corpus-engineering explanation that direct inspection showed
   was wrong — the specific risk this recommendation exists to guard
   against.

## Verification performed

- Re-read the Abstract and Conclusion back-to-back in the copy after all
  edits; both now state the ablation result at a consistent confidence
  level and the Conclusion's recommendation is explicitly scoped to its
  evidence.
- Checked every reference to q080 (Table III, §V-D/embedding ablation,
  §VI Discussion ×2, Conclusion ×2): all now frame it as a single
  demonstrated case, never as a rate or general efficacy claim.
- Grepped the full document for "match," "equivalent," "not significant,"
  and similar phrasing: every non-significance result now reads "no
  significant difference" / "not significant," never "confirmed
  equivalence." One instance already in §V-D ("point estimates... are
  directional, not confirmed differences," alongside "we report this as
  parity... not as one beating another") was left untouched — it was
  already correctly calibrated and is the passage the task specifically
  flagged not to disturb.
- Checked the "near-ceiling" / "near-optimal" language (3 instances) for
  circular use as both a finding and a justification for not investigating
  further: not circular. It motivates the ablation (§V-D's opening
  question) and summarizes dense's performance (Conclusion), and the
  Future Work paragraph immediately after still proposes four concrete
  next steps, so it isn't being used to foreclose further work.
- Checked nomic's long-context rationale specifically (the task's named
  risk area): §V-D already explicitly separates which of nomic's three
  original selection criteria survived the ablation (open-weight, pure
  bi-encoder — unaffected) from the one that didn't (long-context —
  "not empirically supported here"), and states that the rest of the
  paper's use of nomic rests on the surviving criteria, not the undercut
  one. Already correct; no change needed.
- Checked the difficulty-tag definitional-loop pattern ("hard because hard
  for BM25"): not present. Difficulty tags are assigned at annotation
  time on linguistic grounds (paraphrase vs. vocabulary-gap), independent
  of any system's retrieval results.
- Compiled the copy twice with pdflatex: clean build, 9 pages, no new
  overfull-hbox or undefined-reference warnings beyond a pre-existing
  benign font-shape warning also present in the original.

## Not changed

No numbers, statistics, or citations were altered. No hedging was added;
each fix either grounded an existing claim in a citation/number already
in the paper or narrowed a claim's scope to match its actual evidence.
Prose rhythm and word choice were left alone except where the wording
itself was the source of the logical issue. The original file,
`eu_climate_hybrid_retrieval.tex`, was not modified.

Merging this copy back into the original is an authorial decision, not
made here.
