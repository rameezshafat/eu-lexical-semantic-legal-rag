# Review notes after running everything end to end

Ran the whole thing on my laptop (12 June). Tests pass (35), built the indices,
ran the 71-query eval and the baselines, got generation working through Ollama
(had to use llama3.1:8b, my machine can't fit the 70b), and the Gradio UI works.

Numbers I got:

```
BM25 only      Hit@5 0.9577   MRR@5 0.8397
BGE-M3 only    Hit@5 0.9859   MRR@5 0.8901
Hybrid RRF     Hit@5 1.0000   MRR@5 0.8674
```

Writing down everything I think we need to fix or at least talk about before
the paper. Roughly in order of how much it worries me.

---

## SECOND UPDATE (4 July) - the "fusion is dead" conclusion below didn't survive a bigger test set

Since the last update a lot changed: we switched the dense model to nomic-embed
(8k context), added hierarchical chunking, split the queries into a 49-query
validation set (used for tuning the RRF weights, now 5:1 dense:sparse, k=20) and
a held-out test set, found and fixed a real bug in our NDCG (duplicate articles
from the same document were being credited multiple times, which is why some
NDCG values were over 1.0 - anything citing those numbers is wrong), and
expanded the test set from 22 to 53 queries. The 31 new queries were derived
from actual European Parliament written questions (E-numbers in the notes field
of each query as an audit trail), paraphrased into plain language, with corpus
membership checked for every mapping.

Results on the 53-query test set:

```
BM25           HR@5 0.6981   MRR@5 0.5280   NDCG@5 0.4935
nomic (dense)  HR@5 0.9245   MRR@5 0.7425   NDCG@5 0.7094
Hybrid 5:1     HR@5 0.9057   MRR@5 0.7730   NDCG@5 0.7208
```

Two things happened that matter:

1. We finally have significant results. Hybrid beats BM25 (Wilcoxon p=0.0005,
   d=0.579) and dense beats BM25 (p=0.0036, d=0.433). At n=22 nothing was
   significant. The benchmark expansion did exactly what it was supposed to.

2. The hybrid-vs-dense direction FLIPPED. At n=22 dense led hybrid on MRR; at
   n=53 hybrid leads dense (0.7730 vs 0.7425). Still not significant either way
   (p=0.61, d=0.12, negligible). Which means my earlier conclusion below -
   "naive fusion actively hurts, there is no ratio where adding BM25 helps" -
   was small-sample noise, and I wrote it too confidently. Lesson learned the
   honest way.

So the actual paper claim is now: weighted hybrid fusion and dense-only are
statistically indistinguishable on this benchmark; both significantly
outperform BM25. Fusion doesn't rescue retrieval but it doesn't cost anything
either. That's weaker than "hybrid wins" but it's true, and the direction-flip
story is worth a paragraph in the paper as a warning about reading directional
differences at small n.

Still open before these numbers are final: ~11 test queries carry VERIFY/FLAG
notes for the second annotator, the hard-negative annotations in the master
file (72 of 102 queries) haven't been synced into the test split (only 4 test
queries have them, so ignore HN_Rate for now), and bootstrap_ci.py computes
HN_Rate with a different denominator than eval_test.py (it averages in
unannotated queries as zeros - eval_test's number is the right one). After the
annotator pass: freeze the test set, re-run once, those are the paper numbers.

Everything below this line is the older history - kept because it shows how the
conclusions evolved, but where it contradicts the above, the above wins.

---

## UPDATE - I went ahead and rewrote the queries, and it changes the story

After writing the notes below I actually did the thing I was worried about: I
rewrote all 71 queries so they sound like a real person asking, stripped out the
statute wording and the instrument names, and tagged 19 of them "hard" (the ones
where the question uses completely different words from the law, e.g. "the
don't-make-anything-else-worse test" instead of "do no significant harm"). The
CELEX mappings are unchanged, only the question text. Then I re-ran everything.

The leakage was real and it was basically holding BM25 up. Same corpus, same
indices, just the de-leaked queries:

```
              before (leaky)        after (realistic queries)
BM25 only     0.9577 / 0.8397   ->  0.6901 / 0.5408
BGE-M3 only   0.9859 / 0.8901   ->  0.9577 / 0.8291
Hybrid RRF    1.0000 / 0.8674   ->  0.9437 / 0.7448
                                    (Hit@5 / MRR@5)
```

BM25 falls off a cliff (-27 points hit, -30 MRR). Dense barely moves (-3, -6)
because it works on meaning not words. And the thing nobody wanted to see:
the hybrid is now WORSE than dense-only on both metrics. Equal-weight RRF is
blending BM25's bad rankings into dense's good ones and dragging it down. So
"hybrid wins" is dead on the realistic benchmark.

### Experiment 1 - can weighting dense higher save the hybrid?

Added per-retriever weights to the fusion (`rrf_dense_weight` / `rrf_sparse_weight`
in config, plumbed through the controller) and swept the dense:sparse ratio.
Script is `scripts/sweep_weights.py`.

```
System         Hit@5    MRR@5
BM25 only      0.6901   0.5408
BGE-M3 only    0.9577   0.8291   <- the bar to beat
Hybrid 1:1     0.9437   0.7448
Hybrid 2:1     0.9437   0.7408
Hybrid 3:1     0.9437   0.7549
Hybrid 5:1     0.9437   0.7570
Hybrid 10:1    0.9437   0.7852
```

Weighting dense up does help - MRR climbs from 0.745 toward 0.785 - but it only
ever approaches dense-only from below and never beats it. Makes sense: crank the
dense weight to infinity and the hybrid just becomes dense-only. There is no
ratio where adding BM25 improves on pure dense. Hit@5 is stuck at 0.9437 the whole
way, which is exactly one query that dense alone gets but any amount of BM25 weight
knocks out of the top 5.

### Experiment 2 - does BM25 at least win on the keyword-heavy queries?

This was my last hope for a fusion story - maybe BM25 wins on the "standard"
queries (the ones still using fairly legal wording) even if it loses on the hard
ones, in which case the contribution becomes query routing instead of blind
fusion. Split the results by the difficulty tag. Script is
`scripts/breakdown_by_difficulty.py`.

```
System         standard (n=52)        hard (n=19)
BM25 only      Hit 0.731  MRR 0.589   Hit 0.579  MRR 0.408
BGE-M3 only    Hit 0.942  MRR 0.813   Hit 1.000  MRR 0.873
Hybrid 1:1     Hit 0.923  MRR 0.727   Hit 1.000  MRR 0.794
```

BM25 loses in BOTH buckets, even the keyword-heavy one (0.731 vs 0.942). So
routing won't save it either - there's no slice where lexical or fusion beats
dense. Funny detail: dense is actually better on the hard queries (1.000) than
the standard ones (0.942), because the hard ones are weirdly worded but
conceptually sharp, which is exactly what embeddings are good at. The standard
bucket has the genuinely ambiguous multi-document queries that trip everything up.

### So what's the actual paper now

I think we stop trying to prove "hybrid wins" because on a clean benchmark it
doesn't. The honest and more interesting story is: once you remove query-corpus
vocabulary leakage, dense retrieval beats BM25 and fusion across every query type,
and naive fusion actively hurts. That's a proper negative/cautionary result for
legal RAG and we've got the mechanism and the numbers to back it up. It's a better
paper than the one we were going to write.

Big caveats before any of this goes in the paper, in fairness:

- I wrote these queries myself, which is exactly the thing the notes below say we
  shouldn't do. They need a blind second author and someone checking the relevance
  judgements. Treat these numbers as directional, not final.
- I deliberately took away BM25's best case by never putting instrument names or
  article numbers in the queries. Real users mostly don't know those, so I think
  it's fair, but if our users include experts who do cite article numbers, that
  group needs its own queries before we write BM25 off completely.
- Still document-level scoring, and n=19 on the hard bucket is small. We need the
  significance tests and article-level scoring before trusting the exact margins.
  The direction is clear though.

Everything below is the original notes. Most of it still stands; the gold-standard
section is now partly done (queries rewritten, still need blind validation and
article-level judgements).

---

## The gold standard is too easy

This is the big one. Hit@5 of 1.0 across all 71 queries is not a good sign, it
means the benchmark can't tell the systems apart anymore. I'm fairly sure the
problem is that the queries were written while looking at the corpus, so the
wording of the questions matches the wording of the articles. BM25 especially
benefits from that.

What I think we should do:

- Rewrite the queries using outside sources, e.g. Commission FAQs, parliamentary
  questions, practitioner guides. Basically language an actual user would type,
  not paraphrased statute text.
- Whoever writes the new queries shouldn't be reading the corpus while doing it.
  One of us writes questions from a topic list, the other maps them to CELEX IDs.
- Get a second person to check the relevance judgements on at least 20 of them
  and note the agreement. Reviewers will ask how the benchmark was built so we
  should be able to describe the protocol properly.
- Also: we score at document level right now, but the system retrieves articles.
  Any article from the right document counts as a hit, which is very generous.
  We should record relevant article numbers in the gold standard and score at
  article level too. Keep document level as the lenient metric.
- Add a few queries that have no answer in the corpus at all, to test refusal.
  Heads up, the GoldQuery schema currently requires at least one relevant CELEX
  id (min_length=1) so that needs a small change.

## The hybrid result doesn't say what we want it to say

Dense-only actually beats the hybrid on MRR (0.8901 vs 0.8674). Hybrid only wins
on Hit@5. So "hybrid is better" is not true as a blanket statement - what's true
is that fusion catches the queries that one retriever misses, at a small cost in
how high the first relevant result lands. We either write it that way honestly,
or we try to fix the gap:

- k=60 for RRF is just the default from the Cormack paper, nobody tuned it.
  Worth trying 20/40/80.
- Could also try weighting dense higher in the fusion since it has the better MRR.
- The most interesting thing for the paper imo would be a breakdown of which
  query types BM25 wins vs which dense wins. That's the actual story.

Also with only 71 queries the MRR differences might just be noise. We need a
paired significance test (permutation or bootstrap) and confidence intervals
before claiming anything.

## Citations from the model are not trustworthy

Tested generation with the maritime query. The answer looked fine at first
glance but it contained 4 bracketed citations and only 1 of them actually
matched a provision we gave the model. The others were article numbers it
picked out of cross-references inside the context text. So the strict prompt
is not enough on its own (at least not for the 8b model).

Suggestion: post-process the answer, pull out every [CELEX - Article N]
pattern, check it against the context block, and flag or drop the ones that
weren't provided. Then we can actually report citation precision as a number,
which honestly makes the paper better - measuring faithfulness instead of
just claiming it.

Related: the refusal test ("what is the capital of France") - the model did
refuse, but it said "I can't answer that" instead of the exact INSUFFICIENT
CONTEXT string the prompt demands. The UI keys off that exact string so the
refusal renders like a normal answer. Either we detect refusals more loosely
or we confirm the 70b model actually follows the format.

And to be clear, none of my generation observations count as evaluation -
they're from an 8b model on a laptop. Any generation quality claims in the
paper need to come from the actual target model on a proper machine.

## Amendment articles are polluting the index

When I ran the nonsense query, 3 of the top 5 results were "Amendments to
Regulation X" type articles. These are boilerplate that quote fragments of
other instruments, they weakly match everything, and they attribute content
to the amending act's CELEX instead of the act people actually care about.
We should decide what to do with them - filter them out, split them into the
provisions they insert, or at minimum own it in the limitations section.
Also means the gold standard needs a consistent rule about whether the
amending act or the amended act is the "relevant" document.

## Smaller things

- Retrieval always returns top-5 no matter how bad the scores are, so garbage
  queries still go to the LLM. A score threshold that short-circuits to
  "insufficient context" without an LLM call would be cheap and gives us
  another thing to evaluate.
- The ETL collects cross_references and EuroVoc concept_ids per article and we
  use neither. Cross-reference expansion (pull in the provisions cited by the
  top hits) would actually be a nice contribution for legal RAG specifically.
  If there's no time, fine, but then let's drop the dead fields and put it in
  future work rather than shipping schema nobody reads.
- No reranker - fine as a scope decision, but if we have time one cross-encoder
  ablation row would tell us whether the MRR gap just disappears with reranking.
- Logging is incredibly noisy, every run prints dozens of httpx INFO lines from
  huggingface. One line in main.py/app.py to set httpx to WARNING. Setting
  HF_HUB_OFFLINE=1 after the first download also makes startup faster and means
  the demo runs fully offline, which is literally our selling point.
- requirements.txt has sentence-transformers and openai as >= instead of pinned.
  Pin them, and note the bge-m3 revision hash somewhere, for reproducibility.
- Check that no .venv folders are tracked in git (there's one under docs/Dataset
  that looks suspicious).
- We could run the 35 offline tests in GitLab CI, they need no GPU and take
  about 30 seconds.
- README should mention DENSE_BATCH_SIZE - indexing crashed on my 8GB laptop at
  the default batch 64, worked at 4. Took about 35 min on CPU. Generation with
  8b was 4-6 min per answer on my machine, so demos need either a smaller model
  or a better computer.

## What I'd do first

1. Fix the gold standard (new queries, blind protocol, article-level judgements).
   Everything else depends on this.
2. Re-run the ablations with significance tests, rewrite the hybrid claim honestly.
3. Citation validator + report citation precision.
4. Decide on amendment articles.
5. Generation eval on the real model, on real hardware.
6. The small stuff above whenever, it's all cheap.
7. Cross-reference expansion only if time allows.
