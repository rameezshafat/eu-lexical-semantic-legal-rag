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
