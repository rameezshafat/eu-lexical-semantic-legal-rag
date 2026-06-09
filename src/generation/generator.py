"""
Legal Generator: strictly-grounded LLM response layer.

Design contract
---------------
The generator ONLY receives the output of RankFusionController — it never
calls the retrievers directly. This enforces the separation between the
retrieval layer (independently evaluable) and the generation layer.

The system prompt is deliberately restrictive:
- The LLM may only draw on the provided fused provisions.
- Every claim must be traced to a CELEX ID + article number.
- If the context cannot answer the query, the LLM must say so explicitly
  rather than hallucinating from pre-training knowledge.
"""

from __future__ import annotations

import logging

from openai import OpenAI

from src.models.schemas import FusedResult, GenerationOutput

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a specialist EU climate law analyst. Your role is to answer legal questions \
with strict doctrinal grounding, drawing exclusively on the legislative provisions \
supplied in the CONTEXT block below.

RULES — you must follow all of them:

1. ONLY use information that appears verbatim or is directly inferable from the \
   provisions in CONTEXT. Do not draw on any external knowledge, legal commentary, \
   or pre-training information.

2. Every substantive legal claim in your answer MUST be explicitly cited using the \
   format: [CELEX_ID — Article N]. Multiple citations are permitted and encouraged.

3. If you quote or closely paraphrase a provision, place the citation immediately \
   after the quoted or paraphrased text, not at the end of a paragraph.

4. If the provided context does not contain sufficient information to answer the \
   question, respond with exactly:
   "INSUFFICIENT CONTEXT: The provided legislative provisions do not address this \
   question. A definitive answer requires consultation of [list the specific \
   instruments or articles that would be relevant, if known]."

5. Do not speculate, interpolate between provisions, or apply general legal \
   reasoning that goes beyond what the text of the provisions explicitly states.

6. Structure your response with a direct answer first, followed by the supporting \
   provisions. Keep the answer concise and legally precise.
"""


class LegalGenerator:
    """
    Wraps a local Llama model via Ollama to produce strictly-grounded legal answers.

    Parameters
    ----------
    model:
        Ollama model tag. Defaults to llama3.3:70b.
    max_tokens:
        Upper bound on generated tokens.
    """

    def __init__(
        self,
        model: str = "llama3.3:70b",
        max_tokens: int = 2048,
    ) -> None:
        self._client     = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self._model      = model
        self._max_tokens = max_tokens

    def generate(
        self,
        query: str,
        fused_results: list[FusedResult],
    ) -> GenerationOutput:
        """
        Generate a strictly-grounded answer for *query* using the fused provisions.

        Parameters
        ----------
        query:
            The legal question posed by the user.
        fused_results:
            Ordered list from RankFusionController.fuse_results().
            The generator consumes this directly — never calls retrievers.

        Returns
        -------
        GenerationOutput
            Contains the answer, extracted citations, and token usage.
        """
        if not fused_results:
            return GenerationOutput(
                query=query,
                answer=(
                    "INSUFFICIENT CONTEXT: No relevant legislative provisions "
                    "were retrieved for this query."
                ),
                cited_provisions=[],
                fused_results=[],
                model_used=self._model,
            )

        context_block = self._build_context_block(fused_results)
        user_message  = f"{context_block}\n\nQUESTION: {query}"

        log.info(
            "Generating answer for query: %s… (%d fused provisions)",
            query[:60],
            len(fused_results),
        )

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        if not response.choices:
            log.error(
                "Ollama API returned an empty choices list for query: %s…",
                query[:60],
            )
            return GenerationOutput(
                query=query,
                answer="INSUFFICIENT CONTEXT: The model returned no content for this query.",
                cited_provisions=[],
                fused_results=fused_results,
                model_used=self._model,
                input_tokens=getattr(response.usage, "prompt_tokens", 0),
                output_tokens=getattr(response.usage, "completion_tokens", 0),
            )

        answer = response.choices[0].message.content

        if response.choices[0].finish_reason == "length":
            log.warning(
                "Response truncated at max_tokens=%d for query: %s…",
                self._max_tokens,
                query[:60],
            )
            answer += (
                "\n\n[NOTE: This response was truncated at the token limit "
                "and may be incomplete.]"
            )

        cited = self._extract_citations(answer, fused_results)

        log.info(
            "Generation complete: %d input tokens, %d output tokens, %d citations used.",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            len(cited),
        )

        return GenerationOutput(
            query=query,
            answer=answer,
            cited_provisions=cited,
            fused_results=fused_results,
            model_used=self._model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_context_block(fused_results: list[FusedResult]) -> str:
        """
        Format the fused provisions into a numbered CONTEXT block for the LLM.
        """
        lines = ["CONTEXT (fused legal provisions, ordered by relevance):"]
        lines.append("=" * 70)

        for result in fused_results:
            art = result.article
            lines.append(
                f"\n[{result.rank}] {art.citation_label}"
                f"  (type: {art.doc_type}, RRF rank: {result.rank},"
                f" {result.provenance_str})"
            )
            lines.append("-" * 60)
            lines.append(art.article_text)

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    @staticmethod
    def _extract_citations(
        text: str, fused_results: list[FusedResult]
    ) -> list[str]:
        """
        Extract which citation labels the LLM actually used in its answer.

        Uses the known citation_label values from fused_results as search
        targets, avoiding regex over arbitrary CELEX patterns.
        """
        return [
            result.article.citation_label
            for result in fused_results
            if result.article.citation_label in text
        ]
