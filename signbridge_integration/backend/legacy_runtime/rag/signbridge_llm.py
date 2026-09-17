from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq


PROJECT = Path(__file__).resolve().parent

load_dotenv(PROJECT / "key.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def build_llm_context(
    results: list[tuple[float, dict]],
    max_total_chars: int = 6500,
    max_chunk_chars: int = 3200,
) -> str:
    """Convert retrieved chunks into grounded LLM context with a safe size cap."""
    context_parts: list[str] = []
    used_chars = 0

    for rank, (score, document) in enumerate(results, start=1):
        text = str(document.get("text", "")).strip()
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars].rstrip() + "\n[chunk truncated for context limit]"

        slide_number = document.get("slide_number", "N/A")
        slide_title = str(document.get("slide_title", "")).strip()
        chunk_id = str(document.get("chunk_id", f"source_{rank}"))

        part = (
            f"SOURCE {rank}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Slide: {slide_number}\n"
            f"Title: {slide_title}\n"
            f"Retrieval score: {score:.3f}\n\n"
            f"{text}"
        )

        separator_cost = 30 if context_parts else 0
        remaining = max_total_chars - used_chars - separator_cost
        if remaining <= 0:
            break

        if len(part) > remaining:
            part = part[:remaining].rstrip() + "\n[context truncated]"
            context_parts.append(part)
            break

        context_parts.append(part)
        used_chars += len(part) + separator_cost

    return "\n\n====================\n\n".join(context_parts)


def build_sources(
    results: list[tuple[float, dict]],
) -> list[dict[str, Any]]:
    """
    Build deterministic grounding metadata from retrieval results.

    The LLM never creates or edits source metadata.
    """
    sources: list[dict[str, Any]] = []

    for rank, (score, document) in enumerate(results, start=1):
        sources.append(
            {
                "rank": rank,
                "chunk_id": document.get("chunk_id"),
                "slide_number": document.get("slide_number"),
                "slide_title": document.get("slide_title", ""),
                "retrieval_score": round(float(score), 3),
            }
        )

    return sources


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences if the model wraps JSON in them."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return text.strip()


def _parse_llm_json(raw_text: str) -> dict[str, str]:
    """
    Parse the model JSON safely and validate the two required fields.
    """
    cleaned = _strip_code_fences(raw_text)

    try:
        data = json.loads(cleaned)

    except json.JSONDecodeError:
        # Small recovery path: locate the outermost JSON object.
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(
                "The LLM response was not valid JSON."
            )

        try:
            data = json.loads(
                cleaned[start : end + 1]
            )

        except json.JSONDecodeError as error:
            raise RuntimeError(
                "The LLM returned malformed JSON."
            ) from error

    if not isinstance(data, dict):
        raise RuntimeError(
            "The LLM JSON response must be an object."
        )

    educational_answer = str(
        data.get("educational_answer", "")
    ).strip()

    simplified_text = str(
        data.get("simplified_text", "")
    ).strip()

    if not educational_answer:
        raise RuntimeError(
            "The LLM response is missing educational_answer."
        )

    if not simplified_text:
        raise RuntimeError(
            "The LLM response is missing simplified_text."
        )

    return {
        "educational_answer": educational_answer,
        "simplified_text": simplified_text,
    }


def generate_grounded_response(
    question: str,
    results: list[tuple[float, dict]],
    course_name: str = "Course Material",
) -> dict[str, Any]:
    """
    Generate the final structured SignBridge response.

    Returns:
        {
            "educational_answer": str,
            "simplified_text": str,
            "sources": [
                {
                    "rank": int,
                    "chunk_id": str,
                    "slide_number": ...,
                    "slide_title": str,
                    "retrieval_score": float
                }
            ]
        }

    `simplified_text` is the field intended for the avatar pipeline.
    """
    if groq_client is None:
        raise RuntimeError(
            "GROQ_API_KEY was not found. Make sure key.env exists "
            "in the project root."
        )

    if not results:
        raise ValueError(
            "No retrieved results were provided to the LLM."
        )

    context = build_llm_context(results)

    system_prompt = f"""
You are SignBridge, a retrieval-augmented educational assistant
for deaf and hard-of-hearing university students.

CURRENT KNOWLEDGE DOMAIN:
{course_name}

Use ONLY the retrieved course context provided to you.

STRICT GROUNDING RULES:
1. Every factual statement must be supported by the retrieved context.
2. Do not use outside knowledge.
3. Do not invent definitions, examples, code behavior, advantages,
   complexity claims, or consequences.
4. Do not strengthen a source statement into a more absolute claim.
   For example, do not say "always", "never", "fixed", or "cannot"
   unless the retrieved context explicitly supports that wording.
5. If the context is insufficient, say so clearly instead of guessing.
6. Answer in the same language as the student's question.
7. Preserve useful English technical terms when appropriate.
8. Give an example only if it is supported by the retrieved material.
9. Keep the educational answer clear and concise.
10. The simplified text must contain ONLY facts already present in the
    educational answer.
11. The simplified text is an accessibility planning text for downstream
    sign/avatar processing. Make it SIGN-FRIENDLY, not merely shorter:
    - use very short, direct sentences;
    - keep one core concept per sentence when possible;
    - remove filler, rhetorical wording, and unnecessary pronouns;
    - preserve essential technical terms such as Stack, Queue, LIFO, FIFO,
      Pointer, Schema, Instance, and normalization terms when they are relevant;
    - prefer explicit concept wording (for example: "الإضافة", "الحذف",
      "القراءة", "الأعلى", "المصفوفة") over vague references such as "ذلك" or "هذه العملية";
    - when the retrieved material supports a named technical operation, keep its
      standard term explicitly (for example: Push, Pop, Enqueue, Dequeue, Top);
    - when LIFO or FIFO is relevant, prefer writing the standard acronym explicitly
      rather than paraphrasing the same rule multiple times;
    - avoid repeating the same fact in both a definition and an unnecessary paraphrase;
    - prefer 3--6 compact clauses total for ``simplified_text``;
    - if an acronym such as LIFO/FIFO is present, do not immediately restate the same
      rule in another sentence unless needed for understanding;
    - avoid low-value implementation detail in ``simplified_text`` when it does not
      change the core concept (keep that detail in ``educational_answer`` instead);

    - for Pointer explanations, keep ``simplified_text`` concept-focused: prefer
      Pointer + variable + memory address + stored value/location. Include data type
      only when it is useful to the question; avoid extra grammatical paraphrases;

    - for Database Schema/Instance comparisons, keep ``simplified_text`` focused on:
      Schema = database structure; Instance = actual data at a specific time;
      Schema is comparatively stable while Instance changes when rows are added/removed.
    - for Primary Key explanations, keep ``simplified_text`` focused on:
      Primary Key = selected Candidate Key; it identifies each Entity uniquely;
      avoid low-value wording and express minimal/no-extra-attributes only if supported by context.
    - for Queue explanations: prioritize Queue/FIFO/front/rear and the core
      add/remove operations; do not include secondary access restrictions such as
      inability to access the middle in ``simplified_text`` unless the user
      specifically asks about access restrictions;
    - do not add facts or invent sign-language grammar.
12. Do NOT include citations, source numbers, line numbers, brackets,
    reference markers, or a Sources section in either answer.
    The application displays sources separately.

OUTPUT FORMAT:
Return ONLY valid JSON.
Do not use markdown fences.
Do not add text before or after the JSON.

Use exactly this schema:

{{
  "educational_answer": "clear grounded explanation",
  "simplified_text": "short direct sign-friendly version"
}}
""".strip()

    user_prompt = f"""
STUDENT QUESTION:
{question}

RETRIEVED COURSE CONTEXT:
{context}
""".strip()

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.0,
    )

    raw_answer = response.choices[0].message.content

    if not raw_answer:
        raise RuntimeError(
            "Groq returned an empty answer."
        )

    parsed = _parse_llm_json(
        raw_answer
    )

    return {
        "educational_answer": parsed["educational_answer"],
        "simplified_text": parsed["simplified_text"],
        "sources": build_sources(results),
    }


def generate_grounded_answer(
    question: str,
    results: list[tuple[float, dict]],
    course_name: str = "Course Material",
) -> str:
    """
    Backward-compatible wrapper for the current Queue/Stack/Pointers/
    Database CLI files.

    Existing RAG scripts can keep calling this function unchanged.
    """
    response = generate_grounded_response(
        question=question,
        results=results,
        course_name=course_name,
    )

    return (
        "EDUCATIONAL ANSWER:\n"
        f"{response['educational_answer']}\n\n"
        "SIGN-FRIENDLY ANSWER:\n"
        f"{response['simplified_text']}"
    )


def get_avatar_text(
    response: dict[str, Any],
) -> str:
    """
    Return only the simplified text intended for the avatar.
    """
    text = str(
        response.get("simplified_text", "")
    ).strip()

    if not text:
        raise ValueError(
            "The response does not contain simplified_text."
        )

    return text


def to_json(
    response: dict[str, Any],
) -> str:
    """
    Serialize a SignBridge structured response for an API,
    frontend, or avatar service.
    """
    return json.dumps(
        response,
        ensure_ascii=False,
        indent=2,
    )
