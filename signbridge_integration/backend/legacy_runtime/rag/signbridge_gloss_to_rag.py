"""Convert recognized sign glosses into a natural-language educational question.

This module is intentionally independent from retrieval.  It is the bridge between
sign recognition and the unified SignBridge router, so it must not be tied to a
single course domain such as Stack.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Final

# Arabic + English markers that may be emitted by the recognizer/gloss vocabulary.
QUESTION_MARKERS: Final = {
    "سوال", "سؤال", "استفهام", "ماذا", "ما", "what", "question",
}
EXPLAIN_MARKERS: Final = {
    "شرح", "اشرح", "توضيح", "explain", "describe",
}
HOW_MARKERS: Final = {
    "كيف", "كيفيه", "كيفية", "how",
}
WHY_MARKERS: Final = {
    "لماذا", "سبب", "ليه", "why",
}
REMOVABLE_LINKING_TOKENS: Final = {
    "هو", "هي", "هذا", "هذه", "the", "is", "are",
}


def _normalize_token(token: str) -> str:
    """Normalize a token only for intent matching; preserve original output text."""
    token = token.casefold().strip()
    token = token.translate(
        str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})
    )
    token = re.sub(r"[\u064B-\u065F\u0670]", "", token)
    return token


def _normalized_set(values: set[str]) -> set[str]:
    return {_normalize_token(value) for value in values}


_NORMALIZED_QUESTION = _normalized_set(QUESTION_MARKERS)
_NORMALIZED_EXPLAIN = _normalized_set(EXPLAIN_MARKERS)
_NORMALIZED_HOW = _normalized_set(HOW_MARKERS)
_NORMALIZED_WHY = _normalized_set(WHY_MARKERS)
_NORMALIZED_REMOVABLE = _normalized_set(REMOVABLE_LINKING_TOKENS)


def glosses_to_question(gloss_text: str) -> str:
    """Convert a recognized gloss sequence into a router-ready question.

    Examples
    --------
    ``سوال هو Stack`` -> ``ما هو Stack؟``
    ``كيف push stack`` -> ``كيف push stack؟``
    ``explain queue`` -> ``اشرح queue؟``

    The function does not retrieve course content and therefore works for Stack,
    Queue, Pointers, Database, and future SignBridge domains.
    """
    if not isinstance(gloss_text, str):
        raise TypeError("gloss_text must be a string")

    tokens = [token for token in gloss_text.strip().split() if token]
    if not tokens:
        raise ValueError("The gloss sequence is empty")

    normalized = [_normalize_token(token) for token in tokens]
    normalized_set = set(normalized)

    if normalized_set & _NORMALIZED_HOW:
        prefix = "كيف"
        markers = _NORMALIZED_HOW
    elif normalized_set & _NORMALIZED_WHY:
        prefix = "لماذا"
        markers = _NORMALIZED_WHY
    elif normalized_set & _NORMALIZED_EXPLAIN:
        prefix = "اشرح"
        markers = _NORMALIZED_EXPLAIN
    elif normalized_set & _NORMALIZED_QUESTION:
        prefix = "ما هو"
        markers = _NORMALIZED_QUESTION
    else:
        # A recognized educational term on its own is interpreted as a definition
        # request. This keeps isolated-sign input useful without guessing a domain.
        prefix = "ما هو"
        markers = set()

    content = [
        original
        for original, norm in zip(tokens, normalized)
        if norm not in markers and norm not in _NORMALIZED_REMOVABLE
    ]

    if not content:
        raise ValueError(
            "The recognized glosses contain an intent but no educational topic"
        )

    return f"{prefix} {' '.join(content)}؟"


def build_gloss_payload(gloss_text: str) -> dict[str, str]:
    """Return a small stable payload useful for APIs and tests."""
    return {
        "recognized_gloss": gloss_text.strip(),
        "question": glosses_to_question(gloss_text),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert recognized sign glosses into a SignBridge question."
    )
    parser.add_argument(
        "--gloss",
        required=True,
        help='Recognized gloss sequence, e.g. "سوال هو Stack"',
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of readable text.",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    payload = build_gloss_payload(args.gloss)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Recognized gloss:", payload["recognized_gloss"])
    print("Converted question:", payload["question"])


if __name__ == "__main__":
    main()

