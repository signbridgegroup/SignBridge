from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from signbridge_llm import generate_grounded_answer, GROQ_MODEL


PROJECT = Path(__file__).resolve().parent

DEFAULT_CHUNKS = PROJECT / "data" / "processed" / "05_Pointers_Core_chunks.jsonl"
DEFAULT_INDEX = PROJECT / "data" / "vector_store" / "pointers_bm25_index.json"


EXPANSIONS = {
    "pointer": ["pointer", "مؤشر", "مؤشرات"],
    "address": ["address", "عنوان", "memory address", "عنوان الذاكرة"],
    "dereference": [
        "dereference",
        "dereferencing",
        "indirection",
        "فك الإشارة",
        "القيمة المشار إليها",
    ],
    "new": ["new", "allocate", "allocation", "حجز", "ذاكرة ديناميكية"],
    "delete": ["delete", "deallocate", "تحرير", "memory leak", "dangling"],
    "null": ["null", "NULL", "0", "مؤشر فارغ"],
    "array": ["array", "dynamic array", "مصفوفة", "مصفوفة ديناميكية"],
    "copy": [
        "copy",
        "shallow copy",
        "deep copy",
        "نسخ",
        "نسخ سطحي",
        "نسخ عميق",
    ],
    "arrow": ["arrow", "->", "سهم"],
    "class": ["class", "object", "كلاس", "كائن"],
    "arithmetic": [
        "pointer arithmetic",
        "arithmetic",
        "++",
        "--",
        "حساب المؤشرات",
    ],
    "leak": ["memory leak", "تسرب ذاكرة"],
    "dangling": ["dangling pointer", "مؤشر معلق"],
    "destructor": ["destructor", "مدمر"],
    "bad_alloc": ["bad_alloc", "allocation failure", "فشل الحجز"],
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\u0600-\u06FF+\-*>&]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def expanded_query(question: str) -> str:
    out = [question]
    normalized_question = normalize(question)

    for key, values in EXPANSIONS.items():
        if (
            key in normalized_question
            or any(
                normalize(value) in normalized_question
                for value in values
            )
        ):
            out.extend(values)

    return " ".join(out)


def read_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def build_index(
    chunks_path: Path,
    index_path: Path,
) -> dict:
    docs = read_chunks(chunks_path)

    tokenized = [
        tokenize(
            " ".join(
                [doc.get("slide_title", "")] * 3
                + doc.get("section_headings", []) * 2
            )
            + "\n"
            + doc["text"]
        )
        for doc in docs
    ]

    df = Counter()

    for tokens in tokenized:
        df.update(set(tokens))

    index = {
        "method": "BM25 with bilingual Pointers query expansion",
        "avgdl": (
            sum(map(len, tokenized))
            / max(1, len(tokenized))
        ),
        "df": dict(df),
        "docs": [],
    }

    for doc, tokens in zip(docs, tokenized):
        index["docs"].append(
            {
                **doc,
                "tokens": tokens,
                "tf": dict(Counter(tokens)),
                "dl": len(tokens),
            }
        )

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path.write_text(
        json.dumps(
            index,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Pointers RAG index built.")
    print(f"Chunks: {len(docs)}")
    print("Method: BM25 with bilingual Pointers query expansion")
    print(f"Output: {index_path}")

    return index


def bm25(
    index: dict,
    query: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[float, dict]]:
    query_tokens = tokenize(
        expanded_query(query)
    )

    count = len(index["docs"])
    average_length = index["avgdl"]

    scores = []

    for document in index["docs"]:
        score = 0.0

        for term in query_tokens:
            document_frequency = index["df"].get(
                term,
                0,
            )

            if not document_frequency:
                continue

            inverse = math.log(
                1
                + (
                    count
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )

            frequency = document["tf"].get(
                term,
                0,
            )

            if not frequency:
                continue

            denominator = (
                frequency
                + k1
                * (
                    1
                    - b
                    + b
                    * document["dl"]
                    / average_length
                )
            )

            score += (
                inverse
                * (
                    frequency
                    * (k1 + 1)
                )
                / denominator
            )

        scores.append(
            (
                score,
                document,
            )
        )

    return sorted(
        scores,
        key=lambda item: item[0],
        reverse=True,
    )


def retrieval_score(
    score: float,
    document: dict,
    question: str,
) -> float:
    normalized_question = normalize(question)
    slide_number = document["slide_number"]

    boosts = [
        (
            [
                "ما هو pointer",
                "what is pointer",
                "pointer definition",
                "ما هو المؤشر",
            ],
            {5},
        ),
        (
            [
                "declare",
                "declaration",
                "تصريح",
            ],
            {6, 7},
        ),
        (
            [
                "address",
                "عنوان",
            ],
            {8},
        ),
        (
            [
                "dereference",
                "dereferencing",
                "indirection",
                "فك الاشاره",
            ],
            {9, 10},
        ),
        (
            [
                "arrow",
                "->",
            ],
            {12, 13},
        ),
        (
            [
                "null",
                "مؤشر فارغ",
            ],
            {14},
        ),
        (
            [
                "dynamic",
                "ديناميكي",
                "ذاكره ديناميكيه",
            ],
            {
                15, 16, 17, 18, 19,
                25, 26, 30,
            },
        ),
        (
            [
                "new",
                "allocate",
                "allocation",
                "حجز",
            ],
            {
                16, 17, 18, 19,
                20, 25, 30, 33,
            },
        ),
        (
            [
                "delete",
                "deallocate",
                "تحرير",
            ],
            {20, 21, 32, 35},
        ),
        (
            [
                "leak",
                "memory leak",
                "تسرب",
            ],
            {20, 21, 35},
        ),
        (
            [
                "dangling",
                "معلق",
            ],
            {21, 32},
        ),
        (
            [
                "arithmetic",
                "++",
                "--",
            ],
            {22, 23, 24, 25, 29},
        ),
        (
            [
                "array",
                "مصفوفه",
            ],
            {25, 26, 27, 28, 29, 30},
        ),
        (
            [
                "shallow",
                "نسخ سطحي",
            ],
            {31, 32},
        ),
        (
            [
                "deep",
                "نسخ عميق",
            ],
            {33},
        ),
        (
            [
                "copy",
                "نسخ",
            ],
            {31, 32, 33},
        ),
        (
            [
                "destructor",
                "مدمر",
            ],
            {35},
        ),
        (
            [
                "bad_alloc",
                "allocation failure",
                "فشل الحجز",
            ],
            {19},
        ),
        (
            [
                "class",
                "object",
                "كلاس",
                "كائن",
            ],
            {11, 12, 13, 34, 35},
        ),
    ]

    for keys, slides in boosts:
        if (
            any(
                normalize(key)
                in normalized_question
                for key in keys
            )
            and slide_number in slides
        ):
            score += 20

    return score


def clean_markdown(text: str) -> str:
    text = re.sub(
        r"```(?:[A-Za-z0-9_+-]+)?",
        "",
        text,
    )
    text = text.replace(
        "```",
        "",
    )
    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text,
    )
    text = re.sub(
        r"__(.*?)__",
        r"\1",
        text,
    )
    text = text.replace(
        "`",
        "",
    )
    text = re.sub(
        r"^[-*]\s+",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^---+$",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def local_fallback(
    results: list[tuple[float, dict]],
) -> tuple[str, str]:
    best = results[0][1]
    text = best["text"]

    match = re.search(
        r"##\s*الفكرة الأساسية\s*(.*?)(?=\n##|\Z)",
        text,
        flags=re.S,
    )

    answer = (
        match.group(1).strip()
        if match
        else clean_markdown(text)
    )

    return (
        best["chunk_id"],
        answer,
    )



def retrieve(
    index: dict,
    question: str,
    top_k: int = 3,
) -> list[tuple[float, dict]]:
    """Return ranked Pointers chunks for the unified SignBridge router.

    Uses the same BM25 + Pointers-specific scoring already used by ask(),
    but returns results instead of printing/generating an answer.
    """
    ranked = [
        (
            retrieval_score(
                score,
                document,
                question,
            ),
            document,
        )
        for score, document in bm25(
            index,
            question,
        )
    ]

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        item
        for item in ranked
        if item[0] > 0
    ][:top_k]


def ask(
    index: dict,
    question: str,
    top_k: int = 3,
):
    ranked = [
        (
            retrieval_score(
                score,
                document,
                question,
            ),
            document,
        )
        for score, document in bm25(
            index,
            question,
        )
    ]

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    results = [
        item
        for item in ranked
        if item[0] > 0
    ][:top_k]

    if not results:
        print(
            "No relevant Pointers content was found."
        )
        return

    try:
        answer = generate_grounded_answer(
            question,
            results,
            course_name="Pointers",
        )
        answer_method = (
            f"Groq LLM ({GROQ_MODEL})"
        )
        fallback_source = None

    except Exception as error:
        print(
            "\nWarning: Groq generation failed."
        )
        print(
            "Reason:",
            error,
        )

        (
            fallback_source,
            answer,
        ) = local_fallback(
            results
        )

        answer_method = "Local RAG fallback"

    print(
        "\nRetrieved sources:"
    )

    for score, document in results:
        print(
            f"- {document['chunk_id']} "
            f"| Slide {document['slide_number']} "
            f"| score {score:.3f}"
        )

    print(
        "\nGeneration method:",
        answer_method,
    )

    print(
        "\nAnswer:"
    )

    print(answer)

    print(
        "\nGrounding sources:"
    )

    for score, document in results:
        print(
            f"- {document['chunk_id']} "
            f"| Slide {document['slide_number']} "
            f"| {document.get('slide_title', '')} "
            f"| score {score:.3f}"
        )

    if fallback_source:
        print(
            "\nFallback primary source:",
            fallback_source,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Local Pointers educational RAG"
    )

    parser.add_argument(
        "--build",
        action="store_true",
    )

    parser.add_argument(
        "--question",
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS,
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
    )

    args = parser.parse_args()

    if (
        args.build
        or not args.index.is_file()
    ):
        index = build_index(
            args.chunks,
            args.index,
        )
    else:
        index = json.loads(
            args.index.read_text(
                encoding="utf-8"
            )
        )

    if args.question:
        ask(
            index,
            args.question,
        )
        return

    print(
        "\nPointers RAG is ready."
    )
    print(
        "Type your question below."
    )
    print(
        "Type exit to stop.\n"
    )

    while True:
        try:
            question = input(
                "Question: "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print(
                "\nGoodbye."
            )
            break

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print(
                "Goodbye."
            )
            break

        print()

        ask(
            index,
            question,
        )

        print(
            "\n"
            + "=" * 60
            + "\n"
        )


if __name__ == "__main__":
    main()
