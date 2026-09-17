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

CHUNKS_FILENAME = "01_Database_Chapter2_chunks.jsonl"
INDEX_FILENAME = "01_Database_Chapter2_bm25_index.json"

DEFAULT_CHUNKS = PROJECT / "data" / "processed" / CHUNKS_FILENAME
DEFAULT_INDEX = PROJECT / "data" / "vector_store" / INDEX_FILENAME


ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)
TOKEN = re.compile(r"[A-Za-z0-9_+%/\->*]+|[\u0600-\u06FF]+")


EXPANSIONS: dict[str, list[str]] = {
    "er": ["e-r", "entity relationship", "entity-relationship", "نموذج الكيان العلاقة"],
    "erd": ["er diagram", "entity relationship diagram", "مخطط الكيان العلاقة"],
    "entity": ["entity set", "كيان", "مجموعة كيانات"],
    "كيان": ["entity", "entity set"],
    "relationship": ["relationship set", "علاقة", "مجموعة علاقات"],
    "علاقه": ["relationship", "relationship set"],
    "attribute": ["attributes", "خاصية", "صفة", "خصائص"],
    "attributes": ["attribute", "خصائص"],
    "خاصيه": ["attribute", "attributes"],
    "simple": ["simple attribute", "خاصية بسيطة"],
    "composite": ["composite attribute", "component attribute", "خاصية مركبة"],
    "multivalued": ["multivalued attribute", "multi valued", "متعددة القيم"],
    "derived": ["derived attribute", "مشتقة"],
    "domain": ["attribute domain", "permitted values", "مجال"],
    "key": ["primary key", "candidate key", "superkey", "مفتاح"],
    "primary": ["primary key", "مفتاح أساسي"],
    "candidate": ["candidate key", "مفتاح مرشح"],
    "superkey": ["super key", "superkey", "مفتاح فائق"],
    "cardinality": [
        "mapping cardinality",
        "one to one",
        "one to many",
        "many to one",
        "many to many",
        "1:1",
        "1:n",
        "n:1",
        "m:n",
        "الكارديناليتي",
        "قيد التعددية",
    ],
    "one-to-one": ["one to one", "1:1"],
    "one-to-many": ["one to many", "1:n"],
    "many-to-one": ["many to one", "n:1"],
    "many-to-many": ["many to many", "m:n"],
    "participation": ["total participation", "partial participation", "مشاركة"],
    "total": ["total participation", "مشاركة كلية"],
    "partial": ["partial participation", "مشاركة جزئية"],
    "weak": ["weak entity", "weak entity set", "كيان ضعيف"],
    "identifying": ["identifying relationship", "علاقة تعريفية"],
    "discriminator": ["partial key", "discriminator", "مفتاح جزئي"],
    "role": ["roles", "role indicator", "دور"],
    "binary": ["binary relationship", "degree two", "علاقة ثنائية"],
    "ternary": ["ternary relationship", "degree three", "علاقة ثلاثية"],
    "degree": ["degree of relationship", "درجة العلاقة"],
    "non-binary": ["non binary relationship", "ternary", "علاقة غير ثنائية"],
    "design": ["database design", "conceptual design", "logical design", "physical design", "تصميم"],
    "conceptual": ["conceptual schema", "conceptual design", "مفاهيمي"],
    "logical": ["logical design", "database schema", "منطقي"],
    "physical": ["physical design", "physical layout", "فيزيائي"],
    "normalization": ["normalization theory", "تطبيع"],
    "constraint": ["constraints", "cardinality constraint", "participation constraint", "قيد"],
    "diamond": ["relationship set", "diamond", "معين"],
    "rectangle": ["entity set", "rectangle", "مستطيل"],
    "underline": ["primary key", "underlined attribute", "تسطير"],
    "فرق": ["difference", "compare", "comparison", "مقارنة"],
    "مقارنه": ["difference", "compare", "comparison", "فرق"],
    "اشرح": ["explain", "definition", "concept", "شرح"],
    "شرح": ["explain", "definition", "concept", "اشرح"],
    "كيف": ["how", "steps", "representation", "implementation", "خطوات"],
    "لماذا": ["why", "reason", "purpose", "importance", "سبب"],
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = ARABIC_DIACRITICS.sub("", text)
    return text.translate(
        str.maketrans(
            {
                "أ": "ا",
                "إ": "ا",
                "آ": "ا",
                "ى": "ي",
                "ة": "ه",
            }
        )
    )


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(normalize(text))


def expanded_query(question: str) -> list[str]:
    terms = tokenize(question)
    additions: list[str] = []

    normalized_question = normalize(question)

    for term in terms:
        for expansion in EXPANSIONS.get(term, []):
            additions.extend(tokenize(expansion))

    for key, expansions in EXPANSIONS.items():
        if normalize(key) in normalized_question:
            for expansion in expansions:
                additions.extend(tokenize(expansion))

    if any(
        phrase in normalized_question
        for phrase in ("ما هو", "ما هي", "what is", "define", "meaning")
    ):
        additions.extend(tokenize("definition meaning concept تعريف الفكرة الأساسية"))

    if any(
        phrase in normalized_question
        for phrase in ("فرق", "مقارنه", "difference", "compare", "comparison")
    ):
        additions.extend(tokenize("difference compare comparison الفرق مقارنة"))

    if any(
        phrase in normalized_question
        for phrase in ("كيف", "how", "represent", "representation")
    ):
        additions.extend(tokenize("how representation notation diagram تمثيل مخطط"))

    if any(
        phrase in normalized_question
        for phrase in ("لماذا", "why", "purpose", "importance")
    ):
        additions.extend(tokenize("purpose importance reason السبب الأهمية"))

    return terms + additions


def read_chunks(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Chunks file not found: {path}\n"
            f"Expected it at:\n{DEFAULT_CHUNKS}"
        )

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
    chunks = read_chunks(chunks_path)

    if not chunks:
        raise ValueError("The chunks file is empty.")

    documents: list[dict] = []
    document_frequency: Counter[str] = Counter()

    for chunk in chunks:
        emphasized = " ".join(
            [str(chunk.get("slide_title", ""))] * 3
            + [str(x) for x in chunk.get("section_headings", [])] * 2
        )

        tokens = tokenize(
            emphasized + "\n" + str(chunk["text"])
        )

        frequencies = Counter(tokens)
        document_frequency.update(
            frequencies.keys()
        )

        documents.append(
            {
                "chunk_id": chunk["chunk_id"],
                "slide_number": chunk["slide_number"],
                "slide_title": chunk["slide_title"],
                "part_number": chunk["part_number"],
                "section_headings": chunk.get(
                    "section_headings",
                    [],
                ),
                "text": chunk["text"],
                "length": len(tokens),
                "term_frequency": dict(frequencies),
            }
        )

    index = {
        "method": "BM25 with bilingual Database Chapter 2 query expansion",
        "source": chunks_path.name,
        "documents": documents,
        "document_frequency": dict(document_frequency),
        "average_document_length": (
            sum(
                document["length"]
                for document in documents
            )
            / len(documents)
        ),
    }

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with index_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            index,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("Database Chapter 2 RAG index built.")
    print("Chunks:", len(documents))
    print("Method:", index["method"])
    print("Output:", index_path)

    return index


def bm25(
    index: dict,
    query: list[str],
    document: dict,
) -> float:
    score = 0.0

    document_count = len(
        index["documents"]
    )

    average_length = index[
        "average_document_length"
    ]

    k1 = 1.5
    b = 0.75

    frequencies = document[
        "term_frequency"
    ]

    for term in query:
        frequency = frequencies.get(
            term,
            0,
        )

        if not frequency:
            continue

        document_frequency = index[
            "document_frequency"
        ].get(
            term,
            0,
        )

        inverse_frequency = math.log(
            1
            + (
                document_count
                - document_frequency
                + 0.5
            )
            / (
                document_frequency
                + 0.5
            )
        )

        denominator = (
            frequency
            + k1
            * (
                1
                - b
                + b
                * document["length"]
                / average_length
            )
        )

        score += (
            inverse_frequency
            * frequency
            * (k1 + 1)
            / denominator
        )

    return score


def slide_boost(
    document: dict,
    question: str,
) -> float:
    normalized_question = normalize(question)
    slide = int(document["slide_number"])

    rules = [
        (["design phase", "مراحل التصميم", "initial phase", "conceptual phase"], {2, 3}),
        (["design approach", "er model", "e-r model", "normalization"], {4, 5}),
        (["entity set", "كيان", "entity"], {6, 7}),
        (["relationship set", "علاقه", "relationship"], {8, 9, 10, 11, 12}),
        (["degree", "binary", "ثنائي"], {13}),
        (["ternary", "non-binary", "ثلاثي"], {14}),
        (["attribute", "خاصيه", "خصائص"], {15, 16, 17}),
        (["primary key", "مفتاح اساسي", "superkey", "candidate key"], {18, 19, 20}),
        (["relationship key", "primary key for relationship"], {21, 22}),
        (["cardinality", "one to one", "one-to-one", "1:1"], {23, 24, 26}),
        (["one to many", "one-to-many", "1:n"], {23, 24, 26, 27}),
        (["many to one", "many-to-one", "n:1"], {23, 25, 26, 28}),
        (["many to many", "many-to-many", "m:n"], {23, 25, 26, 29}),
        (["participation", "total participation", "partial participation"], {30, 31}),
        (["weak entity", "weak entity set", "كيان ضعيف", "discriminator"], {32, 33, 34, 35}),
        (["role", "roles", "دور"], {36}),
        (["symbol", "notation", "رموز", "er notation"], {37, 38}),
    ]

    boost = 0.0

    for keys, slides in rules:
        if (
            any(
                normalize(key)
                in normalized_question
                for key in keys
            )
            and slide in slides
        ):
            boost += 20.0

    return boost


def retrieve(
    index: dict,
    question: str,
    top_k: int = 3,
) -> list[tuple[float, dict]]:
    query = expanded_query(question)
    original_terms = set(
        tokenize(question)
    )

    scored: list[
        tuple[float, dict]
    ] = []

    for document in index[
        "documents"
    ]:
        score = bm25(
            index,
            query,
            document,
        )

        title_terms = set(
            tokenize(
                document.get(
                    "slide_title",
                    "",
                )
            )
        )

        heading_terms = set(
            tokenize(
                " ".join(
                    document.get(
                        "section_headings",
                        [],
                    )
                )
            )
        )

        score += (
            3.0
            * len(
                original_terms
                & title_terms
            )
        )

        score += (
            1.5
            * len(
                original_terms
                & heading_terms
            )
        )

        score += slide_boost(
            document,
            question,
        )

        scored.append(
            (
                score,
                document,
            )
        )

    ranked = sorted(
        scored,
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        item
        for item in ranked
        if item[0] > 0
    ][:top_k]


def clean_markdown(text: str) -> str:
    text = re.sub(
        r"```(?:[A-Za-z0-9_+/-]+)?",
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
    results: list[
        tuple[float, dict]
    ],
) -> tuple[str, str]:
    best = results[0][1]
    text = best["text"]

    preferred_headings = [
        "الفكرة الأساسية",
        "Key Concept",
        "Key Concepts",
        "Detailed Explanation",
        "English Academic Summary",
    ]

    for heading in preferred_headings:
        match = re.search(
            rf"##\s+.*{re.escape(heading)}.*\n(.*?)(?=\n##|\Z)",
            text,
            flags=re.S | re.I,
        )

        if match:
            return (
                best["chunk_id"],
                clean_markdown(
                    match.group(1)
                ),
            )

    return (
        best["chunk_id"],
        clean_markdown(text),
    )


def ask(
    index: dict,
    question: str,
    top_k: int = 3,
) -> None:
    results = retrieve(
        index,
        question,
        top_k,
    )

    if not results:
        print(
            "No relevant Database Chapter 2 content was found."
        )
        return

    try:
        answer = generate_grounded_answer(
            question,
            results,
            course_name="Database Chapter 2 - E-R Model",
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

        answer_method = (
            "Local RAG fallback"
        )

    print(
        "\nRetrieved sources:"
    )

    for score, document in results:
        print(
            f"- {document['chunk_id']} "
            f"| Slide {document['slide_number']} "
            f"| {document['slide_title']} "
            f"| score {score:.3f}"
        )

    print(
        "\nGeneration method:",
        answer_method,
    )

    print(
        "\nAnswer:\n"
    )

    print(answer)

    print(
        "\nGrounding sources:"
    )

    for score, document in results:
        print(
            f"- {document['chunk_id']} "
            f"| Slide {document['slide_number']} "
            f"| {document['slide_title']} "
            f"| score {score:.3f}"
        )

    if fallback_source:
        print(
            "\nFallback primary source:",
            fallback_source,
        )


def interactive_loop(
    index: dict,
    top_k: int,
) -> None:
    print(
        "Database Chapter 2 RAG is ready."
    )
    print(
        "اكتبي سؤالك بالعربي أو الإنجليزي، أو اكتبي exit للخروج."
    )

    while True:
        try:
            question = input(
                "\nQuestion: "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print(
                "\nGoodbye."
            )
            return

        if (
            not question
            or normalize(question)
            in {
                "exit",
                "quit",
                "خروج",
            }
        ):
            print(
                "Goodbye."
            )
            return

        ask(
            index,
            question,
            top_k,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Database Chapter 2 educational RAG"
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="Rebuild the BM25 index from chunks",
    )

    parser.add_argument(
        "--question",
        help="Ask one Arabic or English question",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks",
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS,
        help="Path to Chapter 2 chunks JSONL",
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Path to Chapter 2 BM25 index JSON",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error(
            "--top-k must be at least 1"
        )

    if (
        args.build
        or not args.index.is_file()
    ):
        index = build_index(
            args.chunks,
            args.index,
        )

    else:
        with args.index.open(
            "r",
            encoding="utf-8",
        ) as handle:
            index = json.load(handle)

    if args.question:
        ask(
            index,
            args.question,
            args.top_k,
        )

    elif not args.build:
        interactive_loop(
            index,
            args.top_k,
        )


if __name__ == "__main__":
    main()
