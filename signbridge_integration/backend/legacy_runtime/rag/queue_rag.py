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


DEFAULT_CHUNKS = PROJECT / "data" / "processed" / "04_Queue_Core_chunks.jsonl"
DEFAULT_INDEX = PROJECT / "data" / "vector_store" / "queue_bm25_index.json"

EXPANSIONS = {
    "queue": ["طابور", "fifo", "first in first out"],
    "طابور": ["queue", "fifo"],
    "fifo": ["first in first out", "اول عنصر يدخل اول عنصر يخرج", "queue"],
    "سوال": ["ما هو", "تعريف", "meaning"],
    "استفهام": ["ما هو", "تعريف", "meaning"],
    "شرح": ["اشرح", "explain", "الفكرة", "تعريف"],
    "اشرح": ["شرح", "explain", "الفكرة"],
    "addqueue": ["اضافة", "إضافة", "ادخال", "إدخال", "enqueue", "insert", "rear"],
    "enqueue": ["addqueue", "اضافة", "إضافة", "rear"],
    "deletequeue": ["حذف", "ازالة", "إزالة", "dequeue", "remove", "front"],
    "dequeue": ["deletequeue", "حذف", "ازالة", "إزالة", "front"],
    "front": ["queuefront", "اول", "أول", "first", "first element"],
    "queuefront": ["front", "first element", "اول عنصر", "أول عنصر"],
    "back": ["rear", "queuerear", "last", "last element"],
    "rear": ["back", "queuerear", "نهاية", "اخر", "آخر"],
    "queuerear": ["rear", "back", "last element"],
    "isemptyqueue": ["فارغ", "empty", "count zero"],
    "empty": ["isemptyqueue", "فارغ", "count"],
    "فارغ": ["isemptyqueue", "empty"],
    "isfullqueue": ["ممتلئ", "full", "maxqueuesize"],
    "full": ["isfullqueue", "ممتلئ"],
    "ممتلئ": ["isfullqueue", "full"],
    "initializequeue": ["تهيئة", "initialize", "empty queue"],
    "initialize": ["initializequeue", "تهيئة"],
    "circular": ["circular queue", "circular array", "دائري", "modulo"],
    "دائري": ["circular", "circular queue", "modulo"],
    "modulo": ["%", "circular", "wrap around"],
    "array": ["مصفوفة", "queuefront", "queuerear"],
    "مصفوفة": ["array", "queuefront", "queuerear"],
    "count": ["عدد", "number of elements", "empty", "full"],
    "maxqueuesize": ["size", "capacity", "حجم", "سعة"],
    "size": ["حجم", "سعة", "maxqueuesize"],
    "priority": ["priority queue", "priority_queue", "اولوية", "أولوية"],
    "priority_queue": ["priority queue", "priority", "stl"],
    "اولوية": ["priority", "priority queue"],
    "stl": ["standard template library", "queue container adapter", "priority_queue"],
    "simulation": ["محاكاة", "queuing system", "server", "customer"],
    "محاكاة": ["simulation", "queuing system", "server", "customer"],
    "server": ["خادم", "simulation", "customer", "transaction time"],
    "customer": ["عميل", "simulation", "waiting time", "arrival time"],
    "waiting": ["waiting time", "waiting customers queue", "انتظار"],
    "constructor": ["انشاء", "تهيئة", "queue constructor"],
    "destructor": ["تحرير", "memory", "delete", "dynamic memory"],
    "فرق": ["difference", "compare", "مقارنة"],
    "مقارنة": ["difference", "compare", "فرق"],
    "كيف": ["how", "خطوات", "تنفيذ"],
    "لماذا": ["why", "سبب", "فائدة"],
}

ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

TOKEN = re.compile(r"[A-Za-z0-9_+%]+|[\u0600-\u06FF]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
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

    additions = []

    for term in terms:
        for extra in EXPANSIONS.get(term, []):
            additions.extend(tokenize(extra))

    normalized_question = normalize(question)

    if (
        "ما هو" in normalized_question
        or "ما هي" in normalized_question
        or "what is" in normalized_question
    ):
        additions.extend(
            tokenize(
                "تعريف الفكرة الأساسية meaning definition concept"
            )
        )

    if (
        "كيف" in normalized_question
        or "how" in normalized_question
    ):
        additions.extend(
            tokenize(
                "implementation steps operation تنفيذ خطوات"
            )
        )

    if (
        "فرق" in normalized_question
        or "مقارنه" in normalized_question
        or "difference" in normalized_question
    ):
        additions.extend(
            tokenize(
                "difference comparison compare مقارنة"
            )
        )

    return terms + additions


def read_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def build_index(chunks_path: Path, index_path: Path) -> dict:
    chunks = read_chunks(chunks_path)

    documents = []

    document_frequency = Counter()

    for chunk in chunks:
        emphasized = " ".join(
            [chunk["slide_title"]] * 3
            + chunk.get("section_headings", []) * 2
        )

        tokens = tokenize(
            emphasized + "\n" + chunk["text"]
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
                    []
                ),
                "text": chunk["text"],
                "length": len(tokens),
                "term_frequency": dict(frequencies),
            }
        )

    index = {
        "method": "BM25 with bilingual Queue query expansion",
        "source": str(chunks_path),
        "documents": documents,
        "document_frequency": dict(document_frequency),
        "average_document_length": (
            sum(d["length"] for d in documents)
            / len(documents)
        ),
    }

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with index_path.open(
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            index,
            handle,
            ensure_ascii=False
        )

    print("Queue RAG index built.")
    print("Chunks:", len(documents))
    print("Method:", index["method"])
    print("Output:", index_path)

    return index


def bm25(
    index: dict,
    query: list[str],
    document: dict
) -> float:
    score = 0.0

    count = len(index["documents"])

    average = index[
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
            0
        )

        if not frequency:
            continue

        df = index[
            "document_frequency"
        ].get(
            term,
            0
        )

        inverse = math.log(
            1
            + (
                count - df + 0.5
            )
            / (
                df + 0.5
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
                / average
            )
        )

        score += (
            inverse
            * frequency
            * (k1 + 1)
            / denominator
        )

    return score


def split_sections(
    text: str
) -> list[tuple[str, str]]:
    matches = list(
        re.finditer(
            r"^##\s+(.+)$",
            text,
            flags=re.MULTILINE,
        )
    )

    sections = []

    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        sections.append(
            (
                match.group(1).strip(),
                text[
                    match.end():end
                ].strip(),
            )
        )

    return sections or [
        (
            "Retrieved explanation",
            text
        )
    ]


def clean_markdown(text: str) -> str:
    text = re.sub(
        r"```(?:[A-Za-z0-9_+-]+)?",
        "",
        text
    )

    text = text.replace(
        "```",
        ""
    )

    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text
    )

    text = re.sub(
        r"__(.*?)__",
        r"\1",
        text
    )

    text = text.replace(
        "`",
        ""
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
        text
    )

    return text.strip()


def best_explanation(
    results: list[tuple[float, dict]],
    query: list[str],
    question: str
) -> tuple[str, str]:
    candidates = []

    excluded = (
        "original slide",
        "keywords",
        "academic summary",
        "source vs",
        "quick review",
    )

    normalized_question = normalize(
        question
    )

    definition_intent = (
        "ما هو" in normalized_question
        or "ما هي" in normalized_question
        or "what is" in normalized_question
    )

    question_terms = set(
        tokenize(question)
    )

    focus_terms = question_terms.intersection(
        {
            "queue",
            "fifo",
            "addqueue",
            "enqueue",
            "deletequeue",
            "dequeue",
            "front",
            "back",
            "rear",
            "queuefront",
            "queuerear",
            "isemptyqueue",
            "isfullqueue",
            "initializequeue",
            "circular",
            "modulo",
            "priority",
            "priority_queue",
            "simulation",
            "server",
            "customer",
            "constructor",
            "destructor",
        }
    )

    for document_rank, (
        _,
        document
    ) in enumerate(
        results[:3]
    ):
        for heading, content in split_sections(
            document["text"]
        ):
            normalized_heading = normalize(
                heading
            )

            if any(
                value in normalized_heading
                for value in excluded
            ):
                continue

            section_tokens = Counter(
                tokenize(
                    heading
                    + " "
                    + content
                )
            )

            overlap = sum(
                section_tokens.get(
                    term,
                    0
                )
                for term in set(query)
            )

            heading_tokens = tokenize(
                heading
            )

            heading_overlap = sum(
                heading_tokens.count(
                    term
                )
                for term in set(query)
            )

            score = (
                overlap
                + 5 * heading_overlap
                - document_rank
            )

            if definition_intent:
                if (
                    "الفكره الاساسيه"
                    in normalized_heading
                    or "definition"
                    in normalized_heading
                    or "ترجمه"
                    in normalized_heading
                ):
                    score += 30

            if any(
                term in heading_tokens
                for term in focus_terms
            ):
                score += 35

            candidates.append(
                (
                    score,
                    document["chunk_id"],
                    heading,
                    content,
                )
            )

    if not candidates:
        document = results[0][1]

        return (
            document["chunk_id"],
            clean_markdown(
                document["text"]
            ),
        )

    (
        _,
        chunk_id,
        heading,
        content,
    ) = max(
        candidates,
        key=lambda item: item[0],
    )

    cleaned = clean_markdown(
        content
    )

    paragraphs = [
        part.strip()
        for part in cleaned.split("\n\n")
        if part.strip()
    ]

    answer = ""

    for paragraph in paragraphs:
        if (
            len(answer)
            + len(paragraph)
            > 1100
            and answer
        ):
            break

        answer += (
            "\n\n"
            if answer
            else ""
        ) + paragraph

    return (
        chunk_id,
        f"{heading}\n\n{answer}"
    )




def ask(
    index: dict,
    question: str
):
    query = expanded_query(
        question
    )

    normalized_question = normalize(
        question
    )

    question_terms = set(
        tokenize(question)
    )

    definition_intent = (
        "ما هو" in normalized_question
        or "ما هي" in normalized_question
        or "what is" in normalized_question
    )

    queue_definition = (
        definition_intent
        and (
            "queue" in question_terms
            or "طابور" in question_terms
        )
    )

    focus_terms = question_terms.intersection(
        {
            "addqueue",
            "enqueue",
            "deletequeue",
            "dequeue",
            "front",
            "back",
            "rear",
            "queuefront",
            "queuerear",
            "isemptyqueue",
            "isfullqueue",
            "initializequeue",
            "circular",
            "modulo",
            "priority",
            "priority_queue",
            "simulation",
            "server",
            "customer",
            "constructor",
            "destructor",
        }
    )

    def retrieval_score(
        document
    ):
        score = bm25(
            index,
            query,
            document
        )

        title_normalized = normalize(
            document.get(
                "slide_title",
                ""
            )
        )

        heading_tokens = set(
            tokenize(
                " ".join(
                    document.get(
                        "section_headings",
                        []
                    )
                )
            )
        )

        if queue_definition:
            if (
                document["slide_number"] == 1
                or "introduction" in title_normalized
            ):
                score += 30

        if focus_terms.intersection(
            heading_tokens
        ):
            score += 25

        if (
            "addqueue" in question_terms
            or "enqueue" in question_terms
        ):
            if (
                document["slide_number"] == 18
                or "add queue"
                in title_normalized
            ):
                score += 35

        if (
            "deletequeue" in question_terms
            or "dequeue" in question_terms
        ):
            if (
                document["slide_number"] == 19
                or "delete queue"
                in title_normalized
            ):
                score += 35

        if "front" in question_terms:
            if document["slide_number"] == 16:
                score += 35

        if (
            "back" in question_terms
            or "rear" in question_terms
        ):
            if document["slide_number"] == 17:
                score += 35

        if (
            "isemptyqueue" in question_terms
            or "isfullqueue" in question_terms
            or "empty" in question_terms
            or "full" in question_terms
        ):
            if document["slide_number"] in {
                10,
                11,
                12,
                14,
            }:
                score += 25

        if (
            "initializequeue" in question_terms
            or "initialize" in question_terms
        ):
            if document["slide_number"] == 15:
                score += 35

        if (
            "circular" in question_terms
            or "دائري" in question_terms
            or "modulo" in question_terms
        ):
            if document["slide_number"] in {
                7,
                8,
                9,
                10,
                11,
                12,
            }:
                score += 30

        if (
            "priority" in question_terms
            or "priority_queue"
            in question_terms
        ):
            if document["slide_number"] in {
                24,
                25,
            }:
                score += 35

        if "stl" in question_terms:
            if document["slide_number"] in {
                22,
                23,
                25,
            }:
                score += 30

        if (
            "simulation" in question_terms
            or "محاكاه" in question_terms
        ):
            if document["slide_number"] >= 26:
                score += 25

        if "server" in question_terms:
            if document["slide_number"] in {
                29,
                33,
                34,
                35,
                37,
            }:
                score += 30

        if "customer" in question_terms:
            if document["slide_number"] in {
                29,
                31,
                32,
                35,
                37,
            }:
                score += 30

        if "constructor" in question_terms:
            if document["slide_number"] == 20:
                score += 35

        if "destructor" in question_terms:
            if document["slide_number"] in {
                20,
                21,
            }:
                score += 35

        return score

    scored = sorted(
        (
            (
                retrieval_score(
                    document
                ),
                document,
            )
            for document
            in index["documents"]
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    results = [
        item
        for item in scored
        if item[0] > 0
    ][:3]

    if not results:
        print(
            "No relevant Queue content was found."
        )
        return

    try:
        answer = generate_grounded_answer(
            question,
            results,
            course_name="Queue",
        )
        answer_method = f"Groq LLM ({GROQ_MODEL})"
        fallback_source = None
    except Exception as error:
        print("\nWarning: Groq generation failed.")
        print("Reason:", error)

        fallback_source, answer = best_explanation(
            results,
            query,
            question,
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
        answer_method
    )

    print(
        "\nAnswer:"
    )

    print(
        answer
    )

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
            fallback_source
        )

def main():
    parser = argparse.ArgumentParser(
        description="Local Queue educational RAG"
    )

    parser.add_argument(
        "--build",
        action="store_true"
    )

    parser.add_argument(
        "--question"
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX
    )

    args = parser.parse_args()

    if args.build or not args.index.is_file():
        index = build_index(
            args.chunks,
            args.index
        )
    else:
        with args.index.open(
            "r",
            encoding="utf-8"
        ) as handle:
            index = json.load(handle)

    if args.question:
        ask(index, args.question)
        return

    print("\nQueue RAG is ready.")
    print("Type your question below.")
    print("Type exit to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:
            print("Goodbye.")
            break

        print()
        ask(index, question)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()