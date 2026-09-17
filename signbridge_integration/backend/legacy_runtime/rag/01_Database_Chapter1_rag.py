
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from signbridge_llm import generate_grounded_answer, GROQ_MODEL


BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILENAME = "01_Database_Chapter1_chunks.jsonl"
INDEX_FILENAME = "01_Database_Chapter1_bm25_index.json"

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TOKEN = re.compile(r"[A-Za-z0-9_+%/-]+|[\u0600-\u06FF]+")

# These mappings supplement the exact course wording.  They make equivalent
# Arabic/English terms meet in the same BM25 query without adding noisy
# standalone keyword chunks to the learning material.
EXPANSIONS: dict[str, list[str]] = {
    "database": ["database system", "dbms", "قاعدة بيانات", "نظام قواعد البيانات"],
    "databases": ["database", "dbms", "قواعد بيانات"],
    "dbms": ["database system", "database", "نظام قواعد البيانات"],
    "قاعده": ["database", "dbms", "قاعدة بيانات"],
    "بيانات": ["database", "data", "database system"],
    "file": ["file system", "traditional file system", "ملفات"],
    "files": ["file system", "traditional file system", "ملفات"],
    "file-system": ["file system", "traditional file system", "ملفات"],
    "redundancy": ["data redundancy", "تكرار البيانات", "inconsistency"],
    "inconsistency": ["data inconsistency", "عدم الاتساق", "redundancy"],
    "atomicity": ["atomicity of updates", "all or nothing", "ذرية"],
    "concurrency": ["concurrent transactions", "concurrency control", "التزامن"],
    "security": ["authorization", "access control", "الأمان"],
    "model": ["data model", "relational model", "نموذج بيانات"],
    "relational": ["relational model", "relation", "table", "علائقي"],
    "relation": ["relational model", "table", "relation schema", "جدول"],
    "table": ["relation", "rows", "columns", "جدول"],
    "rows": ["row", "tuples", "records", "صفوف"],
    "columns": ["column", "attributes", "fields", "أعمدة"],
    "schema": ["database schema", "logical schema", "physical schema", "مخطط"],
    "instance": ["database instance", "actual data", "حالة قاعدة البيانات"],
    "ddl": ["data definition language", "schema definition", "تعريف البيانات"],
    "dml": ["data manipulation language", "access and update", "معالجة البيانات"],
    "sql": ["sql query language", "query", "nonprocedural", "استعلام"],
    "query": ["sql", "query language", "استعلام"],
    "nonprocedural": ["declarative", "what not how", "غير إجرائية"],
    "procedural": ["how to get data", "procedural dml", "إجرائية"],
    "embedded": ["embedded sql", "host language", "sql in program"],
    "odbc": ["jdbc", "api", "database access"],
    "jdbc": ["odbc", "api", "database access"],
    "application": ["application program", "host language", "database access", "برنامج تطبيقي"],
    "host": ["host language", "python", "java", "لغة مضيفة"],
    "design": ["database design", "logical design", "physical design", "تصميم"],
    "logical": ["logical design", "logical schema", "منطقي"],
    "physical": ["physical design", "physical schema", "فيزيائي"],
    "transaction": ["transaction management", "atomicity", "consistency", "معاملة"],
    "transactions": ["transaction", "concurrency", "transactions management"],
    "architecture": ["database architecture", "centralized", "client server", "معمارية"],
    "centralized": ["centralized database", "single server", "مركزية"],
    "client": ["client server", "two tier", "three tier", "عميل"],
    "server": ["client server", "application server", "خادم"],
    "parallel": ["parallel database", "shared disk", "shared nothing", "متوازي"],
    "distributed": ["distributed database", "geographical distribution", "موزع"],
    "tier": ["two tier", "three tier", "application server", "طبقة"],
    "two": ["two tier", "client database server"],
    "three": ["three tier", "application server", "middle tier"],
    "dba": ["database administrator", "authorization", "backup", "مدير قاعدة البيانات"],
    "administrator": ["database administrator", "dba", "backup", "authorization"],
    "history": ["history of database systems", "timeline", "تاريخ"],
    "فرق": ["difference", "compare", "comparison", "مقارنة"],
    "مقارنه": ["difference", "compare", "comparison", "فرق"],
    "اشرح": ["explain", "definition", "concept", "شرح"],
    "شرح": ["explain", "definition", "concept", "اشرح"],
    "كيف": ["how", "steps", "implementation", "خطوات"],
    "لماذا": ["why", "reason", "purpose", "سبب"],
    "ما": ["definition", "meaning", "concept"],
}


def default_chunk_candidates() -> list[Path]:
    """Support both the six-file bundle and the SignBridge project layout."""
    working_directory = Path.cwd().resolve()
    candidates = [
        # Six downloaded files placed beside this script.
        BASE_DIR / CHUNKS_FILENAME,
        # Standard SignBridge project layout.
        BASE_DIR / "data" / "processed" / CHUNKS_FILENAME,
        # Script placed inside a child folder such as src/ or scripts/.
        BASE_DIR.parent / "data" / "processed" / CHUNKS_FILENAME,
        # Running from the project root while the script is elsewhere.
        working_directory / CHUNKS_FILENAME,
        working_directory / "data" / "processed" / CHUNKS_FILENAME,
    ]
    return list(dict.fromkeys(candidates))


def resolve_default_chunks() -> Path:
    for candidate in default_chunk_candidates():
        if candidate.is_file():
            return candidate
    # Prefer the standard project location in the error message when no file
    # has been copied yet; read_chunks() will list every searched location.
    return BASE_DIR / "data" / "processed" / CHUNKS_FILENAME


def resolve_default_index(chunks_path: Path) -> Path:
    # Keep an index with the project-wide vector-store convention whenever the
    # chunks use data/processed.  Otherwise, keep the six-file portable bundle
    # self-contained beside the script.
    if chunks_path.parent.name == "processed" and chunks_path.parent.parent.name == "data":
        return chunks_path.parent.parent / "vector_store" / INDEX_FILENAME
    return BASE_DIR / INDEX_FILENAME


DEFAULT_CHUNKS = resolve_default_chunks()
DEFAULT_INDEX = resolve_default_index(DEFAULT_CHUNKS)


def normalize(text: str) -> str:
    """Normalise Arabic and English safely for lexical retrieval."""
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

    for term in terms:
        for expansion in EXPANSIONS.get(term, []):
            additions.extend(tokenize(expansion))

    normalized = normalize(question)
    if any(phrase in normalized for phrase in ("ما هو", "ما هي", "what is", "define", "meaning")):
        additions.extend(tokenize("definition meaning concept الفكرة الأساسية التعريف"))
    if any(phrase in normalized for phrase in ("كيف", "how", "steps", "implementation")):
        additions.extend(tokenize("how implementation steps process خطوات تنفيذ"))
    if any(phrase in normalized for phrase in ("فرق", "مقارنه", "difference", "compare", "comparison")):
        additions.extend(tokenize("difference compare comparison الفرق مقارنة"))
    if any(phrase in normalized for phrase in ("لماذا", "why", "purpose", "importance")):
        additions.extend(tokenize("purpose importance reason السبب الأهمية"))

    return terms + additions


def read_chunks(path: Path) -> list[dict]:
    if not path.is_file():
        searched = "\n  - ".join(str(candidate) for candidate in default_chunk_candidates())
        raise FileNotFoundError(
            "Chunks file not found. Put '01_Database_Chapter1_chunks.jsonl' in "
            "data/processed or provide its exact path with --chunks.\n"
            f"Requested: {path}\n"
            f"Tried:\n  - {searched}"
        )
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_index(chunks_path: Path, index_path: Path) -> dict:
    chunks = read_chunks(chunks_path)
    if not chunks:
        raise ValueError("The chunks file is empty.")

    documents: list[dict] = []
    document_frequency: Counter[str] = Counter()

    for chunk in chunks:
        keywords = chunk.get("rag_keywords", [])
        emphasized = " ".join(
            [str(chunk.get("slide_title", ""))] * 3
            + [str(heading) for heading in chunk.get("section_headings", [])] * 2
            + [str(keyword) for keyword in keywords] * 2
        )
        tokens = tokenize(emphasized + "\n" + str(chunk["text"]))
        frequencies = Counter(tokens)
        document_frequency.update(frequencies.keys())

        documents.append(
            {
                "chunk_id": chunk["chunk_id"],
                "slide_number": chunk["slide_number"],
                "slide_title": chunk["slide_title"],
                "part_number": chunk["part_number"],
                "section_headings": chunk.get("section_headings", []),
                "rag_keywords": keywords,
                "text": chunk["text"],
                "length": len(tokens),
                "term_frequency": dict(frequencies),
            }
        )

    index = {
        "method": "BM25 with bilingual Database Chapter 1 query expansion",
        "source": chunks_path.name,
        "documents": documents,
        "document_frequency": dict(document_frequency),
        "average_document_length": sum(document["length"] for document in documents) / len(documents),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)

    print("Database Chapter 1 RAG index built.")
    print("Chunks:", len(documents))
    print("Method:", index["method"])
    print("Output:", index_path)
    return index


def bm25(index: dict, query: list[str], document: dict) -> float:
    score = 0.0
    document_count = len(index["documents"])
    average_length = index["average_document_length"]
    k1 = 1.5
    b = 0.75
    frequencies = document["term_frequency"]

    for term in query:
        frequency = frequencies.get(term, 0)
        if not frequency:
            continue
        document_frequency = index["document_frequency"].get(term, 0)
        inverse_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        denominator = frequency + k1 * (1 - b + b * document["length"] / average_length)
        score += inverse_frequency * frequency * (k1 + 1) / denominator
    return score


def split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^##\s+(.+)$", text, flags=re.MULTILINE))
    if not matches:
        return [("Retrieved explanation", text)]

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group(1).strip(), text[match.end() : end].strip()))
    return sections


def split_subsections(text: str) -> list[tuple[str, str]]:
    """Expose conceptual level-three headings inside an explanation section."""
    matches = list(re.finditer(r"^###\s+(.+)$", text, flags=re.MULTILINE))
    if not matches:
        return [("", text)]

    subsections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        subsections.append(("", preamble))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        subsections.append((match.group(1).strip(), text[match.end() : end].strip()))
    return subsections


def clean_markdown(text: str) -> str:
    text = re.sub(r"```(?:[A-Za-z0-9_+/-]+)?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def answer_from_results(results: list[tuple[float, dict]], query: list[str], question: str) -> tuple[str, str]:
    """Return the most relevant educational section from the strongest chunks."""
    normalized_question = normalize(question)
    definition_intent = any(phrase in normalized_question for phrase in ("ما هو", "ما هي", "what is", "define"))
    excluded = ("original slide", "possible student questions")
    candidates: list[tuple[float, str, str, str]] = []

    for result_rank, (_, document) in enumerate(results[:3]):
        for heading, content in split_sections(document["text"]):
            normalized_heading = normalize(heading)
            if any(excluded_heading in normalized_heading for excluded_heading in excluded):
                continue

            for subheading, subsection in split_subsections(content):
                candidate_heading = heading if not subheading else f"{heading}\n\n### {subheading}"
                candidate_text = candidate_heading + " " + subsection
                section_tokens = Counter(tokenize(candidate_text))
                overlap = sum(section_tokens.get(term, 0) for term in set(query))
                heading_overlap = sum(tokenize(candidate_heading).count(term) for term in set(query))
                # Prefer the highest-ranked retrieval source strongly; the
                # subsection score then identifies its most direct explanation.
                score = overlap + 6 * heading_overlap - 12 * result_rank

                if definition_intent and any(
                    marker in normalize(candidate_heading)
                    for marker in ("ما هي", "what is", "definition", "تعريف", "detailed explanation", "key concepts")
                ):
                    score += 24
                candidates.append((score, document["chunk_id"], candidate_heading, subsection))

    if not candidates:
        document = results[0][1]
        return document["chunk_id"], clean_markdown(document["text"])

    _, chunk_id, heading, content = max(candidates, key=lambda item: item[0])
    cleaned = clean_markdown(content)
    answer_parts: list[str] = []
    answer_length = 0
    for paragraph in (part.strip() for part in cleaned.split("\n\n")):
        if not paragraph:
            continue
        if answer_parts and answer_length + len(paragraph) > 1200:
            break
        answer_parts.append(paragraph)
        answer_length += len(paragraph)
    return chunk_id, f"{heading}\n\n" + "\n\n".join(answer_parts)


def retrieve(index: dict, question: str, top_k: int) -> list[tuple[float, dict]]:
    query = expanded_query(question)
    original_terms = set(tokenize(question))
    scored: list[tuple[float, dict]] = []

    for document in index["documents"]:
        score = bm25(index, query, document)
        title_terms = set(tokenize(document["slide_title"]))
        heading_terms = set(tokenize(" ".join(document.get("section_headings", []))))
        keyword_terms = set(tokenize(" ".join(document.get("rag_keywords", []))))

        # A direct match in the slide title, headings, or metadata is more
        # trustworthy than an accidental mention buried inside an example.
        score += 2.5 * len(original_terms & title_terms)
        score += 1.5 * len(original_terms & heading_terms)
        score += 1.0 * len(original_terms & keyword_terms)
        scored.append((score, document))

    return [item for item in sorted(scored, key=lambda item: item[0], reverse=True) if item[0] > 0][:top_k]


def ask(index: dict, question: str, top_k: int) -> None:
    results = retrieve(index, question, top_k)

    if not results:
        print("No relevant Database Chapter 1 content was found.")
        return

    try:
        answer = generate_grounded_answer(
            question,
            results,
            course_name="Database Chapter 1",
        )
        answer_method = f"Groq LLM ({GROQ_MODEL})"
        fallback_source = None

    except Exception as error:
        print("\nWarning: Groq generation failed.")
        print("Reason:", error)

        fallback_source, answer = answer_from_results(
            results,
            expanded_query(question),
            question,
        )
        answer_method = "Local RAG fallback"

    print("\nRetrieved sources:")

    for score, document in results:
        print(
            f"- {document['chunk_id']} | Slide {document['slide_number']} "
            f"| {document['slide_title']} | score {score:.3f}"
        )

    print("\nGeneration method:", answer_method)

    print("\nAnswer:\n")
    print(answer)

    print("\nGrounding sources:")

    for score, document in results:
        print(
            f"- {document['chunk_id']} | Slide {document['slide_number']} "
            f"| {document['slide_title']} | score {score:.3f}"
        )

    if fallback_source:
        print("\nFallback primary source:", fallback_source)

def interactive_loop(index: dict, top_k: int) -> None:
    """Make the VS Code Run button useful without requiring terminal flags."""
    print("Database Chapter 1 RAG is ready.")
    print("اكتبي سؤالك بالعربي أو الإنجليزي، أو اكتبي exit للخروج.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not question or normalize(question) in {"exit", "quit", "خروج"}:
            print("Goodbye.")
            return
        ask(index, question, top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Database Chapter 1 educational RAG")
    parser.add_argument("--build", action="store_true", help="Rebuild the BM25 index from chunks")
    parser.add_argument("--question", help="Ask one Arabic or English question")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks to show")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS, help="Path to the chunks JSONL")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Path to the BM25 index JSON")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    if args.build or not args.index.is_file():
        index = build_index(args.chunks, args.index)
    else:
        with args.index.open("r", encoding="utf-8") as handle:
            index = json.load(handle)

    if args.question:
        ask(index, args.question, args.top_k)
    elif not args.build:
        interactive_loop(index, args.top_k)


if __name__ == "__main__":
    main()
