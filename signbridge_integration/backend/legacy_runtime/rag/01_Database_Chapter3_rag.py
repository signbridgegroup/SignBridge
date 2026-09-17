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
DEFAULT_CHUNKS = PROJECT / "data" / "processed" / "01_Database_Chapter3_chunks.jsonl"
DEFAULT_INDEX = PROJECT / "data" / "vector_store" / "01_Database_Chapter3_bm25_index.json"

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TOKEN = re.compile(r"[A-Za-z0-9_+%/\->*:]+|[\u0600-\u06FF]+")

EXPANSIONS = {'erd': ['entity relationship diagram', 'er diagram', 'مخطط الكيان العلاقة'], 'mapping': ['transform', 'convert', 'تحويل', 'ربط'], 'regular': ['regular entity', 'كيان عادي'], 'weak': ['weak entity', 'dependent', 'كيان ضعيف'], 'binary': ['binary relationship', 'علاقة ثنائية'], 'unary': ['unary relationship', 'recursive relationship', 'علاقة أحادية'], 'one-to-many': ['1:n', 'one to many', 'واحد لمتعدد'], 'many-to-many': ['m:n', 'many to many', 'متعدد لمتعدد'], 'one-to-one': ['1:1', 'one to one', 'واحد لواحد'], 'primary': ['primary key', 'pk', 'مفتاح أساسي'], 'foreign': ['foreign key', 'fk', 'مفتاح أجنبي'], 'composite': ['composite attribute', 'component attributes', 'خاصية مركبة'], 'multivalued': ['multivalued attribute', 'new relation', 'متعددة القيم'], 'eer': ['enhanced er', 'supertype', 'subtype', 'specialization', 'generalization'], 'supertype': ['subtype', 'inheritance', 'نوع أعلى'], 'subtype': ['supertype', 'inheritance', 'نوع فرعي']}

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = ARABIC_DIACRITICS.sub("", text)
    return text.translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه"}))

def tokenize(text: str) -> list[str]:
    return TOKEN.findall(normalize(text))

def expanded_query(question: str) -> list[str]:
    terms = tokenize(question)
    additions = []
    nq = normalize(question)
    for term in terms:
        for expansion in EXPANSIONS.get(term, []):
            additions.extend(tokenize(expansion))
    for key, expansions in EXPANSIONS.items():
        if normalize(key) in nq:
            for expansion in expansions:
                additions.extend(tokenize(expansion))
    if any(x in nq for x in ("ما هو","ما هي","what is","define","meaning")):
        additions.extend(tokenize("definition meaning concept تعريف شرح"))
    if any(x in nq for x in ("فرق","مقارنه","difference","compare","comparison")):
        additions.extend(tokenize("difference compare comparison الفرق مقارنة"))
    if any(x in nq for x in ("كيف","how","map","mapping","convert","transform")):
        additions.extend(tokenize("how steps mapping transformation تحويل خطوات"))
    if any(x in nq for x in ("لماذا","why","purpose","importance")):
        additions.extend(tokenize("purpose importance reason السبب الأهمية"))
    return terms + additions

def read_chunks(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_index(chunks_path: Path, index_path: Path) -> dict:
    chunks = read_chunks(chunks_path)
    if not chunks:
        raise ValueError("The chunks file is empty.")
    documents = []
    df = Counter()
    for chunk in chunks:
        emphasized = " ".join(
            [str(chunk.get("slide_title",""))] * 3
            + [str(x) for x in chunk.get("section_headings", [])] * 2
        )
        tokens = tokenize(emphasized + "\n" + str(chunk["text"]))
        tf = Counter(tokens)
        df.update(tf.keys())
        documents.append({
            "chunk_id": chunk["chunk_id"],
            "slide_number": chunk["slide_number"],
            "slide_title": chunk["slide_title"],
            "part_number": chunk.get("part_number"),
            "section_headings": chunk.get("section_headings", []),
            "text": chunk["text"],
            "length": len(tokens),
            "term_frequency": dict(tf),
        })
    index = {
        "method": "BM25 + bilingual domain query expansion",
        "source": chunks_path.name,
        "documents": documents,
        "document_frequency": dict(df),
        "average_document_length": sum(d["length"] for d in documents) / len(documents),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print("Database Chapter 3 - Logical Database Design / ERD Mapping RAG index built.")
    print("Chunks:", len(documents))
    print("Output:", index_path)
    return index

def bm25(index: dict, query: list[str], document: dict) -> float:
    score = 0.0
    N = len(index["documents"])
    avgdl = index["average_document_length"]
    k1, b = 1.5, 0.75
    tf = document["term_frequency"]
    for term in query:
        freq = tf.get(term, 0)
        if not freq:
            continue
        df = index["document_frequency"].get(term, 0)
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        denom = freq + k1 * (1 - b + b * document["length"] / avgdl)
        score += idf * freq * (k1 + 1) / denom
    return score

def slide_boost(document: dict, question: str) -> float:
    """
    Strong intent-aware slide boosting for Chapter 3.

    Specific relationship/cardinality intents should outrank general
    introductory mapping slides.
    """
    nq = normalize(question)
    slide = int(document["slide_number"])
    title = normalize(str(document.get("slide_title", "")))

    boost = 0.0

    intents = [
        (
            ["many-to-many", "many to many", "m:n", "متعدد لمتعدد", "متعدد الى متعدد"],
            {12, 13},
            85.0,
        ),
        (
            ["one-to-many", "one to many", "1:n", "واحد لمتعدد", "واحد الى متعدد"],
            {9, 10, 11},
            80.0,
        ),
        (
            ["one-to-one", "one to one", "1:1", "واحد لواحد", "واحد الى واحد"],
            {14, 15},
            80.0,
        ),
        (
            ["weak entity", "weak entities", "dependent", "كيان ضعيف", "كيانات ضعيفه"],
            {6, 7},
            75.0,
        ),
        (
            ["regular entity", "regular entities", "كيان عادي", "كيانات عاديه"],
            {4, 5},
            70.0,
        ),
        (
            ["unary", "recursive", "recursive relationship", "علاقه احاديه", "علاقه ذاتيه"],
            {16, 17, 18, 19},
            75.0,
        ),
        (
            ["eer", "supertype", "subtype", "specialization", "generalization",
             "نوع اعلى", "نوع فرعي", "تخصص", "تعميم"],
            set(range(20, 27)),
            75.0,
        ),
    ]

    specific_intent_found = False

    for keys, target_slides, weight in intents:
        if any(normalize(key) in nq for key in keys):
            specific_intent_found = True
            if slide in target_slides:
                boost += weight
                # Extra title match boost for precise targeting.
                if any(normalize(key) in title for key in keys):
                    boost += 15.0

    # Penalize generic intro slides when the question has a specific intent.
    if specific_intent_found and slide in {1, 2, 3, 8}:
        boost -= 35.0

    return boost

def retrieve(index: dict, question: str, top_k: int = 3):
    query = expanded_query(question)
    original = set(tokenize(question))
    scored = []
    for document in index["documents"]:
        score = bm25(index, query, document)
        title = set(tokenize(document.get("slide_title","")))
        headings = set(tokenize(" ".join(document.get("section_headings",[]))))
        score += 5.0 * len(original & title)
        score += 2.0 * len(original & headings)
        score += slide_boost(document, question)
        scored.append((score, document))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x for x in scored if x[0] > 0][:top_k]

def clean_markdown(text: str) -> str:
    text = re.sub(r"```(?:[A-Za-z0-9_+/-]+)?", "", text)
    text = text.replace("```","")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("`","")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def local_fallback(results):
    best = results[0][1]
    text = best["text"]
    for heading in ["الفكرة الأساسية","Detailed Explanation","English Academic Summary","Key Concepts"]:
        m = re.search(rf"##\s+.*{re.escape(heading)}.*\n(.*?)(?=\n##|\Z)", text, re.S|re.I)
        if m:
            return best["chunk_id"], clean_markdown(m.group(1))
    return best["chunk_id"], clean_markdown(text)

def ask(index: dict, question: str, top_k: int = 3):
    results = retrieve(index, question, top_k)
    if not results:
        print("No relevant course content was found.")
        return
    try:
        answer = generate_grounded_answer(question, results, course_name="Database Chapter 3 - Logical Database Design / ERD Mapping")
        method = f"Groq LLM ({GROQ_MODEL})"
        fallback_source = None
    except Exception as error:
        print("\nWarning: Groq generation failed.")
        print("Reason:", error)
        fallback_source, answer = local_fallback(results)
        method = "Local RAG fallback"

    print("\nRetrieved sources:")
    for score, d in results:
        print(f"- {d['chunk_id']} | Slide {d['slide_number']} | {d['slide_title']} | score {score:.3f}")
    print("\nGeneration method:", method)
    print("\nAnswer:\n")
    print(answer)
    print("\nGrounding sources:")
    for score, d in results:
        print(f"- {d['chunk_id']} | Slide {d['slide_number']} | {d['slide_title']} | score {score:.3f}")
    if fallback_source:
        print("\nFallback primary source:", fallback_source)

def interactive_loop(index: dict, top_k: int):
    print("Database Chapter 3 - Logical Database Design / ERD Mapping RAG is ready.")
    print("اكتبي سؤالك بالعربي أو الإنجليزي، أو exit للخروج.")
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not q or normalize(q) in {"exit","quit","خروج"}:
            print("Goodbye.")
            return
        ask(index, q, top_k)

def main():
    parser = argparse.ArgumentParser(description="Database Chapter 3 - Logical Database Design / ERD Mapping educational RAG")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--question")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.build or not args.index.is_file():
        index = build_index(args.chunks, args.index)
    else:
        with args.index.open("r", encoding="utf-8") as f:
            index = json.load(f)
    if args.question:
        ask(index, args.question, args.top_k)
    elif not args.build:
        interactive_loop(index, args.top_k)

if __name__ == "__main__":
    main()
