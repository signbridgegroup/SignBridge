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
DEFAULT_CHUNKS = PROJECT / "data" / "processed" / "01_Database_Chapter4_chunks.jsonl"
DEFAULT_INDEX = PROJECT / "data" / "vector_store" / "01_Database_Chapter4_bm25_index.json"

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TOKEN = re.compile(r"[A-Za-z0-9_+%/\->*:]+|[\u0600-\u06FF]+")

EXPANSIONS = {'normalization': ['normal forms', 'تطبيع'], 'redundancy': ['duplicate data', 'data redundancy', 'تكرار'], 'anomaly': ['anomalies', 'update anomaly', 'insertion anomaly', 'deletion anomaly', 'شذوذ'], 'anomalies': ['anomaly', 'update', 'insertion', 'deletion'], 'functional': ['functional dependency', 'fd', 'اعتماد وظيفي'], 'dependency': ['functional dependency', 'partial dependency', 'transitive dependency', 'اعتماد'], '1nf': ['first normal form', 'atomic', 'single-valued'], '2nf': ['second normal form', 'partial dependency'], '3nf': ['third normal form', 'transitive dependency'], 'partial': ['partial functional dependency', '2nf', 'اعتماد جزئي'], 'transitive': ['transitive dependency', '3nf', 'اعتماد انتقالي'], 'atomic': ['1nf', 'single-valued', 'قيمة ذرية'], 'decomposition': ['decompose', 'relations', 'تفكيك']}

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
    print("Database Chapter 4 - Normalization RAG index built.")
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
    nq = normalize(question)
    slide = int(document["slide_number"])
    rules = [(['normalization', 'تطبيع'], {1, 2, 3, 6, 8}), (['anomaly', 'anomalies', 'update anomaly', 'insertion anomaly', 'deletion anomaly'], {5, 7}), (['functional dependency', 'اعتماد وظيفي'], {9, 10}), (['1nf', 'first normal form', 'atomic'], {11}), (['2nf', 'second normal form', 'partial'], {12, 13, 14}), (['3nf', 'third normal form', 'transitive'], {16, 17, 15})]
    boost = 0.0
    for keys, slides in rules:
        if any(normalize(k) in nq for k in keys) and slide in slides:
            boost += 20.0
    return boost

def retrieve(index: dict, question: str, top_k: int = 3):
    query = expanded_query(question)
    original = set(tokenize(question))
    scored = []
    for document in index["documents"]:
        score = bm25(index, query, document)
        title = set(tokenize(document.get("slide_title","")))
        headings = set(tokenize(" ".join(document.get("section_headings",[]))))
        score += 3.0 * len(original & title)
        score += 1.5 * len(original & headings)
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
        answer = generate_grounded_answer(question, results, course_name="Database Chapter 4 - Normalization")
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
    print("Database Chapter 4 - Normalization RAG is ready.")
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
    parser = argparse.ArgumentParser(description="Database Chapter 4 - Normalization educational RAG")
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
