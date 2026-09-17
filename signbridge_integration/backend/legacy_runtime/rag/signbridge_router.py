from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from avatar_sign_mapper import attach_avatar_plan

import query_understanding as qu
from signbridge_llm import (
    GROQ_MODEL,
    generate_grounded_response,
    get_avatar_text,
    to_json,
)


PROJECT = Path(__file__).resolve().parent

DOMAIN_FILES = {
    "stack": "stack_rag.py",
    "queue": "queue_rag.py",
    "pointers": "pointers_rag.py",
    "database_ch1": "01_Database_Chapter1_rag.py",
    "database_ch2": "01_Database_Chapter2_rag.py",
    "database_ch3": "01_Database_Chapter3_rag.py",
    "database_ch4": "01_Database_Chapter4_rag.py",
}

DOMAIN_LABELS = {
    "stack": "Stack",
    "queue": "Queue",
    "pointers": "Pointers",
    "database_ch1": "Database Chapter 1",
    "database_ch2": "Database Chapter 2 - E-R Model",
    "database_ch3": "Database Chapter 3 - ERD Mapping",
    "database_ch4": "Database Chapter 4 - Normalization",
}

ROUTING_KEYWORDS = {
    "stack": [
        "stack", "stacks", "مكدس", "المكدس", "push", "pop", "peek", "top", "lifo",
    ],
    "queue": [
        "queue", "queues", "طابور", "الطابور", "enqueue", "dequeue",
        "rear", "front", "fifo", "circular queue", "priority queue",
        "circular", "دائري",
    ],
    "pointers": [
        "pointer", "pointers", "مؤشر", "مؤشرات", "shallow copy", "deep copy",
        "dereference", "memory address", "dangling", "memory leak",
    ],
    "database_ch1": [
        "database", "dbms", "schema", "instance", "ddl", "dml", "sql",
        "database architecture", "قواعد البيانات", "قاعدة بيانات",
    ],
    "database_ch2": [
        "primary key", "primary keys", "key attribute",
        "مفتاح اساسي", "المفتاح الاساسي", "مفتاح أساسي", "المفتاح الأساسي",
        "entity", "relationship", "entity set", "relationship set",
        "cardinality", "participation", "total participation",
        "partial participation", "weak entity", "er model",
        "كيان", "علاقه", "مشاركه كليه", "مشاركه جزئيه",
    ],
    "database_ch3": [
        "mapping erd", "map erd", "erd to relational", "erd to relation",
        "many-to-many", "many to many", "one-to-many", "one to many",
        "one-to-one", "m:n", "1:n", "foreign key", "composite key",
        "تحويل erd", "مفتاح اجنبي",
    ],
    "database_ch4": [
        "normalization", "normal form", "1nf", "2nf", "3nf",
        "functional dependency", "partial dependency", "transitive dependency",
        "anomaly", "anomalies", "التطبيع", "اعتماديه وظيفيه",
        "اعتماد جزئي", "اعتماد انتقالي",
    ],
}


def normalize(text: str) -> str:
    """Delegates to the centralized normalizer (query_understanding.py),
    which additionally strips tatweel, Arabic/English punctuation, and
    hamza-on-carrier variants this function used to miss. Kept as a thin
    wrapper (same name/signature) so every existing caller in this file -
    is_stack_queue_comparison, retrieve_stack, retrieve_queue - keeps
    working unchanged.
    """
    return qu.normalize_text(text)


def load_module(domain: str):
    path = PROJECT / DOMAIN_FILES[domain]

    if not path.is_file():
        raise FileNotFoundError(f"Required RAG file not found: {path}")

    spec = importlib.util.spec_from_file_location(
        f"signbridge_domain_{domain}",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load domain module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_index(module) -> dict:
    index_path = Path(module.DEFAULT_INDEX)

    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    return module.build_index(
        Path(module.DEFAULT_CHUNKS),
        index_path,
    )


def is_stack_queue_comparison(question: str) -> bool:
    nq = normalize(question)

    has_stack = any(
        normalize(k) in nq
        for k in ["stack", "مكدس", "المكدس"]
    )
    has_queue = any(
        normalize(k) in nq
        for k in ["queue", "طابور", "الطابور"]
    )
    comparison = any(
        normalize(k) in nq
        for k in [
            "الفرق", "فرق", "مقارنه", "قارن",
            "difference", "compare", "vs", "versus"
        ]
    )

    return has_stack and has_queue and comparison


def route_domains(question: str) -> list[str]:
    """Routing now delegates the actual matching to query_understanding.py
    (centralized alias table, definite-article-aware token matching, and a
    conservative fuzzy-typo tier used only when nothing else matched) -
    ROUTING_KEYWORDS above is kept only for backward compatibility with any
    external caller that imports it directly; it is no longer read here.
    The existing safe behaviors are unchanged: a specific database chapter
    still displaces the generic database_ch1 bucket, and "no evidence for
    any domain" still falls back to every domain (never a blind single-
    domain guess) exactly as before.
    """
    matched, _confidence = qu.route_domains_with_confidence(question)

    specific_db = {
        "database_ch2",
        "database_ch3",
        "database_ch4",
    }.intersection(matched)

    if specific_db and "database_ch1" in matched:
        matched.remove("database_ch1")

    if not matched:
        matched = list(DOMAIN_FILES)

    return matched


def retrieve_stack(module, index: dict, question: str, top_k: int):
    query = module.expanded_query(question)
    normalized_question = module.normalize(question)
    question_terms = set(module.tokenize(question))

    definition_intent = (
        "ما هو" in normalized_question
        or "ما هي" in normalized_question
        or "what is" in normalized_question
    )

    stack_definition = (
        definition_intent
        and ("stack" in question_terms or "مكدس" in question_terms)
    )

    focus_terms = question_terms.intersection(
        {"push", "pop", "top", "isempty", "isfull", "peek", "constructor", "destructor"}
    )

    scored = []

    for document in index["documents"]:
        score = module.bm25(index, query, document)

        if (
            stack_definition
            and document["slide_number"] == 4
            and document["part_number"] == 1
        ):
            # Slide 4 ("Stack Definition and the LIFO Principle") is the
            # chapter's actual definition slide. Slide 1 ("Building a Stack
            # Using an Array") was verified by inspection to be an
            # array-implementation walkthrough, not a definition, so it must
            # not receive this boost (a prior version of this boost targeted
            # slide 1 by mistake, which is exactly the "domain-correct but
            # answer-irrelevant chunk outranks the real definition" failure
            # pattern this task's retrieval fix is about).
            score += 30

        heading_tokens = set(
            module.tokenize(" ".join(document.get("section_headings", [])))
        )

        if focus_terms.intersection(heading_tokens):
            score += 25

        if is_stack_queue_comparison(question):
            title_n = module.normalize(document.get("slide_title", ""))

            # Prefer definition/LIFO material for a cross-domain comparison.
            if document["slide_number"] in {1, 4}:
                score += 45

            if "lifo" in title_n or "definition" in title_n:
                score += 35

            # Avoid implementation-detail slides when the user only asks
            # for the conceptual difference between Stack and Queue.
            if "full" in title_n or "constructor" in title_n or "destructor" in title_n:
                score -= 30

        scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for item in scored if item[0] > 0][:top_k]


def retrieve_queue(module, index: dict, question: str, top_k: int):
    query = module.expanded_query(question)
    normalized_question = module.normalize(question)
    question_terms = set(module.tokenize(question))

    definition_intent = (
        "ما هو" in normalized_question
        or "ما هي" in normalized_question
        or "what is" in normalized_question
    )

    queue_definition = (
        definition_intent
        and ("queue" in question_terms or "طابور" in question_terms)
    )

    focus_terms = question_terms.intersection(
        {
            "addqueue", "enqueue", "deletequeue", "dequeue", "front", "back",
            "rear", "queuefront", "queuerear", "isemptyqueue", "isfullqueue",
            "initializequeue", "circular", "modulo", "priority", "priority_queue",
            "simulation", "server", "customer", "constructor", "destructor",
        }
    )

    scored = []

    for document in index["documents"]:
        score = module.bm25(index, query, document)
        title_normalized = module.normalize(document.get("slide_title", ""))
        heading_tokens = set(
            module.tokenize(" ".join(document.get("section_headings", [])))
        )

        if queue_definition and (
            document["slide_number"] == 1
            or "introduction" in title_normalized
        ):
            score += 30

        if focus_terms.intersection(heading_tokens):
            score += 25

        if ("addqueue" in question_terms or "enqueue" in question_terms):
            if document["slide_number"] == 18 or "add queue" in title_normalized:
                score += 35

        if ("deletequeue" in question_terms or "dequeue" in question_terms):
            if document["slide_number"] == 19 or "delete queue" in title_normalized:
                score += 35

        if "front" in question_terms and document["slide_number"] == 16:
            score += 35

        if ("back" in question_terms or "rear" in question_terms):
            if document["slide_number"] == 17:
                score += 35

        if (
            "isemptyqueue" in question_terms
            or "isfullqueue" in question_terms
            or "empty" in question_terms
            or "full" in question_terms
        ):
            if document["slide_number"] in {10, 11, 12, 14}:
                score += 25

        if (
            "circular" in question_terms
            or "دائري" in question_terms
            or "modulo" in question_terms
        ):
            if document["slide_number"] in {7, 8, 9, 10, 11, 12}:
                score += 30

        if ("priority" in question_terms or "priority_queue" in question_terms):
            if document["slide_number"] in {24, 25}:
                score += 35

        if is_stack_queue_comparison(question):
            # Prefer the introductory FIFO/front/rear explanation.
            if document["slide_number"] == 1:
                score += 55

            if any(term in title_normalized for term in ["introduction", "fifo", "queue"]):
                score += 20

            # Priority Queue is a different topic and should not displace
            # the normal Queue definition in a basic Stack-vs-Queue question.
            if "priority" in title_normalized:
                score -= 50

        scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for item in scored if item[0] > 0][:top_k]


def retrieve_domain(
    domain: str,
    question: str,
    top_k: int,
) -> list[tuple[float, dict]]:
    """The raw `question` still drives every existing intent heuristic
    (definition/comparison detection, focus-term boosts) unchanged. Only
    the text actually handed to each domain module's own
    tokenize/expanded_query/BM25 pipeline is enriched with that domain's
    canonical alias terms first (query_understanding.enrich_query_for_
    retrieval) - this is query expansion in front of the existing BM25,
    not a new retrieval architecture, and every domain module's own local
    EXPANSIONS dict still runs exactly as before on top of it.
    """

    module = load_module(domain)
    index = load_index(module)
    enriched_question = qu.enrich_query_for_retrieval(question, [domain])

    # Ask each domain's own BM25 pipeline for a wider candidate pool than we
    # actually need, so the intent-aware reranker below has real on-topic
    # alternatives to promote (e.g. an introductory definition chunk that
    # BM25 alone ranked 3rd or 4th) before truncating back down to top_k.
    # This never lowers the relevance floor: every domain module's own
    # retrieve() still only returns candidates it scored above zero.
    wide_k = max(top_k * 4, 8)

    if domain == "stack":
        results = retrieve_stack(module, index, enriched_question, wide_k)
    elif domain == "queue":
        results = retrieve_queue(module, index, enriched_question, wide_k)
    else:
        try:
            results = module.retrieve(index, enriched_question, wide_k)
        except TypeError:
            results = module.retrieve(index, enriched_question, top_k=wide_k)

    # Intent detection and the "Possible Student Questions" match both use
    # the raw, un-enriched question - same rule already followed by the
    # definition/comparison heuristics inside retrieve_stack/retrieve_queue.
    results = qu.rerank_by_intent(question, results)[:top_k]

    enriched = []

    for score, document in results:
        document = dict(document)
        document["domain"] = DOMAIN_LABELS[domain]
        enriched.append((float(score), document))

    return enriched


def collect_results(
    question: str,
    domains: list[str],
    per_domain_k: int = 2,
) -> list[tuple[float, dict]]:

    combined = []

    for domain in domains:
        combined.extend(
            retrieve_domain(
                domain,
                question,
                per_domain_k,
            )
        )

    return combined


def answer_question(
    question: str,
    per_domain_k: int = 2,
) -> dict[str, Any]:

    domains = route_domains(question)

    results = collect_results(
        question,
        domains,
        per_domain_k=per_domain_k,
    )

    if not results:
        raise RuntimeError("No relevant course content was retrieved.")

    course_name = " + ".join(DOMAIN_LABELS[d] for d in domains)

    response = generate_grounded_response(
        question=question,
        results=results,
        course_name=course_name,
    )

    response["routed_domains"] = [
        DOMAIN_LABELS[d]
        for d in domains
    ]

    # Stable alias for the avatar/frontend integration layer.
    # Keep simplified_text unchanged for the research pipeline.
    response["avatar_text"] = get_avatar_text(response)

    # Add canonical sign/motion integration fields for the avatar frontend.
    # This does not alter the grounded educational answer or simplified_text.
    attach_avatar_plan(response)

    return response


def print_readable(response: dict[str, Any]) -> None:
    print("\nRouted domains:", ", ".join(response["routed_domains"]))
    print("\nGeneration method:", f"Groq LLM ({GROQ_MODEL})")

    print("\nEDUCATIONAL ANSWER:")
    print(response["educational_answer"])

    print("\nSIGN-FRIENDLY ANSWER:")
    print(response["simplified_text"])

    print("\nGrounding sources:")
    for source in response["sources"]:
        print(
            f"- {source.get('domain', '')} | "
            f"{source['chunk_id']} | "
            f"Slide {source['slide_number']} | "
            f"{source['slide_title']} | "
            f"score {source['retrieval_score']:.3f}"
        )

    print("\nAvatar text:")
    print(response["avatar_text"])

    print("\nSign tokens:")
    print(response.get("sign_tokens", []))

    if response.get("unmapped_words"):
        print("\nUnmapped content words:")
        print(response["unmapped_words"])

    print("\nPlanner version:", response.get("planner_version", "unknown"))
    print("Semantic coverage:", response.get("semantic_coverage", response.get("coverage", 0.0)))
    print("Motion coverage:", response.get("motion_coverage", 0.0))
    if response.get("missing_concepts"):
        print("Missing concepts:", response["missing_concepts"])
    print("Avatar ready:", response.get("avatar_ready", False))
    if response.get("planner_warnings"):
        print("Planner warnings:", response["planner_warnings"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified SignBridge multi-domain RAG router"
    )

    parser.add_argument("--question", required=True)
    parser.add_argument("--per-domain-k", type=int, default=2)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    response = answer_question(
        question=args.question,
        per_domain_k=max(1, args.per_domain_k),
    )

    if args.json:
        print(to_json(response))
    else:
        print_readable(response)


if __name__ == "__main__":
    main()
