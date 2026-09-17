"""Comprehensive, machine-readable RAG robustness report.

Runs a broad query corpus (every authoritative subject/concept in the
indexed corpus, across formal Arabic, colloquial Arabic, English, code-
switched, transliterated, misspelled, and alternate-structure phrasings,
plus definition/operation/comparison forms and hallucination-prevention
negatives) against the REAL router and REAL BM25 retrieval - no mocking of
routing or retrieval. Groq itself is not called here (this checks retrieval
quality, which is what determines whether Groq *can* answer, not whether
the network call succeeds) - see the live-API validation step for that.

Produces a JSON report with: total cases, passed, failed, pass rate,
failures grouped by reason, per-subject results, per-language-form results,
and the exact failed queries - not just "the API returned HTTP 200".

Usage:
    python backend/tests/rag_test_report.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BACKEND_DIR / "legacy_runtime" / "rag"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(RAG_DIR))

import query_understanding as qu  # noqa: E402
import signbridge_router as router  # noqa: E402


@dataclass
class Case:
    query: str
    subject: str  # domain key, or "unrelated" for negative cases
    language_form: str
    intent: str
    concept_markers: list[str] = field(default_factory=list)  # any-of, checked in retrieved text
    require_top_chunk_prefix: str | None = None  # optional: top result chunk_id must start with this


# --------------------------------------------------------------------------- #
# Corpus - covers every wired domain (stack, queue, pointers, database
# chapters 1-4) with multiple phrasing styles per concept, plus negative
# (unrelated) cases for hallucination-prevention.
# --------------------------------------------------------------------------- #

DB_DEFINITION_MARKERS = ["تخزين", "تنظيم", "استرجاع", "ادار", "store", "organiz", "manage", "retriev"]
STACK_MARKERS = ["lifo", "push", "pop", "مكدس", "top"]
QUEUE_MARKERS = ["fifo", "طابور", "rear", "front"]
POINTER_MARKERS = ["عنوان", "address", "مؤشر", "pointer"]
PRIMARY_KEY_MARKERS = ["مفتاح", "key", "primary"]
ER_MODEL_MARKERS = ["entity", "كيان", "relationship", "علاقه", "er"]
SCHEMA_INSTANCE_MARKERS = ["schema", "instance", "مخطط", "حاله"]
NORMALIZATION_MARKERS = ["normalization", "تطبيع", "normal form"]

CASES: list[Case] = [
    # --- Database Chapter 1: database definition ------------------------- #
    Case("ما هي قاعدة البيانات؟", "database_ch1", "formal_arabic", "definition", DB_DEFINITION_MARKERS, "database-ch1-slide-01-part"),
    Case("شو هي قاعدة البيانات؟", "database_ch1", "colloquial_arabic", "definition", DB_DEFINITION_MARKERS),
    Case("ما هي الداتا بيس؟", "database_ch1", "transliteration", "definition", DB_DEFINITION_MARKERS, "database-ch1-slide-01-part"),
    Case("ما هي الداتابيس؟", "database_ch1", "transliteration", "definition", DB_DEFINITION_MARKERS, "database-ch1-slide-01-part"),
    Case("شو يعني داتابيس؟", "database_ch1", "colloquial_arabic", "definition", DB_DEFINITION_MARKERS),
    Case("what is a database?", "database_ch1", "english", "definition", DB_DEFINITION_MARKERS, "database-ch1-slide-01-part"),
    Case("define database", "database_ch1", "english", "definition", DB_DEFINITION_MARKERS),
    Case("database meaning", "database_ch1", "english", "definition", DB_DEFINITION_MARKERS),
    Case("داتا بيس شو هي", "database_ch1", "colloquial_arabic", "alternate_structure", DB_DEFINITION_MARKERS),
    Case("دااتابيس", "database_ch1", "typo", "definition", DB_DEFINITION_MARKERS),  # extra character
    Case("ما هي الدايتابيس", "database_ch1", "typo", "definition", DB_DEFINITION_MARKERS),  # substituted/extra
    # --- Stack ------------------------------------------------------------ #
    Case("ما هو الستاك؟", "stack", "formal_arabic", "definition", STACK_MARKERS),
    Case("ما هو المكدس؟", "stack", "formal_arabic", "definition", STACK_MARKERS, "stacks-slide-04"),
    Case("شو يعني stack؟", "stack", "code_switch", "definition", STACK_MARKERS),
    Case("اشرحلي المكدس", "stack", "colloquial_arabic", "definition", STACK_MARKERS),
    Case("كيف بشتغل ال stack؟", "stack", "code_switch", "operation", STACK_MARKERS),
    Case("what is a stack?", "stack", "english", "definition", STACK_MARKERS),
    Case("shu ya3ni stak", "stack", "transliteration", "definition", []),  # loose Latin transliteration, best-effort
    Case("ما الفرق بين push و pop؟", "stack", "code_switch", "comparison", ["push", "pop"]),
    Case("شو بعمل push؟", "stack", "colloquial_arabic", "operation", ["push"]),
    Case("شو بعمل pop؟", "stack", "colloquial_arabic", "operation", ["pop"]),
    Case("ستاك", "stack", "incomplete", "definition", []),
    Case("staack", "stack", "typo", "definition", []),
    # --- Queue -------------------------------------------------------------#
    Case("ما هو ال queue؟", "queue", "code_switch", "definition", QUEUE_MARKERS),
    Case("اشرح الطابور", "queue", "formal_arabic", "definition", QUEUE_MARKERS),
    Case("شو يعني طابور؟", "queue", "colloquial_arabic", "definition", QUEUE_MARKERS),
    Case("what is a queue?", "queue", "english", "definition", QUEUE_MARKERS),
    Case("ما الفرق بين enqueue و dequeue؟", "queue", "code_switch", "comparison", ["enqueue", "dequeue"]),
    Case("queu what is it?", "queue", "typo", "definition", []),
    # --- Pointers ---------------------------------------------------------- #
    Case("ما هو pointer؟", "pointers", "code_switch", "definition", POINTER_MARKERS),
    Case("شو يعني بوينتر؟", "pointers", "transliteration", "definition", POINTER_MARKERS, "pointer_slide_005"),
    Case("اشرح المؤشر", "pointers", "formal_arabic", "definition", POINTER_MARKERS),
    Case("what is a pointer?", "pointers", "english", "definition", POINTER_MARKERS, "pointer_slide_005"),
    Case("ما الفرق بين shallow copy و deep copy؟", "pointers", "code_switch", "comparison", ["shallow", "deep"]),
    Case("pointr meaning", "pointers", "typo", "definition", []),
    # --- Database Chapter 2 (Primary Key / E-R Model) ---------------------- #
    Case("ما هو primary key؟", "database_ch2", "code_switch", "definition", PRIMARY_KEY_MARKERS),
    Case("شو يعني المفتاح الأساسي؟", "database_ch2", "formal_arabic", "definition", PRIMARY_KEY_MARKERS),
    Case("ما هو ال E-R Model؟", "database_ch2", "code_switch", "definition", ER_MODEL_MARKERS),
    Case("اشرح entity", "database_ch2", "code_switch", "definition", ER_MODEL_MARKERS),
    # --- Database Chapter 4 (Normalization) --------------------------------#
    Case("شو يعني normalization؟", "database_ch4", "code_switch", "definition", NORMALIZATION_MARKERS),
    Case("اشرح التطبيع", "database_ch4", "formal_arabic", "definition", NORMALIZATION_MARKERS),
    # --- Cross-domain comparison -------------------------------------------#
    Case("ما الفرق بين schema و instance؟", "database_ch1", "code_switch", "comparison", SCHEMA_INSTANCE_MARKERS),
    Case("ما الفرق بين stack و queue؟", "stack+queue", "code_switch", "comparison", STACK_MARKERS + QUEUE_MARKERS),
    # --- Hallucination-prevention negatives (answer must not exist here) ---#
    Case("ما هي عاصمة فرنسا؟", "unrelated", "formal_arabic", "unrelated", []),
    Case("what is the weather today?", "unrelated", "english", "unrelated", []),
    Case("how do I bake a cake?", "unrelated", "english", "unrelated", []),
    Case("من هو رئيس الوزراء؟", "unrelated", "formal_arabic", "unrelated", []),
    Case("what is the capital of Japan?", "unrelated", "english", "unrelated", []),
]


def _run_case(case: Case) -> dict[str, Any]:
    domains = router.route_domains(case.query)

    if case.subject == "unrelated":
        _matched, confidence = qu.route_domains_with_confidence(case.query)
        passed = confidence == "fallback_all"
        return {
            "query": case.query,
            "subject": case.subject,
            "language_form": case.language_form,
            "intent": case.intent,
            "passed": passed,
            "reason": None if passed else f"expected fallback_all routing, got confidence={confidence!r}",
            "routed_domains": domains,
        }

    expected_domains = set(case.subject.split("+"))
    results = router.collect_results(case.query, domains, per_domain_k=3)

    if not results:
        return {
            "query": case.query,
            "subject": case.subject,
            "language_form": case.language_form,
            "intent": case.intent,
            "passed": False,
            "reason": "no retrieval results",
            "routed_domains": domains,
        }

    domain_labels = {router.DOMAIN_LABELS[d] for d in expected_domains if d in router.DOMAIN_LABELS}
    domains_seen = {doc.get("domain", "") for _score, doc in results}
    domain_ok = bool(domain_labels & domains_seen) if domain_labels else True

    combined_text = qu.normalize_text(" ".join(str(doc.get("text", "")) for _score, doc in results[:3]))
    concept_ok = True
    if case.concept_markers:
        concept_ok = any(qu.normalize_text(marker) in combined_text for marker in case.concept_markers)

    top_chunk_ok = True
    if case.require_top_chunk_prefix:
        top_chunk_ok = results[0][1]["chunk_id"].startswith(case.require_top_chunk_prefix)

    passed = domain_ok and concept_ok and top_chunk_ok
    reasons = []
    if not domain_ok:
        reasons.append(f"expected domain(s) {domain_labels}, retrieved domains {domains_seen}")
    if not concept_ok:
        reasons.append(f"no expected concept marker found in retrieved text: {case.concept_markers}")
    if not top_chunk_ok:
        reasons.append(
            f"expected top chunk to start with {case.require_top_chunk_prefix!r}, "
            f"got {results[0][1]['chunk_id']!r}"
        )

    return {
        "query": case.query,
        "subject": case.subject,
        "language_form": case.language_form,
        "intent": case.intent,
        "passed": passed,
        "reason": "; ".join(reasons) if reasons else None,
        "routed_domains": domains,
        "top_sources": [doc["chunk_id"] for _score, doc in results[:3]],
    }


def generate_report() -> dict[str, Any]:
    case_results = [_run_case(case) for case in CASES]

    total = len(case_results)
    passed = sum(1 for r in case_results if r["passed"])
    failed = total - passed

    failures_by_reason: Counter[str] = Counter()
    for r in case_results:
        if not r["passed"]:
            failures_by_reason[r["reason"] or "unknown"] += 1

    per_subject: dict[str, dict[str, int]] = {}
    for r in case_results:
        bucket = per_subject.setdefault(r["subject"], {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(r["passed"])

    per_language_form: dict[str, dict[str, int]] = {}
    for r in case_results:
        bucket = per_language_form.setdefault(r["language_form"], {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(r["passed"])

    return {
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "failures_by_reason": dict(failures_by_reason),
        "per_subject": per_subject,
        "per_language_form": per_language_form,
        "failed_queries": [
            {"query": r["query"], "subject": r["subject"], "reason": r["reason"]}
            for r in case_results
            if not r["passed"]
        ],
        "all_results": case_results,
    }


def main() -> None:
    report = generate_report()
    output_path = Path(__file__).resolve().parent / "rag_test_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total cases: {report['total_cases']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    print(f"Pass rate: {report['pass_rate'] * 100:.1f}%")
    print("\nPer subject:")
    for subject, counts in report["per_subject"].items():
        print(f"  {subject}: {counts['passed']}/{counts['total']}")
    print("\nPer language form:")
    for form, counts in report["per_language_form"].items():
        print(f"  {form}: {counts['passed']}/{counts['total']}")
    if report["failed_queries"]:
        print("\nFailed queries:")
        for failure in report["failed_queries"]:
            print(f"  - {failure['query']!r} ({failure['subject']}): {failure['reason']}")
    print(f"\nFull report written to: {output_path}")


if __name__ == "__main__":
    main()
