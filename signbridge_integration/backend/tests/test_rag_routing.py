"""Offline RAG routing/retrieval/normalization tests.

No real Groq API call is made anywhere in this file. Routing and retrieval
(BM25 against the real course indexes) run for real and offline; the one
test that exercises the full answer_question() contract mocks the LLM call
with a canned in-memory response (see MockedLlmContractTest) - never a real
network call, never a real key.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BACKEND_DIR / "legacy_runtime" / "rag"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(RAG_DIR))

import query_understanding as qu  # noqa: E402
import signbridge_llm  # noqa: E402
import signbridge_router as router  # noqa: E402

# --------------------------------------------------------------------------- #
# Test corpus - every example from the task brief, plus explicit typo/
# incomplete/ambiguous/unrelated categories. (question, expected_domains,
# category). expected_domains is a set of acceptable router.route_domains()
# outputs that must be a SUBSET of what's returned (multi-domain routing is
# allowed to include more than the minimum, per the "preserve multi-domain
# routing" requirement) - except for the "unrelated" category, where the
# assertion is instead that the result is the full-fallback (never a
# confident wrong single domain).
# --------------------------------------------------------------------------- #

FORMAL_ARABIC = [
    ("ما هي قاعدة البيانات؟", {"database_ch1"}),
    ("ما هو المكدس؟", {"stack"}),
    ("ما هو الطابور؟", {"queue"}),
    ("ما هو المؤشر؟", {"pointers"}),
    ("ما هو المفتاح الاساسي؟", {"database_ch2"}),
    ("شو الفرق بين entity و attribute؟", {"database_ch2"}),
]

COLLOQUIAL_ARABIC = [
    ("شو يعني داتا بيس؟", {"database_ch1"}),
    ("اشرحلي الداتابيس", {"database_ch1"}),
    ("شو الداتا بس؟", {"database_ch1"}),
    ("قاعده البيانات شو هي؟", {"database_ch1"}),
    ("شو يعني ستاك؟", {"stack"}),
    ("اشرحلي الستاك", {"stack"}),
    ("ستك شو هو؟", {"stack"}),
    ("كيف اعرف اذا المكدس فاضي؟", {"stack"}),
    ("شو يعني كيو؟", {"queue"}),
    ("كيو فاضي كيف افحصه؟", {"queue"}),
    ("شو يعني بوينتر؟", {"pointers"}),
    ("كيف بصير memory leak؟", {"pointers"}),
]

ENGLISH = [
    ("what is a database?", {"database_ch1"}),
    ("explain DB", {"database_ch1"}),
    ("what does DBMS mean?", {"database_ch1"}),
    ("what is a stack?", {"stack"}),
    ("what is LIFO?", {"stack"}),
    ("what is FIFO?", {"queue"}),
    ("what is a pointer?", {"pointers"}),
    ("stack empty check", {"stack"}),
]

MIXED_LANGUAGE = [
    ("شو الفرق بين schema و instance؟", {"database_ch1"}),
    ("اشرح primary key", {"database_ch2"}),
    ("شو يعني normalization؟", {"database_ch4"}),
    ("شو يعني push؟", {"stack"}),
    ("ما الفرق بين push و pop؟", {"stack"}),
    ("اشرح queue", {"queue"}),
    ("ما الفرق بين enqueue و dequeue؟", {"queue"}),
    ("اشرح circular queue", {"queue"}),
    ("اشرح pointer", {"pointers"}),
    ("ما هو memory address؟", {"pointers"}),
    ("شو يعني dereference؟", {"pointers"}),
    ("ما الفرق بين shallow copy و deep copy؟", {"pointers"}),
    ("شو يعني dangling pointer؟", {"pointers"}),
]

TRANSLITERATION = [
    ("db vs dbms", {"database_ch1"}),
    ("اشرح التطبيع", {"database_ch4"}),
]

# One missing character from a real keyword.
MISSING_CHARACTER_TYPOS = [
    ("queu what is it?", {"queue"}),        # "queue" -> "queu"
    ("pointr meaning", {"pointers"}),        # "pointer" -> "pointr"
    ("ما هو المكدس", {"stack"}),             # no question mark, not a spelling typo but tests punctuation robustness
]

# One extra character inserted into a real keyword.
EXTRA_CHARACTER_TYPOS = [
    ("whatt is a stackk?", {"stack"}),
    ("daatabase explain", {"database_ch1"}),
]

# One substituted character in a real keyword.
SUBSTITUTED_CHARACTER_TYPOS = [
    ("pointar meaning", {"pointers"}),   # e -> a
    ("qeue definition", {"queue"}),       # u -> e swapped position (still close)
]

INCOMPLETE_QUESTIONS = [
    ("ستاك", {"stack"}),
    ("queue", {"queue"}),
    ("مؤشر", {"pointers"}),
    ("primary key", {"database_ch2"}),
]

# Contains real evidence for two domains at once - multi-domain routing must
# be preserved (both/either acceptable), never collapsed to one arbitrarily.
AMBIGUOUS_QUESTIONS = [
    ("ما الفرق بين stack و queue؟", {"stack", "queue"}),
]

# No real evidence for any domain: must never receive a confident single
# wrong-domain match. The router's existing safe behavior is to fall back to
# every domain (multi-domain retrieval), not a blind guess.
UNRELATED_QUESTIONS = [
    "what is the weather today?",
    "من هو رئيس الوزراء؟",
    "how do I bake a cake?",
]

# The confirmed regression case from the task brief - kept as its own,
# explicitly named test as well as part of COLLOQUIAL_ARABIC above.
REGRESSION_CASE = "ما هي الداتا بيس؟"


class NormalizationUnitTests(unittest.TestCase):
    def test_diacritics_stripped(self) -> None:
        self.assertEqual(qu.normalize_text("مُكَدَّس"), qu.normalize_text("مكدس"))

    def test_tatweel_stripped(self) -> None:
        self.assertEqual(qu.normalize_text("مـــكدس"), qu.normalize_text("مكدس"))

    def test_alef_forms_unified(self) -> None:
        self.assertEqual(qu.normalize_text("أشرح"), qu.normalize_text("اشرح"))

    def test_ya_alef_maqsura_unified(self) -> None:
        self.assertEqual(qu.normalize_text("متى"), qu.normalize_text("متي"))

    def test_taa_marbuta_haa_unified(self) -> None:
        self.assertEqual(qu.normalize_text("قاعدة"), qu.normalize_text("قاعده"))

    def test_punctuation_stripped(self) -> None:
        self.assertEqual(qu.normalize_text("ما هو المكدس؟"), qu.normalize_text("ما هو المكدس"))

    def test_repeated_whitespace_collapsed(self) -> None:
        self.assertEqual(qu.normalize_text("ما   هو    المكدس"), qu.normalize_text("ما هو المكدس"))

    def test_case_insensitive_english(self) -> None:
        self.assertEqual(qu.normalize_text("STACK"), qu.normalize_text("stack"))

    def test_definite_article_stripped_for_matching(self) -> None:
        self.assertEqual(qu.strip_definite_article("البيانات"), "بيانات")
        # A short word that merely starts with the same two letters, where
        # the remainder would be too short to be a real word, is left
        # unmangled rather than aggressively stripped.
        self.assertEqual(qu.strip_definite_article("الف"), "الف")

    def test_fuzzy_does_not_match_unrelated_short_word(self) -> None:
        # A short, common, unrelated word must never fuzzy-match into a
        # domain - this is the conservative-tolerance safety requirement.
        matches = qu.fuzzy_domain_matches("هل")
        self.assertEqual(matches, set())

    def test_fuzzy_requires_close_similarity(self) -> None:
        # "banana" is not close to anything in the alias vocabulary.
        matches = qu.fuzzy_domain_matches("banana")
        self.assertEqual(matches, set())


class RoutingCorpusTests(unittest.TestCase):
    def _assert_corpus(self, corpus: list[tuple[str, set[str]]]) -> None:
        for question, expected in corpus:
            with self.subTest(question=question):
                matched = set(router.route_domains(question))
                self.assertTrue(
                    expected.issubset(matched),
                    f"{question!r}: expected {expected} to be a subset of {matched}",
                )

    def test_formal_arabic(self) -> None:
        self._assert_corpus(FORMAL_ARABIC)

    def test_colloquial_arabic(self) -> None:
        self._assert_corpus(COLLOQUIAL_ARABIC)

    def test_english(self) -> None:
        self._assert_corpus(ENGLISH)

    def test_mixed_language(self) -> None:
        self._assert_corpus(MIXED_LANGUAGE)

    def test_transliteration(self) -> None:
        self._assert_corpus(TRANSLITERATION)

    def test_missing_character_typos(self) -> None:
        self._assert_corpus(MISSING_CHARACTER_TYPOS)

    def test_extra_character_typos(self) -> None:
        self._assert_corpus(EXTRA_CHARACTER_TYPOS)

    def test_substituted_character_typos(self) -> None:
        self._assert_corpus(SUBSTITUTED_CHARACTER_TYPOS)

    def test_incomplete_questions(self) -> None:
        self._assert_corpus(INCOMPLETE_QUESTIONS)

    def test_ambiguous_question_preserves_multi_domain(self) -> None:
        for question, expected in AMBIGUOUS_QUESTIONS:
            with self.subTest(question=question):
                matched = set(router.route_domains(question))
                # At least one of the two evidenced domains must be present;
                # a healthy multi-domain match commonly includes both.
                self.assertTrue(matched & expected, f"{question!r} -> {matched}")

    def test_unrelated_questions_never_get_a_confident_wrong_domain(self) -> None:
        for question in UNRELATED_QUESTIONS:
            with self.subTest(question=question):
                matched, confidence = qu.route_domains_with_confidence(question)
                # No specific-domain evidence should ever be found for these.
                self.assertEqual(confidence, "fallback_all", f"{question!r} -> {matched} ({confidence})")
                # router.route_domains() then applies the existing safe
                # fallback: every domain, not a blind single guess.
                full_route = router.route_domains(question)
                self.assertEqual(set(full_route), set(router.DOMAIN_FILES))

    def test_regression_case_routes_to_database(self) -> None:
        matched = set(router.route_domains(REGRESSION_CASE))
        self.assertIn("database_ch1", matched)
        self.assertNotEqual(set(router.DOMAIN_FILES), matched, "must not silently fall back to every domain")


class RetrievalSmokeTests(unittest.TestCase):
    """Confirms retrieval actually returns non-empty, on-topic results -
    not just that routing picked the right domain label.

    NOTE: this class alone is intentionally NOT considered sufficient
    coverage any more. A real browser test found that a question could pass
    every check here (non-empty results, correct domain label) while still
    receiving only off-target sources ("Database Administrator", "Database
    Applications Examples (Cont.)") for a generic definition question,
    causing Groq's grounding rules to correctly refuse to answer. See
    AnswerSupportRetrievalTests below for the stronger, content-aware checks
    added to catch exactly that failure.
    """

    def _assert_relevant(self, question: str, expected_domain_label_substring: str) -> None:
        domains = router.route_domains(question)
        results = router.collect_results(question, domains, per_domain_k=2)
        self.assertTrue(results, f"No retrieval results for {question!r}")
        domains_seen = {doc.get("domain", "") for _score, doc in results}
        self.assertTrue(
            any(expected_domain_label_substring in d for d in domains_seen),
            f"{question!r}: expected a source with domain containing "
            f"{expected_domain_label_substring!r}, got {domains_seen}",
        )

    def test_regression_case_retrieves_database_content(self) -> None:
        self._assert_relevant(REGRESSION_CASE, "Database")

    def test_colloquial_stack_retrieves_stack_content(self) -> None:
        self._assert_relevant("شو يعني ستاك؟", "Stack")

    def test_typo_queue_retrieves_queue_content(self) -> None:
        self._assert_relevant("queu what is it?", "Queue")

    def test_typo_pointer_retrieves_pointer_content(self) -> None:
        self._assert_relevant("pointr meaning", "Pointer")


# The 11 exact Database definition-question variants from the task brief -
# every one of these must retrieve a chunk that actually answers "what is a
# database", not merely a chunk that belongs to the Database domain.
DATABASE_DEFINITION_QUERIES = [
    "ما هي قاعدة البيانات؟",
    "ما هي الداتا بيس؟",
    "ما هي الداتابيس؟",
    "شو يعني داتا بيس؟",
    "شو الداتا بس؟",
    "عرف قاعدة البيانات",
    "اشرحلي الداتابيس",
    "what is a database?",
    "define database",
    "explain DB",
    "what is DBMS?",
]

# The exact real-world failure: a title match on one of these, alone in the
# top results, means retrieval found a domain-correct but answer-irrelevant
# chunk rather than an actual definition.
_OFF_TARGET_DEFINITION_TITLES = {"database administrator", "database applications examples"}

# Words that show the retrieved context actually explains what a database
# is (store/organize/manage/retrieve data) rather than merely mentioning
# the word "database" in passing.
_DATABASE_DEFINITION_CONCEPT_MARKERS = [
    "تخزين", "تنظيم", "استرجاع", "ادار",  # ادار matches ادارة/إدارة after ة->ه normalization
    "store", "organiz", "manage", "retriev",
]


class AnswerSupportRetrievalTests(unittest.TestCase):
    """Verifies retrieved context actually contains the concepts needed to
    answer each foundational question - not just that a result came back
    from the right domain. This is the direct regression test for the real
    browser failure: "ما هي الداتابيس" routed correctly to Database Chapter
    1, but the only retrieved sources were "Database Administrator" and
    "Database Applications Examples (Cont.)", neither of which defines a
    database, so Groq's (correct, unweakened) grounding rules refused to
    answer.
    """

    def _collect(self, question: str, per_domain_k: int = 3):
        domains = router.route_domains(question)
        results = router.collect_results(question, domains, per_domain_k=per_domain_k)
        self.assertTrue(results, f"No retrieval results for {question!r}")
        return results

    def test_database_definition_queries_are_answer_supporting(self) -> None:
        for question in DATABASE_DEFINITION_QUERIES:
            with self.subTest(question=question):
                results = self._collect(question)

                top_titles = [qu.normalize_text(doc.get("slide_title", "")) for _score, doc in results[:2]]
                self.assertFalse(
                    all(any(bad in title for bad in _OFF_TARGET_DEFINITION_TITLES) for title in top_titles),
                    f"{question!r}: top sources are only off-target slides {top_titles} - "
                    "this is the exact real-world failure this test guards against.",
                )

                combined_text = qu.normalize_text(
                    " ".join(str(doc.get("text", "")) for _score, doc in results[:3])
                )
                self.assertTrue(
                    any(marker in combined_text for marker in _DATABASE_DEFINITION_CONCEPT_MARKERS),
                    f"{question!r}: retrieved context contains no direct database definition and no "
                    "explicit store/organize/manage/retrieve description of data.",
                )

                top_chunk_id = results[0][1]["chunk_id"]
                self.assertTrue(
                    top_chunk_id.startswith("database-ch1-slide-01-part"),
                    f"{question!r}: expected the grounded supplemental foundational chunk to rank "
                    f"first, got {top_chunk_id!r}.",
                )

    def test_stack_definition_question_is_answer_supporting(self) -> None:
        results = self._collect("ما هو المكدس؟")
        top_chunk_id = results[0][1]["chunk_id"]
        self.assertTrue(
            top_chunk_id.startswith("stacks-slide-04"),
            f"expected Stack's actual definition slide (4: LIFO principle) to rank first, "
            f"got {top_chunk_id!r} - slide 1 is an array-implementation walkthrough, not a definition.",
        )
        combined_text = qu.normalize_text(" ".join(str(doc.get("text", "")) for _score, doc in results[:3]))
        self.assertIn("lifo", combined_text)

    def test_queue_definition_question_is_answer_supporting(self) -> None:
        results = self._collect("شو يعني كيو؟")
        combined_text = qu.normalize_text(" ".join(str(doc.get("text", "")) for _score, doc in results[:3]))
        self.assertTrue("fifo" in combined_text or "طابور" in combined_text)

    def test_pointer_definition_question_is_answer_supporting(self) -> None:
        results = self._collect("ما هو البوينتر؟")
        top_chunk_id = results[0][1]["chunk_id"]
        self.assertEqual(
            top_chunk_id,
            "pointer_slide_005_part_01",
            "expected Pointers' actual definition slide (5: Pointer Data Type and Pointer "
            f"Variables) to rank first, got {top_chunk_id!r}.",
        )
        combined_text = qu.normalize_text(" ".join(str(doc.get("text", "")) for _score, doc in results[:3]))
        self.assertIn("عنوان", combined_text)  # "address" - a pointer stores a memory address

    def test_ambiguous_stack_vs_queue_retrieves_both_domains(self) -> None:
        question = "ما الفرق بين stack و queue؟"
        results = self._collect(question)
        domains_seen = {doc.get("domain", "") for _score, doc in results}
        self.assertIn("Stack", domains_seen, f"{question!r}: Stack content missing from {domains_seen}")
        self.assertIn("Queue", domains_seen, f"{question!r}: Queue content missing from {domains_seen}")


class GroundingRefusalSafetyTests(unittest.TestCase):
    """Guards Section 1's requirement ('do not weaken Groq grounding') as an
    automated regression: confirms the strict grounding system prompt is
    still in place, and that unrelated questions still receive no confident
    single-domain guess (so they can only ever be answered from the safe
    multi-domain fallback, which the strict grounding prompt above is then
    relied on to correctly refuse rather than guess from arbitrary chunks).
    """

    def test_strict_grounding_rules_are_present_and_not_weakened(self) -> None:
        import inspect

        source = inspect.getsource(signbridge_llm.generate_grounded_response)
        for required_phrase in (
            "STRICT GROUNDING RULES",
            "Use ONLY the retrieved course context",
            "Do not use outside knowledge",
            "Do not invent definitions",
            "If the context is insufficient, say so clearly instead of guessing",
        ):
            with self.subTest(required_phrase=required_phrase):
                self.assertIn(required_phrase, source)

    def test_unrelated_questions_have_no_confident_single_domain(self) -> None:
        for question in UNRELATED_QUESTIONS:
            with self.subTest(question=question):
                _matched, confidence = qu.route_domains_with_confidence(question)
                self.assertEqual(confidence, "fallback_all", f"{question!r} -> {confidence}")


class MockedLlmContractTest(unittest.TestCase):
    """Exercises the real production path (app.rag.service.answer_question,
    real routing, real BM25 retrieval) end-to-end, mocking only the actual
    Groq network call, to prove the API response contract (every field
    listed in the task brief) is unchanged. No real network call, no real
    API key - groq_api_key is set to an obviously-fake placeholder string
    only so the existing "is a key configured" check passes; the function
    that would actually call the network is replaced entirely.
    """

    def test_answer_response_contract_fields_unchanged(self) -> None:
        import dataclasses

        from app.rag import service as rag_service

        canned_llm_response = {
            "educational_answer": "قاعدة البيانات هي مجموعة بيانات مترابطة يتم تخزينها وإدارتها.",
            "simplified_text": "قاعدة البيانات تخزن البيانات.",
            "sources": [
                {
                    "rank": 1,
                    "chunk_id": "mock_chunk_1",
                    "slide_number": 1,
                    "slide_title": "Mock Slide",
                    "retrieval_score": 1.23,
                }
            ],
        }

        # Settings is a frozen dataclass, so the binding is replaced with a
        # copy carrying an obviously-fake placeholder key (never read from
        # any real source, never sent anywhere - the function that would
        # actually call Groq is replaced below) rather than mutating a
        # field in place.
        fake_settings = dataclasses.replace(
            rag_service.settings, groq_api_key="unit-test-placeholder-not-a-real-key"
        )

        with patch("app.rag.service.settings", fake_settings), \
             patch("signbridge_router.generate_grounded_response", return_value=canned_llm_response):
            response = rag_service.answer_question(REGRESSION_CASE, per_domain_k=2)

        for field in (
            "educational_answer",
            "simplified_text",
            "avatar_text",
            "routed_subject",
            "sources",
            "sign_tokens",
            "resolved_motion_sequence",
            "missing_motion_tokens",
            "semantic_coverage",
            "physical_motion_coverage",
            "avatar_ready",
            "warnings",
        ):
            with self.subTest(field=field):
                self.assertTrue(hasattr(response, field), f"AnswerResponse is missing field {field!r}")

        self.assertEqual(response.educational_answer, canned_llm_response["educational_answer"])
        self.assertIn("Database", " ".join(response.routed_subject))
        self.assertTrue(response.educational_answer_available)


if __name__ == "__main__":
    unittest.main()
