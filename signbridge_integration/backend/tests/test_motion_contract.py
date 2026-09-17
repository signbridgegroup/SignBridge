"""Lightweight resolver tests for the two distinct motion-resolution paths:

Path A — recognized video: recognized dataset + recognized token/gloss
    -> exact dataset motion entry (app.motions.manifest.resolve_recognized_token)
Path B — typed question: RAG/Groq -> semantic sign_tokens
    -> exact semantic-token bridge (app.motions.manifest.resolve_lexicon_token)

These must never be confused, and neither may use fuzzy string matching.
Isharah's source alignment was validated with per-sample DTW and its 674
available motions were generated.  Six source-unavailable glosses remain
honestly unavailable.  The resolver tests therefore require a known available
gloss to resolve to a real ready motion without changing the exact/fuzzy rules.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.motions.manifest import motion_manifest  # noqa: E402


def setUpModule() -> None:
    # motion_manifest is a process-wide singleton loaded once at first
    # import. When this module runs alongside test_manifest_correctness
    # (whose RebuildPreservesReadyTests shells out to `rebuild-manifest`),
    # the singleton must be explicitly refreshed rather than trusted to
    # reflect whatever is on disk by the time these tests execute.
    motion_manifest.reload()


class PathAJordanianITTests(unittest.TestCase):
    def test_recognized_it_label_resolves_to_its_own_motion(self) -> None:
        entry = motion_manifest.resolve_recognized_token("it", "Stack")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.dataset, "jordanian_it")
        self.assertEqual(entry.token, "STACK")
        self.assertEqual(entry.motion_file, "stack_07.motion.json")

    def test_recognized_it_label_is_normalized_exactly_not_fuzzily(self) -> None:
        # "Run time error" (as the recognizer emits it, raw dataset label)
        # must normalize deterministically to RUN_TIME_ERROR - not to
        # anything merely similar.
        entry = motion_manifest.resolve_recognized_token("it", "Run time error")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.token, "RUN_TIME_ERROR")

    def test_unrecognized_it_label_resolves_to_nothing_not_a_guess(self) -> None:
        entry = motion_manifest.resolve_recognized_token("it", "Not A Real Label")
        self.assertIsNone(entry)


class PathAKArSLTests(unittest.TestCase):
    def test_recognized_karsl_sign_id_resolves_to_its_own_motion(self) -> None:
        entry = motion_manifest.resolve_recognized_token("karsl", "0001")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.dataset, "karsl")
        self.assertEqual(entry.token, "0001")

    def test_recognized_karsl_vocabulary_token_form_also_resolves(self) -> None:
        # The checkpoint's continuous-vocabulary form is "KARSL_0001" -
        # must normalize to the same manifest entry as the bare sign_id.
        entry = motion_manifest.resolve_recognized_token("karsl", "KARSL_0001")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.token, "0001")

    def test_karsl_does_not_resolve_into_jordanian_it_namespace(self) -> None:
        # A KArSL sign_id that happens to collide with a Jordanian IT token
        # string must never cross into the other dataset's namespace.
        entry = motion_manifest.resolve_recognized_token("karsl", "0001")
        self.assertEqual(entry.dataset, "karsl")
        it_entry = motion_manifest.get("jordanian_it", "0001")
        self.assertIsNone(it_entry)


class PathAIsharahTests(unittest.TestCase):
    def test_recognized_gloss_resolves_to_ready_motion(self) -> None:
        # 'ا' is a real gloss verified to exist verbatim in both clips.csv
        # and the checkpoint's continuous vocabulary. Its source-group-00
        # sequence was aligned and generated through the approved Isharah
        # DTW/MediaPipe path, so it must now resolve as a real ready motion.
        entry = motion_manifest.get("isharah", "ا")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.status, "ready")
        self.assertEqual(entry.motion_file, "ا.motion.json")
        self.assertIsNotNone(motion_manifest.resolve_file("isharah", "ا"))

    def test_isharah_recognized_via_continuous_mode_namespace(self) -> None:
        # resolved_mode "continuous" must bridge to the "isharah" manifest
        # namespace, exactly, via resolve_recognized_token - not guessed.
        entry = motion_manifest.resolve_recognized_token("continuous", "ا")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.dataset, "isharah")

    def test_unknown_gloss_resolves_to_nothing(self) -> None:
        entry = motion_manifest.resolve_recognized_token("continuous", "كلمه_غير_موجوده_اطلاقا")
        self.assertIsNone(entry)


class PathBSemanticLexiconTests(unittest.TestCase):
    def test_stack_lexicon_token_bridges_to_stack_07(self) -> None:
        entry = motion_manifest.resolve_lexicon_token("STACK")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.motion_file, "stack_07.motion.json")

    def test_unbridged_lexicon_token_resolves_to_nothing(self) -> None:
        # PUSH exists in avatar_sign_lexicon.json but has no confirmed
        # motion bridge - must not silently guess one (e.g. reusing STACK's
        # motion just because PUSH is a stack operation).
        entry = motion_manifest.resolve_lexicon_token("PUSH")
        self.assertIsNone(entry)

    def test_lexicon_bridge_never_maps_to_the_forbidden_similarity_guesses(self) -> None:
        # Explicitly forbidden guesses from the brief: START->INITIALIZE,
        # RELATIONSHIP->BINARY_RELATIONSHIP, ERROR->RUN_TIME_ERROR. START and
        # RELATIONSHIP DO have legitimate bridges now (to their own exact
        # dataset match, found via the semantic-bridge recompute) - the
        # point of this test is that they never point at the specific
        # similarity-guessed target, not that they stay unbridged forever.
        start = motion_manifest.resolve_lexicon_token("START")
        self.assertIsNotNone(start)
        self.assertNotEqual((start.dataset, start.token), ("jordanian_it", "INITIALIZE"))

        relationship = motion_manifest.resolve_lexicon_token("RELATIONSHIP")
        self.assertIsNotNone(relationship)
        self.assertNotEqual((relationship.dataset, relationship.token), ("jordanian_it", "BINARY_RELATIONSHIP"))

        # ERROR itself is not even a lexicon token, and must not resolve.
        self.assertIsNone(motion_manifest.resolve_lexicon_token("ERROR"))


class PathAVersusPathBIndependenceTests(unittest.TestCase):
    def test_path_a_and_path_b_are_separate_functions(self) -> None:
        # Structural guard against the two paths being silently merged.
        self.assertIsNot(
            type(motion_manifest).resolve_lexicon_token,
            type(motion_manifest).resolve_recognized_token,
        )
        # A dataset-namespaced token (Path A) is never accidentally
        # resolvable through the Path B lexicon-bridge function.
        self.assertIsNone(motion_manifest.resolve_lexicon_token("0001"))

    def test_full_dataset_generation_does_not_invent_semantic_bridges(self) -> None:
        # The semantic-bridge recompute found exact matches for 30 of the 58
        # lexicon tokens (see docs/SEMANTIC_BRIDGE.md), still well under all
        # 58. Generating the full dataset motion library makes the three
        # already-confirmed Isharah bridges physically ready; it must NOT
        # invent mappings for the other 28 semantic tokens. This preserves
        # the "recognition coverage is not semantic coverage" distinction.
        from app.motions.manifest import LEXICON_TOKEN_BRIDGE
        self.assertEqual(len(LEXICON_TOKEN_BRIDGE), 30)
        self.assertLess(len(LEXICON_TOKEN_BRIDGE), 58)

        isharah_only = {"INCREMENT", "ATTRIBUTE", "READ"}
        for token in isharah_only:
            entry = motion_manifest.resolve_lexicon_token(token)
            self.assertIsNotNone(entry, f"{token} should have a bridge")
            self.assertEqual(entry.status, "ready")
            self.assertIsNotNone(motion_manifest.resolve_file(entry.dataset, entry.token))

        # Representative unbridged lexicon concepts remain unbridged even
        # though the physical dataset library now contains 1,341 motions.
        for token in ("PUSH", "POP", "CANDIDATE_KEY", "UNIQUE"):
            self.assertNotIn(token, LEXICON_TOKEN_BRIDGE)
            self.assertIsNone(motion_manifest.resolve_lexicon_token(token))


if __name__ == "__main__":
    unittest.main()
