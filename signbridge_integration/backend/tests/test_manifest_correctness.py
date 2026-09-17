"""Automated checks for the corrective-verification requirements:

1. Jordanian IT has exactly 165 unique dataset tokens.
2. KArSL has exactly 502 unique dataset tokens.
3. Isharah has exactly 680 unique dataset tokens.
4. The total is exactly 1347.
5. No dataset contains duplicate tokens.
6. Every ready entry points to a real readable motion JSON.
7. Every ready motion JSON has schema signbridge-motion-v1.
8. Rebuilding the manifest does not lose ready statuses for ARRAY or KArSL 0001.
9. STACK points only to stack_07.motion.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = INTEGRATION_ROOT / "backend"
BUILDER_DIR = INTEGRATION_ROOT / "tools" / "motion_library_builder"
MANIFEST_PATH = INTEGRATION_ROOT / "data_manifests" / "motion_manifest.json"
BASELINE_MOTIONS_DIR = INTEGRATION_ROOT / "frontend" / "public" / "motions"
MOTION_LIBRARY_ROOT = Path(r"C:\SignBridge_Project\data\processed\avatar_motion_library")

sys.path.insert(0, str(BUILDER_DIR))
sys.path.insert(0, str(BACKEND_DIR))


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def resolve_ready_file(entry: dict) -> Path | None:
    if entry["status"] != "ready" or not entry.get("motion_file"):
        return None
    candidate = BASELINE_MOTIONS_DIR / entry["motion_file"]
    if candidate.is_file():
        return candidate
    candidate = MOTION_LIBRARY_ROOT / entry["dataset"] / entry["motion_file"]
    return candidate if candidate.is_file() else None


class VocabularySizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = load_manifest()
        self.entries = self.data["entries"]

    def test_jordanian_it_exactly_165_unique_tokens(self) -> None:
        tokens = {e["token"] for e in self.entries if e["dataset"] == "jordanian_it"}
        self.assertEqual(len(tokens), 165)

    def test_karsl_exactly_502_unique_tokens(self) -> None:
        tokens = {e["token"] for e in self.entries if e["dataset"] == "karsl"}
        self.assertEqual(len(tokens), 502)

    def test_isharah_exactly_680_unique_tokens(self) -> None:
        tokens = {e["token"] for e in self.entries if e["dataset"] == "isharah"}
        self.assertEqual(len(tokens), 680)

    def test_total_is_exactly_1347(self) -> None:
        self.assertEqual(len(self.entries), 1347)

    def test_no_duplicate_tokens_within_any_dataset(self) -> None:
        for dataset in ("jordanian_it", "karsl", "isharah"):
            rows = [e for e in self.entries if e["dataset"] == dataset]
            tokens = [e["token"] for e in rows]
            self.assertEqual(
                len(tokens), len(set(tokens)),
                f"{dataset} has duplicate tokens: "
                f"{[t for t in tokens if tokens.count(t) > 1]}",
            )

    def test_status_counts_equal_total(self) -> None:
        by_status: dict[str, int] = {}
        for e in self.entries:
            by_status[e["status"]] = by_status.get(e["status"], 0) + 1
        self.assertEqual(sum(by_status.values()), len(self.entries))


class ReadyEntryIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entries = load_manifest()["entries"]
        self.ready = [e for e in self.entries if e["status"] == "ready"]

    def test_every_ready_entry_has_a_readable_file(self) -> None:
        self.assertGreater(len(self.ready), 0, "No ready entries to check")
        for entry in self.ready:
            path = resolve_ready_file(entry)
            self.assertIsNotNone(
                path, f"{entry['dataset']}:{entry['token']} is 'ready' but its file is not readable"
            )

    def test_every_ready_entry_resolves_through_the_real_backend_resolver(self) -> None:
        # Regression test: test_every_ready_entry_has_a_readable_file above
        # uses its own path-joining helper (resolve_ready_file), which once
        # quietly diverged from the actual production resolver
        # (app.motions.manifest.MotionManifest._resolve_safe_path was
        # missing the per-dataset subfolder the builder actually writes to,
        # so every generated - non-baseline - motion 404'd despite this
        # exact test suite being green). This test calls the real backend
        # method directly so that class of gap cannot recur silently.
        from app.motions.manifest import MotionManifest

        manifest = MotionManifest()
        self.assertGreater(len(self.ready), 0, "No ready entries to check")
        failures = []
        for entry in self.ready:
            path = manifest.resolve_file(entry["dataset"], entry["token"])
            if path is None:
                failures.append(f"{entry['dataset']}:{entry['token']}")
        self.assertEqual(
            failures, [],
            f"{len(failures)} ready entries do not resolve through the real API resolver: {failures[:10]}...",
        )

    def test_a_batch_generated_non_baseline_token_serves_over_http(self) -> None:
        # Specifically exercises a token generated into
        # settings.motion_library/<dataset>/... (not one of the 6 baseline
        # files under frontend/public/motions), through the full FastAPI
        # app, the way a real client would request it.
        sys.path.insert(0, str(BACKEND_DIR))
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/motions/jordanian_it/POINTER")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json().get("format"), "signbridge-motion-v1")

    def test_every_ready_motion_json_has_correct_schema(self) -> None:
        for entry in self.ready:
            path = resolve_ready_file(entry)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data.get("format"), "signbridge-motion-v1", f"{entry['token']} has wrong schema")

    def test_stack_points_only_to_stack_07(self) -> None:
        stack = next(e for e in self.entries if e["dataset"] == "jordanian_it" and e["token"] == "STACK")
        self.assertEqual(stack["status"], "ready")
        self.assertEqual(stack["motion_file"], "stack_07.motion.json")
        # stack.motion.json (the non-preferred take) must not be its own entry.
        stack_tokens = {e["token"] for e in self.entries if e["dataset"] == "jordanian_it" and "STACK" in e["token"]}
        self.assertEqual(stack_tokens, {"STACK"})

    def test_array_and_karsl_0001_are_ready(self) -> None:
        array = next(e for e in self.entries if e["dataset"] == "jordanian_it" and e["token"] == "ARRAY")
        karsl0001 = next(e for e in self.entries if e["dataset"] == "karsl" and e["token"] == "0001")
        self.assertEqual(array["status"], "ready")
        self.assertEqual(karsl0001["status"], "ready")


class RebuildPreservesReadyTests(unittest.TestCase):
    def test_rebuild_manifest_does_not_lose_array_or_karsl_0001(self) -> None:
        before = {(e["dataset"], e["token"]): e["status"] for e in load_manifest()["entries"]}
        self.assertEqual(before.get(("jordanian_it", "ARRAY")), "ready")
        self.assertEqual(before.get(("karsl", "0001")), "ready")

        result = subprocess.run(
            [sys.executable, str(BUILDER_DIR / "builder.py"), "rebuild-manifest"],
            cwd=str(INTEGRATION_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        after = {(e["dataset"], e["token"]): e["status"] for e in load_manifest()["entries"]}
        self.assertEqual(after.get(("jordanian_it", "ARRAY")), "ready")
        self.assertEqual(after.get(("karsl", "0001")), "ready")
        self.assertEqual(len(load_manifest()["entries"]), 1347)


if __name__ == "__main__":
    unittest.main()
