# SignBridge Integration — Runtime Inventory

Every file listed below was included because its necessity was established through one of: (1) the actual import chain starting from `predict_final_signbridge_video.py`, (2) compatibility with the final trained checkpoint and its saved metadata, (3) the active teammates' RAG/router import chain, (4) the current non-backup `avatar_web` entry point, or (5) a documented reproduction step for motion-library generation. Nothing was copied on filename similarity alone.

## Recognition runtime (category: runtime / model definition)

| Original path | Destination | Why required | Imported/used by |
|---|---|---|---|
| `predict_final_signbridge_video.py` | `backend/legacy_runtime/predict_final_signbridge_video.py` | Entry point / reference implementation for the final recognizer | Backend recognition service |
| `predict_it_video.py` | `backend/legacy_runtime/predict_it_video.py` | Supplies `HOLISTIC_MODEL`, `active_bounds`, `extract_video`, imported directly by `predict_final_signbridge_video.py` | `predict_final_signbridge_video.py` |
| `finetune_jordanian_it.py` | `backend/legacy_runtime/finetune_jordanian_it.py` | Hard import of `predict_it_video.py` (`TransferClassifier`, `to_shared198`); pulled in at module-load time even though not directly called | `predict_it_video.py` |
| `prepare_unified_sign_data.py` | `backend/legacy_runtime/prepare_unified_sign_data.py` | Supplies `jordanian_572_to_shared_198` | `predict_final_signbridge_video.py` |
| `scripts/prepare_model_sequences.py` | `backend/legacy_runtime/scripts/prepare_model_sequences.py` | Supplies `prepare_sequence`, `record_key`; kept under a `scripts` sub-package so the original `from scripts.prepare_model_sequences import ...` statements resolve unmodified | `predict_final_signbridge_video.py`, `predict_it_video.py` |
| `train_unified_karsl_multitask.py` | `backend/legacy_runtime/train_unified_karsl_multitask.py` | Defines `UnifiedMultiTaskModel`, the exact architecture matching the checkpoint's `state_dict` keys | `predict_final_signbridge_video.py` |
| `train_karsl_isolated_pretrain.py` | `backend/legacy_runtime/train_karsl_isolated_pretrain.py` | Hard import of `train_unified_karsl_multitask.py` (`IsolatedSignModel`, `speed_perturb`, `build_metadata`, `validate_inputs`); module-level import only — training code itself is never invoked. **Correction**: this file was absent from the previously-collected `integration_files_20260913_144837_696821` snapshot; it was copied fresh from the project root. | `train_unified_karsl_multitask.py` |
| `train_unified_sign_ctc.py` | `backend/legacy_runtime/train_unified_sign_ctc.py` | Supplies `make_model_features`, `BLANK_ID`, `MODEL_FEATURES`, `SHARED_FEATURES` | `train_unified_karsl_multitask.py`, `predict_final_signbridge_video.py` |

**Correction to the previous inspection report**: `predict_it_video.py` and `finetune_jordanian_it.py` were re-checked directly on disk inside `integration_files_20260913_144837_696821/` and are present there. The earlier report was wrong to list them as missing from that collection; the only genuinely missing file was `train_karsl_isolated_pretrain.py`.

Referenced at their existing location, never copied into web/source:

| Resource | Path | Category |
|---|---|---|
| Final checkpoint | `data/processed/unified_karsl_multitask_training/best_unified_karsl_multitask.pt` (~18.5 MB) | model weights |
| MediaPipe Holistic model | `models/holistic_landmarker.task` | model weights |
| KArSL human-readable labels | `data/processed/unified_sign_data_karsl/karsl_sign_mapping.csv` | data metadata |
| IT active-segment report | `data/processed/active_length_report.csv` | data metadata (optional bounds lookup) |

**Status of the recognizer**: trained, implemented, previously evaluated according to the saved training/evaluation reports in `data/processed/unified_karsl_multitask_training/`. It has **not yet been revalidated inside this new integrated backend** — see the testing section of the final report for the one-sample smoke test actually run.

## RAG / LLM / avatar semantic planner (category: RAG)

All from the teammates' zip (`incoming/SignBridge_Project(2).zip`), extracted only into `incoming/teammates_inspection/` first, never over the project root, `key.env` never extracted or read.

| Original path (in zip) | Destination | Why required |
|---|---|---|
| `signbridge_router.py` | `backend/legacy_runtime/rag/signbridge_router.py` | Single orchestration entry point: `answer_question()` |
| `signbridge_llm.py` | `backend/legacy_runtime/rag/signbridge_llm.py` | Groq call, grounding prompt, JSON parsing |
| `avatar_sign_mapper.py` | `backend/legacy_runtime/rag/avatar_sign_mapper.py` | Deterministic text → sign-token planner, called automatically by `answer_question()` |
| `avatar_sign_lexicon.json` | `backend/legacy_runtime/rag/avatar_sign_lexicon.json` | Semantic token lexicon (58 tokens) used by the planner |
| `signbridge_gloss_to_rag.py` | `backend/legacy_runtime/rag/signbridge_gloss_to_rag.py` | Domain-agnostic bridge: recognized gloss sequence → natural-language question |
| `stack_rag.py`, `queue_rag.py`, `pointers_rag.py` | `backend/legacy_runtime/rag/*.py` | Per-subject BM25 retrieval (Stack, Queue, Pointers) |
| `01_Database_Chapter1_rag.py` … `01_Database_Chapter4_rag.py` | `backend/legacy_runtime/rag/*.py` | Retrieval modules belonging to the single **Database** subject (chapters 1–4 are sub-indexes of one subject, not four domains) |

**Not reused**: `signbridge_video_to_rag.py`. It imports `predict_unified_video.py` / `best_unified_ctc.pt`, an older continuous-only checkpoint that predates KArSL support. New glue code (`backend/app/recognition/pipeline.py`) was written instead, combining the traced `predict_final_signbridge_video` logic with `signbridge_gloss_to_rag.glosses_to_question()` and `signbridge_router.answer_question()`.

RAG content data (category: RAG data), copied into `backend/legacy_runtime/rag/data/...` because each `*_rag.py` module resolves `DEFAULT_CHUNKS`/`DEFAULT_INDEX` relative to its own file location:

| Subject | Chunks source | Index source |
|---|---|---|
| Stack | root project `data/processed/03_Stacks_Core*` (already existed there) | root project `data/vector_store/stack_bm25_index.json` |
| Queue | zip `data/processed/04_Queue_Core*` | zip `data/vector_store/queue_bm25_index.json` |
| Pointers | zip `data/processed/05_Pointers_Core*` | zip `data/vector_store/pointers_bm25_index.json` |
| Database Ch1–4 | zip `data/processed/01_Database_Chapter{1-4}*` | zip `data/vector_store/01_Database_Chapter{1-4}_bm25_index.json` (the module's own `resolve_default_index()` logic was read to confirm it resolves to `data/vector_store/`, not the duplicate copy that also exists under `data/processed/` in the zip — that duplicate was skipped) |

`_enriched_backup.md` files and the duplicate `01_Database_Chapter*_bm25_index.json` under `data/processed/` were skipped (see `excluded_files_summary.md`).

## Avatar (category: avatar / frontend)

Active, non-backup files only, confirmed by reading `avatar_web/src/main.js` to see exactly what it loads (`AVATAR_URL = '/avatars/avatar_candidate.glb'`), not by filename:

| Original path | Destination | SHA-256 (before → after) |
|---|---|---|
| `avatar_web/src/motion-retarget.js` | `frontend/src/motion-retarget.js` | `E0178831...BA56B79` → identical |
| `avatar_web/src/avatar-spatial-diagnostics.js` | `frontend/src/avatar-spatial-diagnostics.js` | `5B421B46...74240 6` → identical |
| `avatar_web/src/style.css` | `frontend/src/style.css` | copied as a starting point; extended (not overwritten) for the new production UI |
| `avatar_web/public/avatars/avatar_candidate.glb` | `frontend/public/avatars/avatar_candidate.glb` | `994E083B...72A0CBA5` → identical |
| `avatar_web/public/motions/stack.motion.json` | `frontend/public/motions/stack.motion.json` | canonical, unmodified |
| `avatar_web/public/motions/stack_07.motion.json` | `frontend/public/motions/stack_07.motion.json` | canonical, unmodified — **preferred STACK take** |
| `avatar_web/public/motions/queue.motion.json` | `frontend/public/motions/queue.motion.json` | canonical, unmodified |
| `avatar_web/public/motions/datatype.motion.json` | `frontend/public/motions/datatype.motion.json` | canonical, unmodified |
| `avatar_web/public/motions/initialize.motion.json` | `frontend/public/motions/initialize.motion.json` | canonical, unmodified |
| `avatar_web/public/motions/run_time_error.motion.json` | `frontend/public/motions/run_time_error.motion.json` | canonical, unmodified |
| `avatar_web/public/motions/binary_relationship.motion.json` | `frontend/public/motions/binary_relationship.motion.json` | canonical, unmodified |

Not copied: `signbridge_female.glb`, `female_export/avatar/model.glb` (byte-identical duplicate of `signbridge_female.glb`), `signbridge_female.vrm`, `signbridge_male.vrm` — none of these are the GLB `main.js` actually loads (`GLTFLoader` only, `AVATAR_URL` points at `avatar_candidate.glb`).

`main.js` itself was **not** copied verbatim — it is rewritten in the new frontend as the UI/motion-queue integration layer (Section 21 of the brief). The retargeting mathematics it calls (`AvatarMotionRetargeter`, `BakedAvatarMotion`, `SignMotionClip`, `TUNING`) are the unmodified copied module; the new `main.js` only changes what triggers playback and in what sequence, verified against the original's public API (`sample(timeSeconds)`, `seekFrame(frameIndex)`, `reset()`).

## Motion-library builder reproduction step (category: training reproduction / motion-library generation)

| Original path | Destination | Why required |
|---|---|---|
| `src/extract_sign_motion.py` | `tools/motion_library_builder/legacy_extractor.py` | Byte-for-byte schema match against the 7 approved `public/motions/*.json` files was confirmed (identical top-level keys, no `"processing"` block, hand landmarks smoothed rather than gated) — this, not `extract_sign_motion_reviewed.py` or the `handcrop*` variants, is the script that actually produced the frozen approved motion files. Reused unmodified so newly generated motions stay in the same approved format. |

## Documentation / configuration (category: documentation)

`.env.example`, `.gitignore`, `README.md`, `runtime_inventory.md` (this file), `excluded_files_summary.md`, `docs/*` — authored new for this integration.

## Corrective verification phase (added after the first delivery — category: runtime / documentation)

A follow-up review found real defects in the first pass: a fabricated extra manifest token, a KArSL label that presented a numeric placeholder as a real word, no proof the Isharah frame-bound blocker was actually unresolvable rather than just unexplored, and no direct (non-lexicon) path from a recognized video token to its own motion. These were fixed, not just documented:

| File | Change |
|---|---|
| `tools/motion_library_builder/inventory.py` | Removed the fabricated `STACK_ALT_TAKE01` token; added `jordanian_it_token()`/`karsl_token()` as the single shared normalization source; Isharah tokens changed from placeholder `GLOSS_NNNN` IDs to the exact gloss string (verified 680/680 match the checkpoint's own continuous vocabulary); KArSL label selection no longer presents a numeric placeholder as a real word. |
| `tools/motion_library_builder/builder.py` | `register_baseline_motions` now only attaches a file to an existing canonical token, never creates one; added `find_orphan_motion_files`; `cmd_inventory` now hard-asserts the canonical total (1347) and per-dataset counts, checks for duplicate tokens at the source, refuses to save on any mojibake-flagged label, and merges (not replaces) preserved-ready entries so a source-metadata fix also reaches already-`ready` tokens without losing their generation result. |
| `tools/motion_library_builder/label_quality.py` | New — mojibake detection and label classification, shared by the builder's save-time guard and `validate`. |
| `backend/app/motions/manifest.py` | Added `RECOGNITION_MODE_TO_DATASET` and `resolve_recognized_token()` (Path A), imported straight from `inventory.py` so the normalization can never drift from what generated the manifest. |
| `backend/app/recognition/service.py` | Added `predicted_token` (raw, unformatted) and `motion_dataset_namespace` to the recognition result — previously only a human-formatted `predicted_label` existed, which was unusable for exact motion lookup. |
| `backend/app/services/pipeline.py` | Added `_resolve_path_a_motions()` and the `recognized_motion_sequence` response field, fully independent of the RAG/semantic `resolved_motion_sequence` (Path B). |
| `backend/app/schemas.py` | Added `predicted_token`, `motion_dataset_namespace`, `RecognizedMotion`, `RecognizeAndAnswerResponse.recognized_motion_sequence`. |
| `frontend/src/main.js` | Recognized-video results now render and can play the recognized sign's own motion (Path A) separately from the RAG answer's motion sequence (Path B); fixed a race where both could try to drive the avatar player at once. |
| `backend/tests/test_manifest_correctness.py`, `test_motion_contract.py` | New — the 9-point manifest-correctness suite and the Path A/B resolver suite requested in the corrective review. |
| `docs/ENCODING_INVESTIGATION.md`, `docs/ISHARAH_ALIGNMENT_INVESTIGATION.md`, `docs/MOTION_CONTRACT.md` | New — full evidence trails for the encoding and Isharah-alignment findings, and the Path A/B contract. |

## Full-batch generation phase (category: runtime / documentation)

| File | Change |
|---|---|
| `tools/motion_library_builder/` batch run | 158 remaining Jordanian IT + 501 remaining KArSL tokens generated, zero failures. 357.24 MB written to `C:\SignBridge_Project\data\processed\avatar_motion_library\`. |
| `backend/app/motions/manifest.py` | **Bug fix**: `_resolve_safe_path()` was missing the per-dataset subfolder (`settings.motion_library / <dataset> / <file>`) the builder actually writes generated motions to — every non-baseline ready motion would 404 through the real API despite the manifest correctly saying `ready`. Found by a live spot-check after the (green) test suite had missed it; fixed. Also: `LEXICON_TOKEN_BRIDGE` expanded from 3 to 30 entries from the recomputed semantic-bridge report (`docs/SEMANTIC_BRIDGE.md`). |
| `backend/tests/test_manifest_correctness.py` | Added two regression tests that call the *actual* production resolver / the *actual* FastAPI app for a non-baseline token, specifically so a test suite using a parallel/duplicate implementation instead of the real code path can't mask this class of bug again. |
| `backend/tests/test_motion_contract.py` | Updated for the 30-entry bridge table; `test_lexicon_bridge_never_maps_to_the_forbidden_similarity_guesses` now checks the *specific* forbidden targets (START→INITIALIZE, RELATIONSHIP→BINARY_RELATIONSHIP) rather than asserting those tokens stay unbridged forever, since they now have legitimate bridges of their own. |
| `docs/SEMANTIC_BRIDGE.md` | New — full 58-token exact-match report: 30 bridged (dataset + how matched), 28 with no exact match anywhere. |
| `docs/MOTION_COVERAGE.md`, `.gitignore` | Updated with final post-batch counts/timing and the resolver bug; batch run logs excluded from the repo. |

## Isharah DTW alignment + generation (category: runtime / motion-library generation)

| Original/new path | Destination | Why required |
|---|---|---|
| `tools/motion_library_builder/diagnose_isharah_alignment.py` | (authored directly in place, no copy) | Diagnostic: extracts fresh MediaPipe landmarks from one sample's JPGs, aligns them to the stored 2-D pose timeline with DTW (plus a reversed-sequence control and a calibrated-overlay-error check), writes a JSON report + contact-sheet preview under `debug/`. Never writes a motion file or touches the manifest. Reused by the production generator as a library, not duplicated. |
| `tools/motion_library_builder/generate_isharah_motion_library.py` | (authored directly in place) | Production generator: per source sample, runs the same `extractor._accumulate`/`extractor._assemble_and_write` functions already used for Jordanian IT/KArSL over the full fresh JPG extraction (not the stored 2-D array), aligns it to the stored pose timeline with the same DTW method to find each gloss's frame range, gates on alignment quality, and writes `signbridge-motion-v1` motion files + atomic manifest updates per token. |
| `data/processed/avatar_motion_library/isharah/*.motion.json` | (generated output, not copied) | 674 generated Isharah motion files, schema-validated by `test_manifest_correctness.py`. |
| `debug/isharah_alignment_00_0355/`, `..._00_0682/`, `..._00_0965/` | (generated output) | The 3 representative-sample validation diagnostics (report JSON + contact-sheet JPG) reviewed before the production batch ran. |
| `debug/isharah_batch_alignment_reports/*.json` (509 files) | (generated output) | Per-sample DTW cost / reversed-control ratio / calibrated overlay error / mapped frame interval for every sample processed in the production batch, for independent review. |
| `tools/motion_library_builder/batch_isharah.log` | (generated output, gitignored) | Full production run log: 508/508 samples, 672/672 tokens, 0 failures. |
| `backend/tests/test_motion_contract.py` | Updated (already present on disk when this phase's audit began) | `PathAIsharahTests`/`test_full_dataset_generation_does_not_invent_semantic_bridges` now assert Isharah entries resolve to real `ready` motions instead of asserting they stay blocked. |
| `README.md`, `docs/MOTION_COVERAGE.md`, `docs/ISHARAH_ALIGNMENT_INVESTIGATION.md`, `docs/TESTING.md`, `docs/SEMANTIC_BRIDGE.md`, `docs/DATASETS.md` | Updated | All outdated "Isharah is blocked" statements replaced with the verified final state (1,341/1,347 ready, 674/680 Isharah, 6 intentionally unavailable, zero generation failures). `docs/ISHARAH_ALIGNMENT_INVESTIGATION.md`'s original forensic trace was preserved unedited as a dated historical record, with a new dated resolution section appended rather than rewritten over it. |
