# Genuinely automatic sign-recognition router

## Confirmed root cause

`detect_mode()` in `backend/legacy_runtime/predict_final_signbridge_video.py` (imported directly into, and called on every request by, `backend/app/recognition/service.py::RecognitionService.recognize()`) decided `"auto"` mode by checking whether `"karsl"` appeared anywhere in the video's file path, and otherwise whether the file was found in a saved training-time report (`active_length_report.csv`, keyed by `(label, filename)`). A real camera recording or uploaded file is always saved as `upload_<uuid4>.<ext>` inside a per-request `signbridge_upload_*` temp directory (`backend/app/main.py`), so neither check can ever match live traffic - every `"auto"` web request silently fell through to `"continuous"` regardless of what was actually signed. This is the confirmed cause of a recorded Stack sign being reported as the unrelated word "جرس": the frontend defaults to `mode="auto"` (`frontend/index.html`), the request always lands on a temp path, and `detect_mode` had no way to see the video's actual content.

Two related, now-fixed defects were found in the same investigation:

1. `resolve_bounds()` only trimmed to the signing-active window (`active_bounds()`, also report-lookup-based) when `mode == "it"` - `"karsl"`/`"continuous"` traffic always used the full, untrimmed video, with the same "never matches a temp filename" problem for IT-mode traffic too.
2. `retrieve_stack()`'s definition-question boost (`signbridge_router.py`, fixed as part of the RAG work in this same session) incorrectly assumed Stack's slide 1 held the definition; the router logic above is unrelated to that fix but is mentioned here since both bugs share the same root pattern - an unverified assumption about which "slide 1"/"which file" holds the right content.

## Active runtime chain (confirmed by direct inspection, not assumption)

```
frontend/src/main.js (currentRecognitionMode(), runRecognition())
  -> frontend/src/api.js recognizeVideo() -- POST multipart {video, mode, top_k}
  -> backend/app/main.py POST /api/recognize
       - saves upload_<uuid4>.<ext> in a signbridge_upload_* temp dir
       - optional ffmpeg transcode (_ensure_decodable)
  -> backend/app/services/pipeline.py recognize_and_answer()
  -> backend/app/recognition/service.py RecognitionService.recognize()
       - extract_video() [predict_it_video.py] -- MediaPipe landmarks, once, shared by every mode
       - mode == "auto"  -> NEW: RecognitionService._recognize_auto() (this task)
       - mode != "auto"  -> UNCHANGED: detect_mode()/resolve_bounds() [predict_final_signbridge_video.py]
       - UnifiedMultiTaskModel [train_unified_karsl_multitask.py] -- one checkpoint, three heads
  -> backend/app/motions/manifest.py resolve_recognized_token() -- Path A motion lookup (untouched)
```

Confirmed preprocessing contracts (unchanged, just now fed real content-based bounds instead of "full video always" for the non-IT paths in auto mode):

- **Jordanian IT**: resampled to exactly 200 frames (`prepare_sequence(..., target_frames=200, ...)`).
- **KArSL**: native active-sequence length (`target_frames = active_frames`).
- **Continuous CTC**: also native active-sequence length; previously always the *entire* untrimmed video for `"continuous"`/`"karsl"` since `active_bounds()` never fired for live traffic.

## What was changed

### New file: `backend/legacy_runtime/auto_recognition_router.py`

Pure-function content-based router, independent of any video path/filename/model instance (directly unit-testable):

- **`analyze_video_content(raw)`** (Stage A): from the already-extracted MediaPipe landmarks (the same array every mode already computes), derives: total frames, the active signing window (first/last frame where a hand was observed, MediaPipe-dropout-tolerant), the number of separate motion segments, hand/pose observation rates, and an `enough_motion` gate. Never reads a filename or path.
- **`score_isolated(top_k, diagnostics)`** / **`score_continuous(...)`** (Stage B): explicit, documented heuristic composite scores - top-1 confidence, margin over the runner-up, how spread out the top-k is (normalized entropy), video quality, and segment-count evidence. **Explicitly not a calibrated probability** - no calibration data (temperature scaling, per-head validation accuracy) exists anywhere in this project's checkpoints, so none is claimed; this is a documented heuristic, as the task allows.
- **`decide_automatic_mode(...)`** (Stage C): compares the three candidates' *composite* scores (never the raw softmax values directly - see `test_never_picks_the_largest_raw_confidence_naively`), and safely rejects when there isn't enough motion, when the winning score is below an acceptance floor, or when the winner isn't clearly separated from the runner-up.

### `backend/app/recognition/service.py`

- `RecognitionService.recognize()` now branches: `mode == "auto"` (and no manual frame-bounds override) goes to the new `_recognize_auto()`; every explicit mode (`it`/`karsl`/`continuous`/`single`) is **completely unchanged** - same `detect_mode`/`resolve_bounds` call, same preprocessing, same result shape.
- `_recognize_auto()` runs **two** forward passes through the already-loaded checkpoint (IT-style 200-frame, and native-length shared by KArSL + continuous - matching the two genuinely different contracts above), evaluates all three candidates, and returns either an accepted result (same shape as an explicit-mode result, plus new diagnostic fields) or a rejected result with `predicted_label`/`predicted_token`/`decoded_gloss_sequence`/`motion_dataset_namespace` all empty/`None` - this is what guarantees a rejected recognition can never reach avatar motion playback (`_resolve_path_a_motions`/`_recognition_to_gloss_text` in `pipeline.py` already treat an empty/`None` result as "nothing to resolve", unchanged).

### `backend/app/schemas.py` / `backend/app/services/pipeline.py`

Additive only: `RecognitionResult` gained `accepted: bool`, `routing_reason: str | None`, `candidate_scores: dict`, `quality: dict` (all with safe defaults), and `resolved_mode`'s type widened to allow a new `"rejected"` value alongside the existing `it`/`karsl`/`continuous`. Every existing field is unchanged; `pipeline.py` passes the new fields through additively.

### Frontend

`frontend/index.html`: the four mode radio buttons moved into a collapsed `<details>` ("خيارات متقدمة للمطورين فقط") - still functional for controlled/manual testing, no longer presented as a normal choice; `auto` stays checked by default, so a user who never opens it always sends `mode: "auto"`. `frontend/src/main.js::renderRecognition()` now shows one plain-language status line (`تم التعرف على: <label>` with a confidence percentage only when it reflects one candidate's own top-1 score, or the required clear Arabic retry message when the router rejects the recording) and moved the raw `requested_mode`/`resolved_mode`/`routing_reason`/`candidate_scores` fields into a second collapsed diagnostics block. No dataset terminology (IT/KArSL/Isharah) appears in the normal-flow text a user sees.

## API diagnostics shape (auto mode)

```json
{
  "requested_mode": "auto",
  "resolved_mode": "it",
  "accepted": true,
  "routing_reason": "isolated sign; the Jordanian IT head had the strongest accepted normalized evidence",
  "predicted_label": "Stack",
  "predicted_token": "Stack",
  "confidence": 0.84,
  "candidate_scores": {
    "it": {"label": "Stack", "raw_confidence": 0.8406, "routing_score": 0.8703, "margin": 0.8356},
    "karsl": {"label": "0277", "raw_confidence": 0.0478, "routing_score": 0.1678, "margin": 0.0031},
    "continuous": {"decoded_gloss_sequence": ["Stack"], "routing_score": 0.5131, "average_token_confidence": 0.8949}
  },
  "quality": {"left_hand_observed_rate": 90.48, "right_hand_observed_rate": 78.57, "pose_observed_rate": 100.0, "active_frame_count": 84, "active_to_total_ratio": 0.8155, "segment_count": 1}
}
```
This is the **real, live output** of the fixed router on the exact reproduction scenario below - not a fabricated example.

## Confirmed fix - live reproduction

Copied the real Jordanian-IT Stack sample (`C:\SignBridge_Project\data\raw\sign_videos\Stack\Stack_07_1.mp4`) to a temp-upload-style path with a random uuid filename (no "stack"/"karsl" substring anywhere in the path, exactly mimicking a live upload) and called `recognition_service.recognize(temp_path, mode="auto")` through the real checkpoint:

- **Before this fix** (by code inspection - `detect_mode` returns `"continuous"` for any path with no "karsl" substring and no active-report match, which every temp path satisfies): would have run only the CTC decoder.
- **After this fix**: `resolved_mode = "it"`, `predicted_label = "Stack"`, `confidence = 0.84`, with IT's composite routing score (0.87) decisively beating KArSL (0.17) and continuous (0.51) - see the exact JSON above.

Also confirmed with a real KArSL sample (`data/external/karsl/videos/03/train/0001/...mp4`, copied to the same temp-style path): `resolved_mode = "karsl"` (routing correctness only, no accuracy claim about the specific label).

## Safe rejection

`decide_automatic_mode` rejects (returns `resolved_mode: "rejected"`, `accepted: false`, with an explanatory `routing_reason`) when: no hand is ever observed or motion is too brief (`enough_motion` gate, before any model evaluation even runs), the best candidate's composite score is below `MIN_ACCEPT_SCORE` (0.35), or the winner isn't separated from the runner-up by at least `MIN_WINNER_MARGIN` (0.05). A rejected result's `predicted_label`/`predicted_token`/`decoded_gloss_sequence`/`motion_dataset_namespace` are all empty/`None`, so `pipeline.py`'s existing (unmodified) Path A motion resolution and gloss-to-RAG bridge both naturally treat it as "nothing recognized" - confirmed by `test_rejected_recognition_never_reaches_motion_playback_end_to_end`.

## Tests added (`backend/tests/test_auto_recognition_routing.py`, 22 tests, 1 honestly skipped)

| # | Requirement | Test | Result |
|---|---|---|---|
| 1 | Temp camera-style filename must not force `continuous` | `test_temp_camera_style_filename_does_not_force_continuous` (real Stack video, temp uuid path) | pass |
| 2 | Temp upload filename must not determine dataset | `test_temp_upload_filename_does_not_determine_dataset` | pass |
| 3 | Explicit `it` mode still selects IT | `test_explicit_it_mode_still_selects_it_path` | pass |
| 4 | Explicit `karsl` mode still selects KArSL | `test_explicit_karsl_mode_still_selects_karsl_path` | pass |
| 5 | Explicit `continuous` mode still selects CTC | `test_explicit_continuous_mode_still_selects_continuous_path` | pass |
| 6 | Auto mode evaluates video content | `test_temp_upload_filename_does_not_determine_dataset` (asserts all 3 `candidate_scores` present) | pass |
| 7 | Known Stack video -> IT -> "Stack" | `test_known_stack_video_selects_it_path_and_returns_stack` | pass |
| 8 | Isolated KArSL example -> KArSL path | `test_isolated_karsl_example_selects_karsl_path` | pass |
| 9 | Continuous Isharah example -> continuous path | **No real playable Isharah video exists in this repository** (`data/external/isharah` only has extracted frame JPEGs, a 2.37GB frame-image zip, and a 7.74GB pose pickle - confirmed by a dedicated asset search). `ContinuousPathTests` states this explicitly and is skipped, not marked passed; `test_strong_continuous_evidence_with_multiple_segments_is_accepted` covers the *decision logic* with clearly-labeled synthetic evidence instead. | honestly skipped |
| 10 | Poor-quality/ambiguous input rejected safely | `test_insufficient_motion_is_rejected_before_scoring_candidates`, `test_all_low_confidence_candidates_are_rejected`, `test_near_tied_candidates_are_rejected_as_ambiguous` | pass |
| 11 | Rejected recognition never reaches motion playback | `test_rejected_recognition_never_reaches_motion_playback_end_to_end`, `test_rejected_result_resolves_to_no_motion`, `test_rejected_result_produces_no_gloss_text` | pass |
| 12 | Camera and upload share the same routing contract | `test_recognize_endpoint_uses_one_shared_code_path_for_every_extension` (structural: exactly one `recognize_and_answer(...)` call site in the endpoint, parameterized only by the validated path and the caller's mode - no extension-specific branch) | pass |

Also: `test_never_picks_the_largest_raw_confidence_naively` directly proves the composite-score comparison is not naive argmax-over-raw-softmax.

## What was NOT changed (frozen constraints honored)

No retraining or fine-tuning of any model. No change to `train_unified_karsl_multitask.py`'s model architecture, any checkpoint file, `motion-retarget.js`, `avatar-spatial-diagnostics.js`, `avatar_candidate.glb`, the motion manifest (`data_manifests/motion_manifest.json`) or its resolver (`backend/app/motions/manifest.py`, read-only in this task), Direct FK, or Three.js avatar playback. `frontend/src/camera-recorder.js` (the mirroring fix from a prior task) is untouched - confirmed by the frontend test suite (`npm test`, 12/12 passing) and the fact that `runRecognition()`'s call into `recognizeVideo()` is unchanged; only the *mode value it's called with* is now always `"auto"` by default via the moved-but-still-functional radios.

## Remaining limitations

- The composite-score weights are a documented, conservative heuristic (see module docstring in `auto_recognition_router.py`) - not fitted against a labeled auto-routing dataset, because none exists in this project. They should be revisited if a labeled validation set for auto-routing decisions is ever collected.
- No real continuous/Isharah video asset exists in this repository to validate the continuous path end-to-end against the real checkpoint; only the decision logic itself was validated (with real Stack/KArSL videos for the other two paths).
- The motion-segmentation heuristic in `analyze_video_content` (hand-presence run-length with a fixed gap-merge tolerance) is intentionally simple; a genuinely continuous multi-sign recording with long inter-sign pauses could still be merged into one segment if the pauses are short enough, which would bias the router toward an isolated-sign decision. This is a known, documented conservative bias, not a silent gap.
