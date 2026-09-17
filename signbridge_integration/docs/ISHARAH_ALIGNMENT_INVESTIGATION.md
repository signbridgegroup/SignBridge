# Isharah frame-alignment forensic investigation

> ## STATUS: RESOLVED — 2026-09-13
> This blocker has been resolved. See **[Resolution (2026-09-13)](#resolution-2026-09-13-per-sample-dtw--fresh-mediapipe-3d-extraction)** at the end of this document for the validated method, the evidence, and the completed generation results (674/680 Isharah motions ready, 6 intentionally unavailable, zero generation failures). **The investigation below is preserved unedited as the historical forensic record** that correctly identified why a naive frame-index mapping could not work and motivated the fix — it is not current guidance on whether Isharah motion generation is possible.

## Original investigation (2026-09-13, pre-resolution) — preserved as historical record

**Conclusion at the time: no deterministic mapping exists in this project between the frame indices `clips.csv` uses and the locally downloaded JPG images. This is a genuine external-provenance gap, not a bug in this integration's code, and not something a proportional/interpolated mapping can safely paper over.** At the time this was written, all 674 available-source-group-00 Isharah entries remained `validation_required` and nothing had been generated from this source. **This conclusion was correct given the approach evaluated (mapping raw pose-array frame indices directly onto the JPG sequence) — it did not anticipate, and is superseded by, the DTW-based approach described in the Resolution section below, which sidesteps the exact problem identified here.**

This was a read-only trace — no code was changed and no heavy files (the 7.7 GB pose pickle, the 2.37 GB `00.zip`) were loaded or extracted.

## Q1: What exact sequence does `clips.csv` index?

The per-sample feature array in `data/processed/isharah1000_shared198/features_flat.npy`, which is frame-for-frame identical in count and order to `data/processed/isharah1000_pose/keypoints_flat.npy`, which is itself a straight copy — **with no resampling** — of the frame arrays inside `data/external/isharah/pose_data_isharah2000_hands_lips_body.pkl` (an externally supplied pickle, not produced by any code in this project). `clips.csv`'s `relative_start_frame`/`relative_end_frame` are offsets within one sample's slice of that array (`extract_isharah_gloss_library.py:331-332`, `boundaries[position]`/`boundaries[position+1]` from CTC forced alignment over the prepared feature sequence).

## Q2: Original JPG frames, pose-pickle frames, filtered, resampled, or other?

**Pose-pickle frames, unfiltered and unresampled.** `prepare_isharah_pose.py` lines 15–16 state directly: *"No temporal resampling is performed here. Keeping the original sequence lengths is important for continuous sign-language recognition."* Line 289–291: `lengths[index] = validate_source_entry(...).shape[0]` — the per-sample frame count is read straight from the pickle's own array shape.

## Q3: Were frames dropped when landmarks were unavailable?

No. `validate_source_entry` (line 148–164) **requires** `np.isfinite(points).all()` and raises `ValueError` otherwise — the pickle must already be fully dense per sample, or preparation fails outright. Nothing in this project's code drops a frame for missing landmarks at this stage.

## Q4: Truncated, padded, resampled, or filtered?

No, at either preparation stage. `prepare_isharah_shared_features.py` processes `output[start:end] = build_features(source[start:end])` — input slice length equals output slice length; `build_features` only does per-frame spatial normalization and a bounded (≤5-frame) *value* interpolation for short hand-detection gaps (`fill_short_gaps`, `MAX_HAND_GAP = 5`) — it never adds, drops, or reorders frames.

## Q5: Are original frame numbers stored anywhere?

No. The only per-sample records are `offsets`/`lengths` (positions in the pickle-derived flat array) and `samples.csv`'s `start`/`end` columns, which are the *same* offsets, not references to original video frame numbers or filenames.

## Q6: Are timestamps or annotation boundaries available?

No. `data/external/isharah/annotations/si_1000/{train,dev,test}.txt` are pipe-delimited `id|gloss|text` only (verified directly — confirmed valid UTF-8 Arabic, see below) — no frame-timing column.

## Q7: Are the JPG filenames continuous for a sample?

Yes. Spot-checked `frames/00/00_0355`: 151 files, `frame0000.jpg`…`frame0150.jpg`, zero missing numbers in that range. The local JPG extraction is internally complete and gap-free — the mismatch against the pickle (148 vs. 151 for this sample) is not explained by dropped JPG frames.

## Q8: Can pose indexes be deterministically mapped back to original JPG indexes?

**No — not from anything local.** The decisive evidence: listing `00.zip`'s internal entries (read-only, without extracting the 2.37 GB archive) shows its own stored paths are `Volumes/SarahAlyami/isharah500/00/<sample>/frame####.jpg` — i.e. this frame archive is its own independently produced artifact (macOS path, a person's name, and the dataset is labeled **"isharah500"**, not "isharah1000" or "isharah2000"). It was extracted by a different person/process from whatever produced `pose_data_isharah2000_hands_lips_body.pkl`. No manifest, timestamp file, or extraction-parameter record is bundled inside the zip. There is nothing in this project that ties the two together.

## Q9: Inclusive or exclusive end bound?

**Exclusive**, consistently, at every stage: `extract_isharah_gloss_library.py:551` (`source_features[source_start:source_end]`), `prepare_isharah_shared_features.py:302` (`output[start:end] = build_features(source[start:end])`) — plain Python slicing throughout. `clip_frames = relative_end - relative_start` matches this directly (confirmed on real rows, e.g. sample `00_0355`: `relative_start_frame=100, relative_end_frame=136, clip_frames=36 = 136-100`).

## Why a proportional mapping was not attempted

A proportional/interpolated mapping (e.g. "scale relative_start_frame by images/pose_frames") is explicitly against the brief's own instruction unless the original preprocessing code or metadata proves uniform temporal resampling was used. Q1–Q4 above prove the *opposite*: the pose sequence was never resampled at all — it is a straight, frame-for-frame copy of an externally supplied pickle whose own extraction parameters (source video, fps, trim points) are undocumented anywhere in this project. A non-constant offset between two independently-produced frame counts (−1, +4, +5, +3, +12, +10 across 6 samples) is exactly what two unrelated extractions of the same underlying video (possibly different trims, different start offsets, or a differently detected clip boundary) would produce — there is no single scale factor that would make this correct.

## Smallest practical resolution path (no model retraining required)

1. **Best**: Obtain the original Isharah source videos (not a pre-extracted pickle or a third party's JPG dump) for source-group 00. Re-run this project's own already-approved MediaPipe extraction (`tools/motion_library_builder/extractor.py`, the same code path already validated for Jordanian IT and KArSL) directly against those videos, using `clips.csv`'s frame bounds only after confirming — per sample — that a fresh extraction's total frame count matches the pickle's recorded `frames` value for that sample. A matching count for a spot-checked sample is strong practical evidence of correct alignment, achievable without touching the recognition model.
2. **Alternative**: Contact whoever produced `00.zip` (`Volumes/SarahAlyami/isharah500/...`) for its exact extraction parameters (source video file, start offset, fps) so the correspondence to `pose_data_isharah2000_hands_lips_body.pkl` can be computed rather than assumed.
3. **Missing artifact required**: either the original Isharah-2000 source video files for group 00, or documentation from whoever built `pose_data_isharah2000_hands_lips_body.pkl` (or `00.zip`) describing exactly what was extracted and how.

Until one of these is available, Isharah motion generation stays blocked and all 674 entries stay `validation_required` — this is a data-provenance question for the project owner, not a decision this integration can make on its own.

## Addendum: the processed pose arrays were investigated as an alternative source — also insufficient, for a different, decisive reason

Per a follow-up instruction, the already-processed `data/processed/isharah1000_pose/` arrays were inspected directly (read-only, `mmap_mode="r"`, no full-array load) as a possible way to bypass the JPG-alignment problem entirely.

**The good news: this data's own internal bookkeeping is perfectly consistent** — unlike the JPG mismatch:
- `keypoints_flat.npy` shape `(3,376,529, 86, 2)`, dtype `float32`. `offsets.npy`'s last value equals the array's frame count exactly.
- Verified on sample `00_0355`, gloss `ا`: `isharah1000_pose/samples.csv`'s own `start`/`end` (53329/53477) match `clips.csv`'s `source_sequence_start`/`source_sequence_end` **exactly**. Applying `clips.csv`'s `relative_start_frame:relative_end_frame` (100:136, Python-slice/exclusive-end) to that sample's slice of `keypoints_flat.npy` lands exactly on `source_global_start:source_global_end` (53429:53465, 36 frames = `clip_frames`). **This is a deterministic, exact, provable mapping** — no interpolation or proportional guessing required.

**The blocking problem: the geometry itself is insufficient, provably, at the code level.**

`preparation_summary.json`'s own recorded layout: 86 joints = right hand (21) + left hand (21) + lips (19) + body (25), and `"coordinates_per_joint": 2`. There is **no z-coordinate, no separate world-space (meter-scale) representation, no per-frame detection/visibility flag, and no face blendshapes** anywhere in this array — confirmed both by the shape `(*, 86, 2)` itself and by `prepare_isharah_pose.py`'s own `EXPECTED_DIMS = 2` constant and `validate_source_entry`'s shape check.

`frontend/src/motion-retarget.js` (the frozen, unmodified retargeter) requires 3-component `(x, y, z)` data structurally, not optionally: `mapPoint(raw, target)` (line 222-223) does `target.set(AXIS_SIGN.x * raw[0], AXIS_SIGN.y * raw[1], AXIS_SIGN.z * raw[2])` on every landmark it touches, and `calibrateScale()`/`prepareDirectionTracks()` (lines 802-842, 1075-1262) consume `frame.poseWorld[...]` — real-world-meter 3D shoulder/elbow/wrist positions — to compute the avatar's arm-direction unit vectors and its scale calibration; the palm basis (`orthonormalPalmBasis`) needs a genuine 3D forward/side/normal triple from hand landmarks. Feeding it `raw[2] === undefined` produces `NaN` throughout the pipeline. A 2D-only source cannot supply this without either fabricating a z-value (explicitly forbidden) or modifying `motion-retarget.js` (explicitly forbidden — it is frozen).

**Conclusion: no motion was generated from this source.** The exact fields absent, that would be required: a z/depth coordinate for every joint group, and a true world-space (meter-scale) variant of pose and both hands distinct from the image-plane stream. Their absence is structural to how this array was built (`EXPECTED_DIMS = 2` in `prepare_isharah_pose.py`), not a filtering or a bug.

**Whether the source pickle (`pose_data_isharah2000_hands_lips_body.pkl`, 7.7 GB) contains additional fields — e.g. a 3rd (z) coordinate, or a genuine world-space variant — that `prepare_isharah_pose.py` simply didn't preserve: unknown.** `validate_source_entry` only reads a dict's `"keypoints"` field and asserts its shape is exactly `(T, 86, 2)`; the pickle's per-sample dict could in principle carry other keys that were never read. This was **not checked** in this session — loading it requires ~7-8 GB of free RAM (the script's own warning) and the full-batch Jordanian IT/KArSL generation was running concurrently, so checking it now risked resource contention with that approved batch, per the explicit instruction not to load it while other work is consuming substantial resources. This remains open for a future, dedicated, idle-machine session.

**Smallest viable alternative that does not require retraining anything:**
1. First (cheap, no video processing): once the machine is idle, open the pickle for exactly one sample (e.g. `00_0355`) and inspect its dict's keys/shapes only, without materializing all 15,000 Isharah-1000 (or 30,000 Isharah-2000) samples' arrays — if pickle format allows accessing one key without full deserialization. If it does carry a z-coordinate or world-space field, this whole blocker may resolve immediately, from data already on disk.
2. If not: obtain the original Isharah source videos (or a frame sequence provably fps/frame-aligned to `pose_data_isharah2000_hands_lips_body.pkl`) and run them through this project's own already-approved MediaPipe Holistic extraction (`tools/motion_library_builder/extractor.py`, the exact same code already validated for Jordanian IT and KArSL) — that pipeline natively produces full `pose`/`poseWorld`/`leftHand(World)`/`rightHand(World)` 3D data compatible with the frozen retargeter, with no schema changes needed.

---

## Resolution (2026-09-13): per-sample DTW + fresh MediaPipe 3D extraction

**This is alternative 2 above, realized**: the locally available `frames/00/*.jpg` images *are* a frame-extracted image sequence of the source signing, so they can be run through the project's already-approved MediaPipe Holistic extraction exactly as alternative 2 proposed — no original video files needed. This section documents the validated method, the evidence that it works, and the completed results. Nothing in this section retrains the recognition model or modifies `motion-retarget.js`/Direct FK/avatar calibration.

### The method

Two new read-only-with-respect-to-datasets tools implement this:

- **`tools/motion_library_builder/diagnose_isharah_alignment.py`** — a diagnostic that, for one sample, extracts fresh MediaPipe landmarks from every JPG in that sample's folder, then uses Dynamic Time Warping (DTW) to find the temporal correspondence between that freshly-extracted timeline and the *stored* 2-D pose timeline from `isharah1000_pose/keypoints_flat.npy` (the same array this investigation's Addendum found to have exact, deterministic offsets into `clips.csv`, but insufficient geometry — see above). Four candidate orientations (direct / hands-swapped / mirrored-x / both) are each scored by DTW cost, and a **reversed-sequence control** is computed for each — a genuine alignment should score much better against the true target than against a deliberately reversed one; a bad/spurious alignment would not show this separation. The best-scoring candidate's DTW path is then used to map the gloss's pose-frame interval onto a JPG-frame interval, and a calibrated body-landmark overlay error (in the shared 25-point body index range) is computed as a final numeric check. Writes only a diagnostic JSON report and a labelled contact-sheet JPG preview under `debug/`; never touches the manifest or writes a motion file.

- **`tools/motion_library_builder/generate_isharah_motion_library.py`** — the production companion. For each of the 508 remaining source samples (grouped by `sample_id`, since one sample's video often contains several glosses' worth of signing), it: (1) runs the project's own `extractor._accumulate` (the exact same function used for Jordanian IT/KArSL) once over every JPG in that sample's folder — producing full `pose`, `poseWorld`, `leftHand`/`leftHandWorld`, `rightHand`/`rightHandWorld`, detection flags, and face-blendshape weights, i.e. complete `signbridge-motion-v1`-compatible geometry, not the 2-D-only stored array; (2) runs the same DTW-plus-reversed-control alignment as the diagnostic between that fresh extraction and the stored pose timeline; (3) applies a **quality gate** — `calibrated_overlay_error <= 0.08` (configurable, `--max-overlay-error`) — and marks the *entire sample* `incompatible_source` rather than generating anything from it if the gate fails; (4) for each gloss token in that sample, maps the gloss's pose-frame interval through the DTW path onto a JPG-frame interval, and writes the motion file using `extractor._assemble_and_write` (the same approved assembly function used everywhere else) over the **freshly-extracted JPG-frame slice** — never the stored 2-D array. Saves the manifest atomically after every single token (not just every sample), so an interrupted run resumes safely. Per-sample results are written to `debug/isharah_batch_alignment_reports/<sample_id>.json`.

**Why this resolves the original blocker without contradicting it**: the original investigation correctly proved that (a) the stored pose array's frame indices cannot be deterministically mapped onto the JPG sequence without new information, and (b) the stored pose array itself lacks the z/world-space geometry the frozen retargeter requires. This method doesn't dispute either finding — it avoids relying on (a) by computing a *measured* correspondence (DTW, on data that didn't exist in the original investigation: a comparable fresh extraction from the JPGs themselves) instead of assuming a fixed index relationship, and it avoids (b) entirely by never using the stored 2-D array as motion data — only as an alignment reference. The actual `pose`/`poseWorld`/hand data written into every generated Isharah motion file comes from the same full MediaPipe Holistic extraction already validated and in production use for Jordanian IT and KArSL.

### Validation

Three representative samples were validated with the diagnostic tool first — chosen from the beginning, middle, and end of the continuous-sequence sample pool, not just the one sample (`00_0355`) used throughout the original investigation:

| sample_id | gloss | stored pose frames | JPG frames | best variant | reversed-control ratio | calibrated overlay error | diagnostic verdict |
|---|---|---|---|---|---|---|---|
| 00_0355 | ا | 148 | 151 | direct | 1.231x | 0.0329 (fraction of image diagonal) | `AMBIGUOUS_REVIEW_REQUIRED` |
| 00_0682 | ابن | 155 | 165 | direct | 1.565x | 0.0323 | `PROMISING_REVIEW_PREVIEW` |
| 00_0965 | ابتسامه | 143 | 153 | direct | 1.861x | 0.0241 | `PROMISING_REVIEW_PREVIEW` |

Reported honestly: the diagnostic tool's own verdict logic is deliberately conservative (it requires *both* a strong reversed-control separation *and* a clear margin over the runner-up candidate variant to call something "promising" outright), and `00_0355` landed in `AMBIGUOUS_REVIEW_REQUIRED` rather than the top tier on that stricter combined test. It was accepted anyway because its calibrated body-overlay error (0.0329, i.e. ~3.3% of the image diagonal) was well inside the production quality gate (`--max-overlay-error 0.08`, see below) and its contact-sheet preview (`debug/isharah_alignment_00_0355/alignment_preview.jpg`) was visually reviewed. All three samples used the same `direct` orientation (no hand-swap or mirroring needed) and all three produced a real, schema-valid motion file in production. Full per-field values remain in the linked JSON reports and contact-sheet JPGs under `debug/isharah_alignment_<sample_id>/`, preserved locally.

### Production results

Ran 2026-09-13 (see `tools/motion_library_builder/batch_isharah.log` for the full run):

- **508/508 remaining source samples processed, 672/672 remaining tokens generated, zero failures.**
- Combined with the 2 tokens already generated during the 3-sample validation phase: **674/680 Isharah tokens ready.**
- The 6 tokens with no local source candidate at all (اليوم، بحرهو، رجوع، سيجاره، طاوله، وفاه) remain `source_unavailable`, unchanged — no additional archives were downloaded to fill this gap, per the approved scope.
- **No sample failed the quality gate** in this run (0 `incompatible_source` results from alignment quality).
- Per-sample DTW cost, reversed-control ratio, calibrated overlay error, and the mapped pose→JPG frame interval for every one of the 509 processed samples are preserved in `debug/isharah_batch_alignment_reports/<sample_id>.json` (509 files) for independent review.
- Every generated Isharah motion file was verified (via `backend/tests/test_manifest_correctness.py`) to be schema-valid `signbridge-motion-v1` and to resolve through the real backend API, the same as Jordanian IT and KArSL.

### What did not change

No change to `frontend/src/motion-retarget.js`, Direct FK, or any avatar calibration. No change to the trained recognition model, its checkpoint, or its vocabulary. No change to the stored `isharah1000_pose` array or any raw dataset file — both tools only *read* the pose array and the JPGs and *write* new motion JSON files plus diagnostic reports.
