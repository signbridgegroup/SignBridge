# Dataset acquisition, licensing, and expected layout

None of the three datasets' raw media are committed to this repository. Only small metadata CSVs/JSON that the pipeline needs at runtime are referenced from their existing location under `C:\SignBridge_Project\data`.

## 1. Custom Jordanian IT isolated-sign dataset

- **Status**: privately collected for this project. **Raw videos are withheld** pending confirmed consent and redistribution rights from the signers — do not publish them.
- **Expected location**: `data/raw/sign_videos/<Label>/<Label>_<signer>_<take>.<ext>`
- **Metadata already safe to include**: `data/processed/final_dataset_audit.csv`, `data/processed/active_length_report.csv` (165 labels, 967 audited videos, per-clip hand-observation rates and active-segment bounds — no video content).
- **Integrity checker**: `tools/motion_library_builder/inventory.py::build_jordanian_it_inventory` will mark any label `source_unavailable` if its selected video file isn't present locally.

## 2. KArSL (King Saud University Arabic Sign Language, isolated)

- **Official source**: KArSL is a published academic dataset — obtain it from its original publishers/authors and follow their license terms; this repository does not redistribute it.
- **Citation**: Sidig, A. A. I., Luqman, H., Mahmoud, S., & Mohandes, M. "KArSL: Arabic Sign Language Database." *ACM TALLIP*, 2021.
- **Expected location**: `data/external/karsl/videos/<signer>/<split>/<sign_id>/...`
- **Metadata already safe to include**: `data/processed/karsl_shared198/samples.csv` (502 sign IDs, per-clip detection rates), `data/processed/unified_sign_data_karsl/karsl_sign_mapping.csv`.

## 3. Isharah (continuous Saudi Sign Language)

- **Official source**: obtain from the Isharah dataset's original publishers; this repository does not redistribute it.
- **Expected location for avatar motion source**: `data/external/isharah/frames/00/<sample_id>/frame####.jpg` (source-group 00 only, already downloaded/extracted locally — the original `00.zip` is untouched).
- **Expected location for recognition training data**: `data/external/isharah/pose_data_isharah2000_hands_lips_body.pkl` (pre-extracted pose features, not raw video).
- **Metadata already safe to include**: `data/processed/isharah_gloss_clip_library/clips.csv`, `data/processed/isharah1000_shared198/samples.csv`.
- **Coverage**: 674 of 680 glosses in the trained vocabulary have at least one source-group-00 candidate and now have a real, generated `signbridge-motion-v1` motion file; 6 do not (اليوم، بحرهو، رجوع، سيجاره، طاوله، وفاه) and are marked `source_unavailable`. No additional Isharah archives were downloaded to fill this gap, per the approved scope.
- **Alignment method**: the local image frame count and the training pose-feature frame count are not 1:1 aligned for a given sample (no fixed offset exists between them), so motion generation does not map frame indices directly. Instead, fresh MediaPipe landmarks are extracted from the JPGs and aligned to the stored pose timeline per-sample with Dynamic Time Warping, with a reversed-sequence control check and a calibrated-overlay-error quality gate; the motion data itself comes from the fresh extraction, never the frame-index mapping. See `docs/ISHARAH_ALIGNMENT_INVESTIGATION.md` for the full method, validation evidence, and results (674/680 ready, zero generation failures).

## Preparation commands (already run upstream; not re-run by this integration)

The original project's `prepare_isharah_pose.py`, `prepare_isharah_shared_features.py`, `prepare_karsl_unified_data.py`, and `prepare_unified_sign_data.py` produced the metadata this integration reads. This integration does not re-run them.
