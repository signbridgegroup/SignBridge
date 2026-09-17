# SignBridge Integration — Excluded Files Summary

Nothing listed here was deleted, moved, or modified. Everything stays at its original location under `C:\SignBridge_Project`. This is a record of what was deliberately **not** copied into `signbridge_integration/`, grouped by reason.

## Backup
- `avatar_web/avatar_ui_backup_20260905_210608/`, `..._212232/`, `..._220305/`, `..._221112/`
- `avatar_web/src/main.backup-before-claude.js`
- `avatar_web/src/motion-retarget.backup.js`, `motion-retarget.backup-before-claude.js`
- `avatar_web/src/motion-retarget.before-finger-relax.js`, `before-fix.js`, `before-rest-pose.js`, `before-smoothing.js`, `before-spline.js`
- `avatar_web/public/motions/stack.motion.upside_down.backup.json`
- `data/processed/01_Database_Chapter2_enriched_backup.md`, `01_Database_Chapter4_enriched_backup.md`, `03_Stacks_Core_enriched_backup.md`, `04_Queue_Core_enriched_backup.md`, `05_Pointers_Core_enriched_backup.md` (in the teammates' zip)

## Duplicate
- `avatar_web/public/avatars/signbridge_female.glb` and `female_export/avatar/model.glb` — byte-identical to each other (16,798,136 bytes) and neither is the GLB `main.js` actually loads.
- Zip's `data/processed/01_Database_Chapter{1-4}_bm25_index.json` — a stale duplicate of the index that each `01_Database_Chapter*_rag.py` module actually resolves from `data/vector_store/`.
- Root project's `integration_files_20260913_144837_696821/` and `integration_metadata_20260913_145251/` — an earlier, less accurate self-collection attempt (missing a real dependency, including some training-only scripts not in the actual inference import chain); superseded by the traced list in `runtime_inventory.md`.

## Abandoned experiment
- `avatar_rig_demo/`, `avatar_v2/`, `avatar_viewer_motion_test/`, `avatar_viewer_package/`, `avatar_viewer_stack_test/`
- `three-mediapipe-rig-main.zip` (old Unity/rig experiment archive)
- `avatar_web/public/motions/candidate_tests/*` (dozens of `handcrop`, `bidir`, `ccw`/`cw`, `reviewed` motion-extraction trial variants — none promoted to the approved `public/motions/` set)
- `src/extract_sign_motion_handcrop.py`, `extract_sign_motion_handcrop_validated.py`, `extract_sign_motion_reviewed.py` — later same-day iterations on the extractor; the approved motion files were confirmed (by exact JSON-schema comparison) to come from `extract_sign_motion.py`, not these.

## Diagnostics
- `check_book_*`, `check_poseworld_vs_video.py`, `check_right_frame44_spike.py`, `check_stack07_hand_gaps.py`, `diagnose_book_palm_orientation.py`, `inspect_book_right_jump.py`, `inspect_handcrop_*.py`, `preview_stack07_landmarks.py`, `retarget_foundation_test.py`, `clip_report_signbridge_reviewed.py`, `book_palm_orientation_diagnostic.png`, `book_hand_distance.csv`, `queue_old_vs_reviewed.csv`, `queue_reviewed_vs_*.csv`, `signbridge_clip_report.csv`
- `scripts/diagnose_book_arms.py`, `diagnose_left_wrist_regression.py`, `inspect_pose_jumps.py`, `check_thumb_max_frame.py`, `compare_pose_geometry.py`, `test_holistic.py`, `test_world_landmarks.py`, `preview_holistic.py`

## Repair / one-off investigation
- `add_bilateral_wrist_spacing.py`, `augment_book_json_right_hand.py`, `build_book_kabsch_palms*.py`, `book_06_1_kabsch_palms*.json`, `compare_book_hand_spacing*.py`, `fix_book_pip_dip_local_flexion.py`, `fix_left_mcp_palm_plane.py`, `fix_right_mcp_palm_plane.py`, `rebuild_book_wrist_features_*.py`, `reconstruct_handcrop_palm.py`, `repair_isolated_palm_spikes_v4.py`, `replace_right_finger_bends_with_raw.py`, `validate_handcrop_bidirectional.py`
- `scripts/retry_low_hand_videos.py`, `fix_dataset_names.py`

## Generated output (not committed; referenced by path when needed)
- `data/processed/*.npy`, `*.npz` feature arrays; `data/processed/model_data/X.npy`, `y.npy`
- `data/processed/*_training/` checkpoints other than the final one (`unified_ctc_training/`, `karsl_isolated_training/`, `isharah_ctc_training/`, `model_data/training_results*/`) — superseded by the final `unified_karsl_multitask_training/best_unified_karsl_multitask.pt`
- `data/processed/previews/`, `data/processed/hand_retry_previews/` (preview MP4s)
- `full_project_tree.txt`, `video_folders.txt` — one-off local notes generated during inspection, not part of the application

## Temporary file
- `integration_files_20260913_144837_696821/`, `integration_files_20260913_144837_696821.zip`
- `integration_metadata_20260913_145251/`, `integration_metadata_20260913_145251.zip`
- `incoming/teammates_inspection/` (this session's own extraction scratch area — left in place per instructions, not deleted, but not part of the delivered integration)

## Local environment
- `.venv/`, `avatar_web/node_modules/`, `avatar_web/dist/`, all `__pycache__/`

## Unused dependency
- `avatar_web/public/avatars/signbridge_female.vrm`, `signbridge_male.vrm` — VRM assets; the current `main.js` only uses `GLTFLoader` against the GLB, never loads a `.vrm`.

## External dataset asset (not committed to git; see `docs/DATASETS.md`)
- `data/external/karsl/videos/` — summarized only; per-sign-id metadata lives in `data/processed/karsl_shared198/samples.csv` and is what the motion builder actually reads.
- `data/external/isharah/frames/00/` — 1000 sample folders, 165,851 image files (~2.2 GB); summarized only. The original `00.zip` was not touched.
- `data/raw/sign_videos/` — 165 Jordanian IT classes, 967 audited videos; summarized only, selection driven by `data/processed/final_dataset_audit.csv` and `data/processed/dataset_inventory.csv`.

## Secret
- `key.env` (teammates' zip) — never extracted, never read, never copied.
- No `.env` file exists in the delivered integration; only `.env.example` with variable names and safe defaults.

## Obsolete model/checkpoint
- `data/processed/model_data/training_results_transfer/best_transfer_model.pt` — superseded transfer-learning checkpoint; not imported by `predict_final_signbridge_video.py`'s actual execution path (only its sibling module's constants are reused, the checkpoint itself is not loaded).
- `data/processed/unified_ctc_training/best_unified_ctc.pt`, `data/processed/karsl_isolated_training/best_karsl_isolated_model.pt` — intermediate teacher checkpoints consumed only during training (`train_unified_karsl_multitask.py`'s own training loop), not at inference time.
