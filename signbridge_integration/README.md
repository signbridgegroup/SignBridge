# SignBridge — Integrated Application

An AI-powered educational assistant for Deaf and hard-of-hearing students, focused on Jordanian Sign Language and introductory IT/CS education (Stack, Queue, Pointers, Database). This is the integration of three previously separate pieces of work: the final trained multi-dataset sign-recognition model, the teammates' RAG/Groq educational pipeline, and the approved Three.js/MetaPerson avatar.

## 1. Project purpose

Let a student either type a question or sign a video (camera or upload), get a grounded educational answer generated from real course material, and see that answer rendered by a 3D avatar where a matching sign motion actually exists — while being honest, everywhere, about which motions don't exist yet.

## 2. Supported educational subjects

Stack, Queue, Pointers, Database (Database chapters 1–4 are one subject with four retrieval sub-indexes, not four domains).

## 3. Supported recognition datasets

1. Custom Jordanian IT isolated-sign dataset (165 classes)
2. KArSL isolated-sign dataset (502 classes)
3. Isharah continuous Saudi Sign Language dataset

No alphabet/fingerspelling dataset is included — that scope was cancelled upstream.

## 4. Architecture

```
Typed question ──────────────┐
                              ▼
Camera / uploaded video → Recognition (final checkpoint) → gloss → question ┐
                                                                              ▼
                                                        signbridge_router (BM25 retrieval)
                                                                              ▼
                                                              Groq (educational_answer,
                                                               simplified_text, avatar_text)
                                                                              ▼
                                                        avatar_sign_mapper (sign_tokens)
                                                                              ▼
                                                     Motion manifest resolver (real files only)
                                                                              ▼
                                              Frontend: written answer + avatar motion queue
```

Backend: FastAPI (`backend/app`). Recognition and RAG/LLM/avatar-planner logic are the original research code, reused via `backend/legacy_runtime` (unmodified, checkpoint-compatible) rather than reimplemented. Frontend: the approved Three.js/Vite avatar viewer (`motion-retarget.js`, `avatar-spatial-diagnostics.js`, unchanged, hash-verified), wrapped in a new production UI — typed question, camera recording, video upload, and a **motion-library browser** (`frontend/src/motion-library.js`) that lists all 1,347 canonical tokens across the three datasets from `GET /api/motions`, with dataset/status filtering, Arabic/English search, and a Play button per ready motion (fetches only that one motion's JSON, on demand).

## 5. Environment requirements

- Windows, with the existing `C:\SignBridge_Project\.venv` (Python 3.13, PyTorch/CUDA, MediaPipe, OpenCV, pandas already installed).
- Node.js 18+ and npm (tested with Node 24 / npm 11).
- A Groq API key for the RAG/LLM step (recognition and the avatar work without one).

## 6. Local installation

```powershell
# Backend — only 5 packages were added to the EXISTING venv, nothing upgraded:
C:\SignBridge_Project\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

# Frontend
cd frontend
npm install
```

## 7. Configuration

```powershell
Copy-Item .env.example backend\.env
```
Edit `backend\.env` and set `GROQ_API_KEY`. Every other variable has a working default for the original development machine (`C:\SignBridge_Project`). See `.env.example` for the full list and `docs/ENVIRONMENT.md` for what each one controls.

## 8. Backend startup

```powershell
cd backend
C:\SignBridge_Project\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```
Verify: `http://localhost:8000/api/health`.

## 9. Frontend startup

```powershell
cd frontend
npm run dev
```
Open `http://localhost:5173`.

## 10. Camera permissions

The camera panel never requests access until the user clicks "تشغيل الكاميرا" (Start camera). It is a record-then-analyze flow, not continuous live recognition: record, stop, preview, then choose retake or analyze. All camera tracks are stopped on close, on retake, on component teardown, and on page unload.

## 11. Recognition modes

`auto` (تلقائي), `it` (الإشارات الأردنية التقنية), `karsl` (KArSL), `continuous` (Isharah / إشارات متصلة) — see `POST /api/recognize`.

## 12. Motion-library inventory

`tools/motion_library_builder/builder.py inventory` builds `data_manifests/motion_manifest.json` covering all 1,347 canonical tokens across the three datasets (165 jordanian_it + 502 karsl + 680 isharah) from existing metadata only. Re-run `rebuild-manifest` any time to refresh without losing already-generated entries.

As of the latest full generation run, **1,341 of 1,347 tokens are `ready`** (165/165 jordanian_it, 502/502 karsl, 674/680 isharah); the remaining 6 are the intentionally-accepted-unavailable Isharah glosses (see `docs/MOTION_COVERAGE.md`). This is physical motion coverage across the three recognition datasets — it is a separate number from semantic-planner coverage (30/58 lexicon concepts bridged; see `docs/SEMANTIC_BRIDGE.md`).

## 13. Motion generation

```powershell
python tools\motion_library_builder\builder.py generate-token <dataset> <token>
```
`dataset` is `jordanian_it`, `karsl`, or `isharah`. Isharah tokens are generated by `tools/motion_library_builder/generate_isharah_motion_library.py` instead (per-sample DTW alignment; see `docs/ISHARAH_ALIGNMENT_INVESTIGATION.md`). See `docs/MOTION_COVERAGE.md` for full current coverage.

## 14. Testing

```powershell
cd signbridge_integration
C:\SignBridge_Project\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
```
See `docs/TESTING.md` for what's covered and what isn't (camera permission states, visual avatar correctness — those need a real browser and a human).

## 15. Dataset acquisition

See `docs/DATASETS.md` for each dataset's expected location, what's already local, and what redistribution policy applies.

## 16. Model reproduction

The final checkpoint is already trained; this integration does not retrain it. For provenance, `backend/legacy_runtime/train_unified_karsl_multitask.py` and `train_karsl_isolated_pretrain.py` are the training entry points that produced it (see the original project's `data/processed/unified_karsl_multitask_training/training_summary.json` for the recorded run).

## 17. Evaluation results

See the original project's saved reports (`data/processed/unified_karsl_multitask_training/training_summary.json`, `isharah_test_predictions.csv`, `karsl_test_predictions.csv`). This integration additionally re-verified the checkpoint with one live inference smoke test (`backend/tests/test_recognition_smoke.py`) — see `docs/TESTING.md` for the result.

## 18. Current limitations

- Physical motion coverage across the three recognition datasets is now 1,341/1,347 (99.6%) — jordanian_it and karsl are fully generated, isharah has 674/680 with 6 glosses intentionally left `source_unavailable` (see `docs/MOTION_COVERAGE.md`). The written answer always works even when a motion is missing, and missing tokens are always reported honestly rather than hidden.
- Physical motion coverage is a separate number from semantic-planner coverage: only 30 of the 58 `avatar_sign_lexicon.json` concepts have an exact, non-guessed bridge to a real motion (see `docs/SEMANTIC_BRIDGE.md`) — generating (or fully covering) the recognition-dataset motion library does not by itself increase semantic coverage.
- `sign_tokens` is a canonical concept sequence for avatar planning, not validated Jordanian Sign Language grammar.
- Remaining avatar visual/motion issues (documented in `avatar_web`'s own history) are known limitations, not something this integration attempted to fix — Direct FK and the frozen retargeting math were preserved unchanged throughout, including through the Isharah motion generation work.
- `/api/ask` was verified structurally without a live Groq key; the actual Groq call path needs a real key to fully exercise.

## 19. GitHub / data-licensing policy

See `LICENSE_OR_DATA_USAGE_NOTES.md`.

## 20. Citation information

This integration does not introduce a new model or dataset; cite the original SignBridge project and the KArSL / Isharah dataset authors per `docs/DATASETS.md`.
