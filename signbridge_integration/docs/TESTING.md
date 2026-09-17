# Testing — what was run, what passed, what wasn't run

## Backend (this session, all passing)

Run with:
```powershell
cd signbridge_integration
C:\SignBridge_Project\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
```

| Test file | Covers | Result |
|---|---|---|
| `test_api.py` | health/config without a Groq key, `/api/motions` list + status, safe token resolution, 404 on an unknown token, path-traversal rejection (2 variants), `/api/ask` returns a structured 503 when Groq is unconfigured | 9/9 passed |
| `test_recognition_smoke.py` | One real inference through the traced checkpoint path (`Stack_07_1.mp4`, mode `it`) | passed — predicted "Stack" at 87.3% confidence on CUDA in 3.74 s |
| `test_pipeline_smoke.py` | Recognition → gloss → question → RAG, with no Groq key configured, must not crash | passed — recognized "Stack", produced the question "ما هو Stack؟", answer correctly marked unavailable with a clear reason |
| `test_recognize_endpoint.py` | Full HTTP round trip through `POST /api/recognize` (multipart upload) + confirms no leftover temp upload directory afterward; rejects an unsupported file extension | 2/2 passed |
| `test_manifest_correctness.py` | Canonical vocabulary sizes (165/502/680, total 1,347), zero duplicate tokens per dataset, status-count sanity, every `ready` entry has a real readable `signbridge-motion-v1` file resolving through the *actual* backend resolver (not a parallel test helper — this specific check caught a real path-resolution bug, see `docs/MOTION_COVERAGE.md`), `STACK` points only to `stack_07.motion.json`, a `rebuild-manifest` subprocess run does not lose ready status, a non-baseline token serves correctly over real HTTP | 13/13 passed |
| `test_motion_contract.py` | Path A (recognized token → exact dataset motion) for all three dataset namespaces including Isharah (now resolves to a real `ready` motion, not just a namespace check), Path B (lexicon token → bridged motion) for all 30 confirmed bridges including the 3 Isharah-only ones (now `ready`), explicitly checks the brief's forbidden guesses (`START`→`INITIALIZE` etc.) still resolve to their own correct bridge rather than the guessed target, confirms generating the full physical library did not invent bridges for the other 28 lexicon concepts | 14/14 passed |

**Total: 40/40 backend tests passing** (latest full run: `Ran 40 tests in 22.041s` / `OK`).

Also confirmed directly (not as an automated test): `uvicorn app.main:app` starts cleanly and answers `GET /api/health` over real HTTP.

## Frontend (this session)

- `npm install` — 15 packages, 0 vulnerabilities.
- `npm run build` — succeeds, produces `dist/` (658 KB main bundle, a size warning only, no errors).
- `npm run dev` — Vite starts in <1s; confirmed over real HTTP that `/` serves the RTL Arabic page, `/avatars/avatar_candidate.glb` serves the correct byte count, and `/src/main.js` resolves through Vite's module graph without a build error (i.e. every `import` in `main.js`, `motion-queue.js`, `camera-recorder.js`, `api.js`, and the copied `motion-retarget.js`/`avatar-spatial-diagnostics.js` resolves).

## Full motion-generation batches: completed, not just tested in isolation

The full Jordanian IT (165) + KArSL (502) batch and the full Isharah (674, via per-sample DTW alignment) batch have both run to completion — see `docs/MOTION_COVERAGE.md` for exact counts, timing, and the two batch logs (`tools/motion_library_builder/batch_jordanian_it.log`, `batch_karsl.log`, `batch_isharah.log`). Zero generation failures across all three. Every `ready` motion file is covered by `test_manifest_correctness.py`'s schema and real-resolver checks above (1,341 files individually checked on every test run, not a sample).

## Not run — and why

- **Camera permission-denied / camera-unsupported UI states, recorded-video preview, responsive layout at phone width, Arabic RTL visual rendering, avatar visually loading/animating correctly in a real browser.** This session has no browser automation available; these are genuine UI/UX checks that need a human (or a browser-automation tool) actually looking at the page. The code paths exist (`CameraRecorder` throws a clear Arabic error when `getUserMedia`/`MediaRecorder` are unsupported; `index.html`/`style.css` are responsive down to a single column under 900px) but were not visually confirmed.
- **`POST /api/ask` with a real Groq key.** No live key is available in this session (`backend/.env` does not exist). The failure path (missing key → 503) and every step up to the Groq API call (routing, BM25 retrieval, prompt construction) were exercised; the actual model call was not.
- **Linguistic/sign-language correctness of any generated or existing motion**, Isharah included. The DTW alignment method (see `docs/ISHARAH_ALIGNMENT_INVESTIGATION.md`) is a *temporal-correspondence* validation (does this JPG range correspond to this gloss in the stored timeline) with a numeric quality gate and a human-reviewed contact-sheet preview for the 3 initial samples — it is not, and does not claim to be, a linguistic correctness review by a qualified signer. That remains out of scope for this integration.
