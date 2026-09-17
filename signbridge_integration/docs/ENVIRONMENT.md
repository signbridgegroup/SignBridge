# Environment variables

All read in `backend/app/config.py`. Copy `.env.example` to `backend/.env` and edit.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Groq API key. Without it, recognition/avatar/motions still work; `/api/ask` returns a 503 configuration error and `/api/recognize` degrades to `educational_answer_available: false`. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Passed straight to `signbridge_llm.py`. |
| `SIGNBRIDGE_PROJECT_ROOT` | `C:\SignBridge_Project` | Root of the original research project. Every other path below defaults relative to this. |
| `SIGNBRIDGE_CHECKPOINT` | `<root>\data\processed\unified_karsl_multitask_training\best_unified_karsl_multitask.pt` | Final recognition checkpoint. |
| `SIGNBRIDGE_MEDIAPIPE_MODEL` | `<root>\models\holistic_landmarker.task` | MediaPipe Holistic model. |
| `SIGNBRIDGE_KARSL_LABELS` | `<root>\data\processed\unified_sign_data_karsl\karsl_sign_mapping.csv` | Human-readable KArSL labels. |
| `SIGNBRIDGE_MOTION_LIBRARY` | `<root>\data\processed\avatar_motion_library` | Where the motion-library builder writes newly generated motion JSON. Must stay outside `frontend/`. |
| `SIGNBRIDGE_FRONTEND_ORIGINS` | `http://localhost:5173` | Comma-separated list; sets FastAPI CORS `allow_origins`. |

## Python packages installed by this integration

Into the **existing** `C:\SignBridge_Project\.venv` — nothing else was upgraded or reinstalled:

```
fastapi  uvicorn[standard]  python-multipart  python-dotenv  groq
```
(their own transitive dependencies — pydantic, starlette, anyio, httpx, etc. — were pulled in automatically by pip and are not separately chosen)

Already present and untouched: `torch`, `numpy`, `pandas`, `opencv-contrib-python`, `mediapipe`.

## Node packages (frontend)

`three` (runtime) and `vite` (dev/build) only — trimmed from `avatar_web/package.json`'s fuller list (`@mediapipe/tasks-vision`, `@pixiv/three-vrm*`, `fflate`, `three-mediapipe-rig`) because the copied `motion-retarget.js` / `avatar-spatial-diagnostics.js` / new `main.js` only import `three`.
