# Webcam orientation audit and fix

## Audit performed (read-only, before any change)

Traced the complete pipeline: `getUserMedia` constraints → `<video id="camera-preview">` element → CSS → `MediaRecorder` → recorded `Blob` → `POST /api/recognize` → backend recognition (`app/recognition/service.py` → `predict_it_video.extract_video`, unchanged in this task).

| Stage | File | Finding |
|---|---|---|
| `getUserMedia` constraints | `frontend/src/camera-recorder.js` | `{ video: true, audio: false }` — no `facingMode` requested at all (browser default). |
| Preview element | `frontend/index.html` | `<video id="camera-preview" playsinline muted>`, no transform attribute. |
| CSS | `frontend/src/style.css` | `.camera-frame video` set only `width`/`height`/`object-fit` — **no `transform: scaleX(-1)` or any mirroring anywhere in the stylesheet.** Grepped the whole `frontend/src` tree for `scaleX`/`mirror`/`flip`/`facingMode` — zero matches before this fix. |
| Canvas drawing | — | None exists; the app never draws camera frames to a `<canvas>`. |
| `MediaRecorder` | `frontend/src/camera-recorder.js` | Records the raw `MediaStream` directly (`new MediaRecorder(this.stream, ...)`) — no transform applied, could not be, since `MediaRecorder` operates on the stream's actual pixel data, not on any on-screen rendering. |
| Recorded/uploaded Blob | `frontend/src/api.js` (`recognizeVideo`), `frontend/src/main.js` (`runRecognition`) | Sent to the backend exactly as produced by `MediaRecorder` or exactly the file the user selected in the upload panel — no client-side re-encoding, cropping, or flipping at any point. |
| Backend recognition frames | `backend/legacy_runtime/predict_it_video.py::extract_video` (not modified in this task) | Reads frames via OpenCV directly from the uploaded file, in whatever orientation the file actually has. Never touched, never should be for this fix. |
| Double-mirroring | — | Not present (there was no mirroring at all, so there could be no duplicate to remove). |
| `facingMode`/camera switching | — | No camera-switch UI exists currently; only a single, unlabeled camera request. |

## Root cause

**No mirroring was ever applied anywhere.** A front-facing camera's raw captured pixels are naturally in "true" orientation (as if someone were facing the user) rather than the conventional selfie-mirror orientation most users expect from a camera preview (raising your right hand should appear on the right side of the screen, like a mirror). With zero CSS transform on the preview `<video>` element, the app displayed the true (non-mirrored) stream directly, which reads as "reversed" or "unnatural" to a user — exactly the report. This was purely a **missing display-layer transform**, not any kind of double-flip or corrupted recording; the actual recorded/uploaded data was never mirrored and was already correct for the recognition backend.

## Fix

1. **`frontend/src/camera-recorder.js`**: requests `facingMode: 'user'` explicitly (a soft/ideal constraint — browsers fall back gracefully to any available camera if no front camera exists, never a hard failure) and tracks `this.facingMode`, exposing a pure `shouldMirrorPreview` getter (`true` unless `facingMode === 'environment'`, per the task's "rear/environment cameras should normally not be mirrored" rule). Also classifies `getUserMedia` errors by `DOMException.name` into distinct, clear Arabic messages (permission denied / no camera found / camera in use / other) instead of one generic message. **The `MediaStream`, `MediaRecorder`, and the resulting `Blob` are never transformed anywhere in this file — confirmed by inspection, and by the fact that `beginRecording`/`stopRecording` are unchanged in this fix.**
2. **`frontend/src/style.css`**: added exactly one new rule, `.camera-frame video.is-mirrored { transform: scaleX(-1); }` — a display-only CSS transform.
3. **`frontend/src/main.js`**: toggles the `is-mirrored` class on both `#camera-preview` (live view) and `#camera-playback` (post-recording review, for consistency with what the user just saw live) based on `cameraRecorder.shouldMirrorPreview`, once, right after the camera starts successfully. Also wires the richer per-state Arabic status text (see below) and passes `{ facingMode: 'user' }` explicitly when starting the camera.

**What was deliberately not done**: no flip/transform was applied to the `MediaStream`, `MediaRecorder`, the recorded `Blob`, the upload payload, or anything on the backend recognition path. No camera-switch UI or mirror toggle was added — none existed before, and the task allows but does not require one ("if useful and low-risk"); adding one now would be a small feature addition better scoped with the rest of a future camera UX pass, not bundled into a targeted orientation fix.

## Arabic camera UI states (all ten required states, verified present)

| State | Where | Text |
|---|---|---|
| Permission requested | `cameraStartButton` click, before `getUserMedia` resolves | "جارٍ طلب إذن الكاميرا…" |
| Camera ready | after `cameraRecorder.start()` succeeds | "الكاميرا جاهزة. اضغطي \"ابدئي التسجيل\"." |
| Permission denied | `_describeError`, `NotAllowedError`/`PermissionDeniedError` | "تم رفض إذن الوصول إلى الكاميرا…" |
| No camera found | `_describeError`, `NotFoundError`/`DevicesNotFoundError` | "لم يتم العثور على كاميرا متصلة بهذا الجهاز." |
| Camera already in use | `_describeError`, `NotReadableError`/`TrackStartError` | "الكاميرا قيد الاستخدام حاليًا من قبل تطبيق آخر." |
| Recording | `cameraRecordButton` click | "جارٍ التسجيل…" |
| Recording stopped | `cameraStopButton` click | "تم التسجيل. يمكنك إعادة التسجيل أو تحليل الإشارة." |
| Uploading | `cameraAnalyzeButton` click, via the new `onStatusChange` hook into `runRecognition` | "جارٍ رفع الفيديو وتحليله…" |
| Recognition completed | `runRecognition` success path | "اكتمل التعرف على الإشارة." |
| Recognition failed | `runRecognition` catch path | "تعذر التعرف على الإشارة." |

(Camera-closed state, "تم إغلاق الكاميرا", already existed and is unchanged.)

## Permission and track-lifecycle behavior (audited, unchanged — already correct)

- `getUserMedia` is only ever called inside the `cameraStartButton` click handler — never on page load, tab switch, or any other implicit trigger. Confirmed by inspection: it is the only call site in the codebase.
- `switchMode()` in `main.js` already calls `cameraRecorder.closeCamera()` whenever the user navigates to any tab other than the camera tab.
- `closeCamera()` already calls `.stop()` on every track via `this.stream?.getTracks().forEach(track => track.stop())`, and a `window.beforeunload` handler already calls it on page leave.
- Both of these existing behaviors are covered by the new `frontend/src/camera-recorder.test.mjs` tests (`closeCamera stops every track...`).

## Automated tests added

`frontend/src/camera-recorder.test.mjs` — 12 tests, run with plain Node (`npm test` or `node src/camera-recorder.test.mjs`), **no new dependency**: `CameraRecorder` has zero imports and does not touch the DOM outside of `start()`, so its pure orientation/error-mapping logic (`shouldMirrorPreview`, `_describeError`, `closeCamera`'s track-stopping) is directly testable under plain Node without a browser or `jsdom`. Covers: front vs. rear mirroring decision, every mapped error state produces a distinct message, unknown errors fall back safely, the original `DOMException` is preserved as `.cause`, and track cleanup.

## What still requires a real browser and physical webcam (cannot be automated here)

This session has no browser automation or physical camera available. The following must be verified manually:

1. **Natural mirror check**: open the camera panel, allow permission, raise your right hand — it should appear on the right side of the preview (mirror-like), not the left.
2. **Live vs. playback consistency**: record a short clip, stop, and confirm the "تم التسجيل" review video shows the same (mirrored) orientation as the live preview did, not a sudden flip.
3. **Recognition correctness with mirroring on**: record and analyze a real sign (e.g. Stack), confirm the recognized label is still correct — proving the mirror is display-only and did not affect what was actually sent to the model. (Not run here since it needs a live camera; the *code path* proving no mirror is applied to the Blob was inspected directly instead.)
4. **Permission-denied path**: deny the camera permission prompt and confirm the "تم رفض إذن الوصول" message appears (not the generic fallback).
5. **No-camera-found path**: test on a device/VM with no camera attached, or disable the camera in OS settings, and confirm "لم يتم العثور على كاميرا".
6. **Camera-in-use path**: open the camera in another application (or another browser tab) first, then try to start it here, and confirm "الكاميرا قيد الاستخدام".
7. **Track cleanup**: after closing the camera or switching tabs, confirm the browser's camera-in-use indicator (tab icon / OS indicator light) turns off.
8. **Responsive/RTL check**: confirm the mirrored preview still renders correctly and without layout shift at phone width, and that the RTL page direction does not itself affect the video's horizontal mirroring (it shouldn't — `transform: scaleX(-1)` is direction-agnostic).
9. **Mobile front/rear default**: on a phone, confirm the browser opens the front camera by default (via the `facingMode: 'user'` preference) and that it is mirrored; there is currently no UI to switch to the rear camera, so rear-camera behavior cannot be checked end-to-end without adding that control in a future pass.
