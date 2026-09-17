# The motion-resolution contract: Path A vs. Path B

Two distinct resolution paths exist. They must never be confused, and neither uses fuzzy string matching.

## Path A — recognized video

```
recognized dataset (resolved_mode: it/karsl/continuous)
  + recognized token/gloss (predicted_token, or decoded_gloss_sequence for continuous)
  -> app.motions.manifest.RECOGNITION_MODE_TO_DATASET  (exact namespace bridge)
  -> exact normalization (jordanian_it_token() / karsl_token() / gloss-as-is)
  -> app.motions.manifest.resolve_recognized_token()
  -> exact dataset motion entry
  -> motion JSON
  -> avatar playback (RecognizeAndAnswerResponse.recognized_motion_sequence)
```

Verified end-to-end (see `docs/TESTING.md`): a `Stack_07_1.mp4` clip recognized as IT label `"Stack"` resolves to `predicted_token: "Stack"` → `jordanian_it_token("Stack") == "STACK"` → manifest entry `jordanian_it:STACK` → `stack_07.motion.json`, served at `GET /api/motions/jordanian_it/STACK`.

The three exact normalizations, each verified against the real checkpoint vocabulary:
- **Jordanian IT**: `label.strip().upper().replace(" ", "_")` — the same function (`jordanian_it_token`) the inventory builder and the backend resolver both import from `tools/motion_library_builder/inventory.py`, so they cannot drift apart.
- **KArSL**: `karsl_token(raw)` strips an optional `"KARSL_"` prefix (the checkpoint's own continuous-vocabulary form, e.g. `"KARSL_0001"`) and zero-pads to 4 digits — resolves the same entry whether the recognizer's raw sign_id or its `KARSL_XXXX` vocabulary form is used.
- **Isharah**: the manifest token *is* the exact gloss string. Verified: all 680 `clips.csv` gloss values were found character-for-character in the trained checkpoint's continuous vocabulary (`ckpt["vocabulary"]`), so `decoded_gloss_sequence` tokens match manifest tokens directly, no transformation needed.

## Path B — typed question

```
question -> signbridge_router (BM25 + Groq) -> avatar_sign_mapper -> sign_tokens
  -> app.motions.manifest.LEXICON_TOKEN_BRIDGE  (exact, hand-verified bridge — only 3 entries)
  -> exact dataset motion entry, if bridged
  -> motion JSON
  -> avatar playback (AnswerResponse.resolved_motion_sequence)
```

`LEXICON_TOKEN_BRIDGE` intentionally contains only `STACK`, `QUEUE`, `DATA_TYPE` — see `docs/MOTION_COVERAGE.md` for why the other 55 lexicon tokens are not bridged, and why generating the full dataset libraries would not automatically fix that (Section 15 of the integration brief: recognition/generation coverage is not semantic-lexicon coverage).

## Why they're kept separate in code

- `app/motions/manifest.py::resolve_recognized_token` (Path A) and `::resolve_lexicon_token` (Path B) are two distinct methods with two distinct exact-mapping tables (`RECOGNITION_MODE_TO_DATASET` + per-dataset normalizers vs. `LEXICON_TOKEN_BRIDGE`).
- `RecognizeAndAnswerResponse` carries `recognized_motion_sequence` (Path A) and `answer.resolved_motion_sequence` (Path B) as separate fields — a client can tell which is which and never has to guess whether a shown motion came from what was recognized or from what the RAG answer talked about.
- `backend/tests/test_motion_contract.py` asserts both paths independently, including that a token resolvable in one dataset's namespace never leaks into another's, and that no lexicon token resolves by similarity to a token it wasn't explicitly bridged to.
