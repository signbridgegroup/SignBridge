# RAG robustness: Arabic/bilingual query understanding

## Audit performed before any change

Read `signbridge_router.py` (routing + orchestration), every domain module's `normalize`/`tokenize`/`EXPANSIONS`/`expanded_query`/`bm25`/`retrieve` (`stack_rag.py`, `queue_rag.py`, `pointers_rag.py`, `01_Database_Chapter{1-4}_rag.py`), and the BM25 index/chunk files under `data/processed/` and `data/vector_store/`. Confirmed content inventory: 4 subjects (Stack, Queue, Pointers, Database — the latter split into 4 retrieval sub-indexes, chapters 1–4, already correctly treated as one subject with sub-topics, not 4 domains, by the existing `specific_db` displacement logic in `route_domains()`).

## Root cause of the confirmed failure ("ما هي الداتا بيس؟")

`signbridge_router.py`'s `ROUTING_KEYWORDS["database_ch1"]` listed only `"database"`, `"dbms"`, `"قواعد البيانات"`, `"قاعدة بيانات"`, and a handful of chapter-1 concept words — no transliteration (`داتا بيس` / `داتابيس`) and no colloquial variant. Routing was (and remains) a normalized-substring match against this fixed list; a transliterated/colloquial phrase absent from the list produced zero matches, so `route_domains()` fell through to its existing "no evidence → every domain" fallback. This is not a crash and not a wrong-domain guess (the existing fallback was already safe), but it meant the Database-specific question got diluted across all 7 domain sub-indexes instead of routing cleanly to Database — and would have looked, from the API's `routed_subject` field, like the system couldn't recognize the topic at all. Also found, while building the alias list: even fully **formal** `"قاعدة البيانات"` (with the definite article) was not an exact substring match for the existing `"قاعدة بيانات"` keyword (without it) — a second, related gap from Arabic definite-article variation that the substring-only approach could not handle generally.

## What was added

### 1. `backend/legacy_runtime/rag/query_understanding.py` (new, centralized)

- **`normalize_text()`**: single source of truth for matching-time normalization — Unicode NFKC, casefold, Arabic diacritics, tatweel, alef forms (أ إ آ ٱ → ا), hamza-on-carrier forms (ؤ → و, ئ → ي), ya/alef-maqsura (ى → ي), taa marbuta → haa, Arabic **and** English punctuation, repeated whitespace. Supersedes three near-duplicate `normalize()` implementations that existed only in `signbridge_router.py` (the others, inside each domain module, are untouched — see "what was not touched" below).
- **`strip_definite_article()`** + **token-set matching**: generally solves the "البيانات vs بيانات" class of gap (not just one alias pair) by comparing definite-article-stripped token sets, not just raw substrings.
- **`DOMAIN_ALIASES`**: the centralized alias/synonym table the task asked for, one dict, one place — replacing the *idea* of scattering aliases across files (each domain module keeps its own local `EXPANSIONS` for its own BM25 query expansion, unchanged, but routing itself now reads from this one table). Every entry is a spelling/transliteration/colloquial/abbreviation variant of a concept that already existed in `ROUTING_KEYWORDS` or a domain module's own `EXPANSIONS` — nothing new was invented.
- **Bounded typo tolerance**: `difflib.get_close_matches` (Python standard library — no new dependency), applied only as a last-resort tier (see priority order below), only to tokens of length ≥ 4, with a conservative cutoff (0.84), so it tolerates roughly one substituted/missing/extra character on a real word and does not fire on short/common tokens.

### 2. `signbridge_router.py` (modified)

- `normalize()` now delegates to `query_understanding.normalize_text` (same name/signature, so every existing caller — `is_stack_queue_comparison`, `retrieve_stack`, `retrieve_queue` — is unaffected).
- `route_domains()` now delegates matching to `query_understanding.route_domains_with_confidence()`. The existing safe behaviors are unchanged and still run exactly as before: a specific Database chapter still displaces the generic `database_ch1` bucket when both match, and "no evidence for any domain" still falls back to *every* domain — never a blind single-domain guess.
- `retrieve_domain()` now enriches the query text handed to each domain module's own BM25 pipeline with that domain's canonical alias terms (`enrich_query_for_retrieval`) *before* calling `module.expanded_query()`/`module.retrieve()` — this is query expansion in front of the existing BM25, not a new retrieval architecture, and it never touches the raw `question` used for intent detection (definition/comparison heuristics) or the question ultimately sent to Groq.

### Priority order (as required)

1. **Exact** normalized-substring match against an alias phrase (highest priority).
2. **Alias/token-set** match (article-stripped, order-independent) — still an exact match of every token, not fuzzy.
3. **Fuzzy** (bounded typo tolerance) — only tried when tiers 1–2 found nothing, only on tokens ≥ 4 characters, cutoff 0.84. Cannot turn an unrelated word into a confident domain match (tested explicitly — see `NormalizationUnitTests.test_fuzzy_does_not_match_unrelated_short_word` / `test_fuzzy_requires_close_similarity`).
4. **Safe multi-domain fallback** — if nothing matched at all, every domain is retrieved (unchanged, pre-existing behavior), never a blind guess at one wrong domain.

### What was **not** touched

- No domain module (`stack_rag.py`, `queue_rag.py`, `pointers_rag.py`, `01_Database_Chapter{1-4}_rag.py`) was edited. Each keeps its own local `EXPANSIONS`, `normalize`, `tokenize`, `bm25`, `retrieve` exactly as before, and each still runs its own existing BM25 exactly as before — this integration only enriches what text reaches that pipeline.
- No BM25 index or chunk file was regenerated or modified.
- `signbridge_llm.py` is unchanged: Groq still only ever answers from the retrieved course context (`build_llm_context`), with the same strict grounding system prompt as before — this task did not touch the grounding rules, so Groq still cannot silently answer from outside knowledge when evidence is unavailable.
- The `AnswerResponse`/API contract fields are unchanged (verified by `MockedLlmContractTest`, see below).
- No `GROQ_API_KEY` was read, logged, printed, or placed in any source, test, or doc file. Tests that need "no key configured" behavior force that state deterministically via `dataclasses.replace()` on the (frozen) `Settings` object, using an explicitly-fake placeholder string (`"unit-test-placeholder-not-a-real-key"`) — never the real `backend/.env`, which was never opened, read, or modified.

## Test corpus

`backend/tests/test_rag_routing.py` — 28 tests, fully offline (real routing, real BM25 retrieval against the real course indexes; the one full-contract test mocks only the Groq call itself, never a real network request):

- Formal Arabic (6), colloquial Arabic (12), English (8), mixed-language (13), transliteration/abbreviation (2), missing-character typos (3), extra-character typos (2), substituted-character typos (2), incomplete questions (4), one explicit ambiguous multi-domain question, three unrelated/unsupported questions, plus the confirmed regression case tested both as part of the colloquial-Arabic corpus and as its own named test (`test_regression_case_routes_to_database`).
- `RetrievalSmokeTests`: confirms retrieval is not just correctly labeled but actually returns non-empty, on-topic sources for four representative cases including the regression case and two typo cases.
- `MockedLlmContractTest`: exercises the real `app.rag.service.answer_question()` end-to-end (real routing, real retrieval, mocked LLM call) and asserts every field in the task's required response contract is present.
- `NormalizationUnitTests`: 11 direct unit tests of the normalizer and the fuzzy-matching safety bounds.

## Known remaining gaps (honestly reported, not silently accepted)

- The 28 lexicon concepts with no exact dataset/course match (documented separately in `docs/SEMANTIC_BRIDGE.md` — unrelated system, the avatar semantic bridge) are a different, already-documented gap; this task's alias table is scoped to the 4 RAG subjects' own routing keywords and does not change that.
- Extremely heavy colloquial/dialectal phrasing beyond the given examples (e.g. multiple compounded typos in the same short word, or dialect words not derived from any existing keyword) is not guaranteed to route correctly — the fuzzy tier is deliberately conservative and will not stretch to cover such cases, by design, per the "must not turn an unrelated question into a confident domain match" requirement.
- Retrieval **ranking quality** within a correctly-routed domain was improved by query enrichment but not redesigned; for some phrasings the top-ranked chunk is on-topic but not the single best possible slide (e.g. a generic "database" question can surface a specific slide like "Database Administrator" rather than the chapter's own introduction slide). This is a ranking nuance, not a routing or retrieval failure, and reworking the per-domain scoring heuristics was out of scope for this task.
