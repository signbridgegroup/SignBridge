# RAG robustness audit: general query normalization, aliasing, and comprehensive testing

This extends `docs/RAG_QUERY_UNDERSTANDING.md` (Arabic/bilingual normalization + alias routing, built to fix the specific "ما هي الداتا بيس؟" regression) and `docs/RAG_ANSWER_SUPPORT_RETRIEVAL.md` (the Database Chapter 1 content gap + intent-aware reranking) to the broader "must generalize across every subject and phrasing style" requirement, and adds the comprehensive, machine-readable test report this task requires.

## 1. Source inventory - is every educational file actually indexed?

Audited (read-only) every file under `backend/legacy_runtime/rag/data/processed/` and `data/vector_store/`, and searched the entire repository for any other candidate educational source material or a second RAG implementation.

**Finding: every educational source file in this repository is already wired into the live RAG system.** No orphaned content, no second index, no unwired subject exists anywhere on disk:

| Domain | Chunks file (wired via `DEFAULT_CHUNKS`) | Index file (wired via `DEFAULT_INDEX`) | Chunk count |
|---|---|---|---|
| stack | `03_Stacks_Core_chunks.jsonl` | `stack_bm25_index.json` | 74 |
| queue | `04_Queue_Core_chunks.jsonl` | `queue_bm25_index.json` | 38 |
| pointers | `05_Pointers_Core_chunks.jsonl` | `pointers_bm25_index.json` | 35 |
| database_ch1 | `01_Database_Chapter1_chunks.jsonl` | `01_Database_Chapter1_bm25_index.json` | 75 |
| database_ch2 | `01_Database_Chapter2_chunks.jsonl` | `01_Database_Chapter2_bm25_index.json` | 208 |
| database_ch3 | `01_Database_Chapter3_chunks.jsonl` | `01_Database_Chapter3_bm25_index.json` | 172 |
| database_ch4 | `01_Database_Chapter4_chunks.jsonl` | `01_Database_Chapter4_bm25_index.json` | 144 |

The sibling `.txt` (raw source) and `_enriched.md` files are inert reference material - no `_rag.py` module reads them at runtime, only their `_chunks.jsonl` derivative. `backend/app/rag/service.py::answer_question()` calls `signbridge_router.answer_question()` directly and nothing else, confirming there is no side-channel content source bypassing this inventory.

**One documentation-drift issue found (not a functional bug):** each domain's `*_chunks_summary.txt` (a stale build report, never read by any code) no longer matches its sibling `.jsonl`'s actual chunk count/granularity - most notably Pointers, whose summary describes a 70-chunk (2-per-slide) build while the live `.jsonl` has 35 chunks (1-per-slide, all 8 documented sections merged into one chunk per slide). This does not affect retrieval - the actual `.jsonl` is well-formed and complete - but the summary should be regenerated whenever a domain's chunks are next rebuilt, so it doesn't mislead a future audit.

No duplicate/backup files exist in this repository (`excluded_files_summary.md` confirms five `*_enriched_backup.md` files and a stray `data/processed/`-level database index were already filtered out before this repo was assembled).

## 2. General alias/synonym system audit

`backend/legacy_runtime/rag/query_understanding.py::DOMAIN_ALIASES` already covers all 7 wired domains (built during earlier work in this session to fix the original "داتا بيس" regression) - this task's requirement to generalize beyond "a few manually added database examples" was checked against the exact broader example list this task specifies:

| Example from the task | Alias present? |
|---|---|
| ما هي قاعدة البيانات / شو هي قاعدة البيانات | yes (`قاعدة بيانات`, + definite-article-stripping handles `قاعدة البيانات`) |
| ما هي الداتا بيس / شو يعني داتابيس | yes (`داتا بيس`, `الداتا بيس`, `داتابيس`, `الداتابيس`, `داتا بيز`, `داته بيس`, `داتا بيسز`, `داتا`) |
| what is a database / define database / database meaning | yes (`database`, `databases`, `dbms`, `db`) |
| ما هو الستاك / شو يعني stack / اشرحلي المكدس | yes (`stack`, `مكدس`, `المكدس`, `ستاك`, `الستاك`, `ستك`, `الستك`, `ستكات`) |
| كيف بشتغل ال stack | yes (exact substring `stack` still matches regardless of the surrounding `ال `/code-switch wrapping) |
| ما هو ال queue / اشرح الطابور | yes (`queue`, `طابور`, `الطابور`, `كيو`, `الكيو`, `كيوز`, `كيوات`, `كيوه`) |
| ما الفرق بين push و pop / شو بعمل push؟ / شو بعمل pop؟ | yes (`push`, `pop` are canonical Stack keywords, matched + comparison-intent detected) |
| ما هو pointer / شو يعني بوينتر | yes (`pointer`, `مؤشر`, `بوينتر`, `البوينتر`, `بوينترز`) |
| ما هو primary key / شو يعني المفتاح الأساسي | yes (`primary key`, `مفتاح اساسي`, `المفتاح الاساسي`, `مفتاح أساسي`, `المفتاح الأساسي`) |
| ما الفرق بين schema و instance | yes (`schema`, `instance` are canonical database_ch1 keywords) |

Every alias is a spelling/transliteration/colloquial/abbreviation variant of a concept that already exists in the course content's own vocabulary - none maps two genuinely different technical concepts together (verified by construction: each domain's alias list is scoped to that domain only, and the routing tiers in `route_domains_with_confidence` never let a fuzzy match cross into an unrelated domain - see `NormalizationUnitTests.test_fuzzy_does_not_match_unrelated_short_word`/`test_fuzzy_requires_close_similarity`).

Typo tolerance is handled the same way as before: exact/alias match first (tiers 1-2), then a conservative `difflib`-based fuzzy tier (cutoff 0.84, tokens length >= 4 only) as a last resort - never a global "replace any similar-looking word" rule, and never applied to short/common tokens where it could misfire.

## 3. Hybrid retrieval - decision and justification

The task asks to "add or evaluate a hybrid approach combining semantic vector retrieval, lexical/BM25 retrieval, canonical concept aliases, [and] reciprocal rank fusion or another justified combination method" if the current system is dense-only. **This system is the opposite case: it is lexical-only (BM25) already, not dense-only**, so the applicable instruction is to evaluate whether adding a dense/embedding layer is warranted.

**Decision: do not add a dense/embedding retrieval layer in this task.** Reasoning, stated honestly rather than skipped:

- The comprehensive test report below (48 cases across every domain, 7 phrasing categories including typos and code-switching) passed at 100% using the existing BM25 + alias-canonicalization + fuzzy-typo-tolerance + intent-aware-reranking stack alone - the failure modes a dense retriever would target (paraphrase/synonym mismatch) are already substantially covered by the alias table and BM25's own term-expansion, for a corpus this size (7 domains, under 700 total chunks).
- Adding a real embedding layer would require either a paid API (explicitly disallowed: "do not add an unnecessary paid service") or a local sentence-embedding model - a substantial new dependency and multi-hundred-MB model download with no offline fallback guaranteed in this environment, for a corpus small enough that BM25 already indexes and scores exhaustively in milliseconds.
- What this system already has *is* meaningfully hybrid in the sense the task cares about: lexical BM25 scoring, combined with canonical-alias query enrichment, combined with a separate conservative reranking layer (`query_understanding.rerank_by_intent`, added in earlier work this session) that re-scores candidates using structural signals (heading markers, an "is-a" definitional-sentence detector, "Possible Student Questions" overlap) BM25 alone doesn't see - three different signal sources combined, just not via vector similarity.

If a future course expansion grows the corpus by an order of magnitude, or introduces genuinely paraphrastic queries the alias table can't anticipate, this decision should be revisited - the honest limitation is recorded, not hidden.

## 4. Chunking audit

Spot-checked whether definitions are split away from their concept title/examples/headings (per the task's chunking-audit requirement): every chunk already carries `slide_title`, `section_headings`, and `rag_keywords` metadata alongside the text, and each chunk is a contiguous group of a slide's own `## N. ...` markdown sections (confirmed in earlier work this session, see `docs/RAG_ANSWER_SUPPORT_RETRIEVAL.md`), so a definition and its immediate explanation/example generally stay together within one chunk. The one gap found and fixed in earlier work this session was Database Chapter 1's missing introductory definition chunk (added as a clearly-labeled grounded supplemental chunk, `database-ch1-slide-01-part-01/02`) - re-verified still present and correctly ranked first for every required definition query in the comprehensive report below. No other domain showed a comparable "definition exists nowhere" gap (database_ch2/ch3/ch4 already have explicit "ما هو X؟" headings per-slide; Stack/Queue/Pointers each have a real, if sometimes mis-ranked rather than missing, definition slide - the Stack mis-ranking was fixed as part of this task's recognition-adjacent RAG work, see below).

## 5. Comprehensive, machine-readable test report

New: `backend/tests/rag_test_report.py` (standalone generator + JSON writer) and `backend/tests/test_rag_comprehensive_report.py` (wires it into the standard test run with an 85% pass-rate floor - not 100%, since a couple of corpus entries are intentionally aspirational loose-transliteration phrasings, and the task explicitly warns against claiming complete robustness).

Real run against the live router/retrieval (no mocking):

```
Total cases: 48
Passed: 48
Failed: 0
Pass rate: 100.0%

Per subject:
  database_ch1: 12/12   stack: 12/12   queue: 6/6   pointers: 6/6
  database_ch2: 4/4     database_ch4: 2/2   stack+queue: 1/1   unrelated: 5/5

Per language form:
  formal_arabic: 9/9   colloquial_arabic: 7/7   transliteration: 4/4
  english: 9/9   typo: 5/5   code_switch: 13/13   incomplete: 1/1
```

Each case checks (not merely "HTTP 200" or "non-empty"): the correct subject was routed (or, for the 5 negative/unrelated cases, that routing fell back to the safe multi-domain fallback rather than a confident wrong guess), and that the retrieved text actually contains the concept markers needed to answer the question (e.g. store/organize/manage/retrieve for a database definition, LIFO/push/pop for Stack, an entity/relationship marker for E-R Model). The full per-case JSON (including exact retrieved chunk IDs) is written to `backend/tests/rag_test_report.json` on every run. This is 48 cases, not an exhaustive enumeration of every possible phrasing - "100%" here means "100% of this corpus", stated precisely rather than as a blanket robustness claim.

## 6. Diagnostics

`query_understanding.route_domains_with_confidence()` already exposes which tier matched (`exact_or_alias`/`fuzzy`/`fallback_all`) for logging; `rerank_by_intent()`'s per-candidate bonus reasoning is derivable from `detect_intents()` + the heading/definitional-sentence/possible-questions checks it applies. These are available to call directly for developer/diagnostic logging but are not injected into the normal student-facing API response, per the task's "do not expose excessive internal diagnostics in the normal student interface" instruction - the same posture already taken for the recognition router's `candidate_scores`/`quality` fields (Part 1).

## 7. What was not changed

No embedding/vector-store dependency was added (see §3). No existing domain module's own local `EXPANSIONS`/`normalize`/`tokenize`/`bm25` was touched. No chunk file was rewritten or deleted - only the two already-documented Database Chapter 1 supplemental chunks from earlier work this session remain the sole content addition. `signbridge_llm.py`'s strict grounding system prompt is unchanged (still verified present by `GroundingRefusalSafetyTests` in `test_rag_routing.py`).
