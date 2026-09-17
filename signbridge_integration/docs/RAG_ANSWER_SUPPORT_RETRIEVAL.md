# Answer-supporting RAG retrieval fix (Database Chapter 1 definition failure)

## The real-world failure

A real browser test asked **"ما هي الداتابيس"**. Routing worked
(`routed_subject = Database Chapter 1`), but Groq correctly refused to
answer, because the only two retrieved sources were **"Database
Administrator"** and **"Database Applications Examples (Cont.)"** - neither
of which defines what a database is. Groq's strict grounding rules are
supposed to refuse rather than guess, so this refusal was the *correct*
behavior given what retrieval handed it; the bug was upstream, in content
coverage and retrieval ranking, not in the LLM layer.

## Why the previous tests passed despite this failure

`backend/tests/test_rag_routing.py`'s `RetrievalSmokeTests` only asserted
two things: retrieval returned a non-empty result list, and at least one
result's `domain` label matched the expected subject. A chunk can satisfy
both of those checks while being completely useless for answering the
actual question - which is exactly what happened: "Database Administrator"
*is* a Database Chapter 1 chunk, so the old test passed, even though it
cannot answer "what is a database". No prior test ever inspected retrieved
*content* for the concepts a foundational question actually needs.

## Content audit (Section 2)

Audited all four subjects for foundational-question coverage
(what is X / define X / explain X / used for / how does X work / difference
between X and Y / main operations / example of X):

| Subject | Finding |
|---|---|
| **Database Chapter 1** | **Real content gap.** The original `01_Database_Chapter1.txt` Slide 1 is title-only ("Chapter 1: Introduction"); no chunk anywhere in the chapter contains a direct database definition. The closest supporting material is a paragraph inside Slide 2's "Detailed Explanation" (see "Grounding" below). |
| Database Chapters 2-4 | No gap. Each chapter's own Slide 1 already contains explicit "ما هو X؟" headings, a "Possible Student Questions" section, and real definitional content (e.g. Ch2 Slide 1 already has a heading "4. ما هو `E-R Model`؟"). |
| **Stack** | No content gap, but a **retrieval bug**: Slide 1 ("Building a Stack Using an Array") is an array-implementation walkthrough, not a definition; the real definition/LIFO explanation is Slide 4 ("Stack Definition and the `LIFO` Principle"). A pre-existing boost in `retrieve_stack()` incorrectly targeted slide 1 for definition questions. |
| **Queue** | No gap. Slide 1 is genuinely titled "Introduction" and contains the real FIFO/front/rear definition. |
| **Pointers** | No content gap - Slide 5 ("Pointer Data Type and Pointer Variables") already contains a clean definition ("الـPointer هو متغير محتواه عنوان في الذاكرة" / "a pointer is a variable whose content is a memory address") - but it was **out-ranked** by other slides for a generic "ما هو البوينتر؟" query because every Pointers chunk shares the same repeating 8-section heading template, so heading text alone could not distinguish the definition slide from any other. |

## Grounded supplemental foundational chunk (Database Chapter 1 only)

Two new chunks were added: `database-ch1-slide-01-part-01` and
`database-ch1-slide-01-part-02`, using `slide_number: 1` (previously
unused - the original Slide 1 has no chunk at all, so this creates no
collision) and a slide title that is explicitly labeled:
**"Chapter 1 Introduction - Database Definition (Grounded Supplemental
Foundational Chunk)"**.

**Every sentence is grounded in, and only restates, this existing passage**
already present in `01_Database_Chapter1_enriched.md`, Slide 2
("Database Applications Examples"), section "## 3. Detailed Explanation"
(chunk `database-ch1-slide-02-part-02`):

> "الفكرة الأساسية من هذا السلايد هي أن قواعد البيانات لا تُستخدم في مجال
> واحد فقط... كل مؤسسة تتعامل مع كمية كبيرة من البيانات، وهذه البيانات
> تحتاج إلى أن يتم تخزينها وتنظيمها واسترجاعها بطريقة فعالة... Real-World
> Application → Generates Data → Database Stores and Organizes the Data →
> Applications Access and Manage the Data... بالتالي، قاعدة البيانات تعتبر
> جزءًا أساسيًا من الأنظمة التي تحتاج إلى تخزين وإدارة كميات منظمة من
> المعلومات."

No fact, number, or technical claim was added beyond that passage. The new
chunk explicitly documents this grounding inline (a "Grounding Notice"
section) and states, in its own "Limited Note About DBMS" section, that
Chapter 1's own content does not provide an independent definition of DBMS
or an explicit Database-vs-DBMS distinction beyond the acronym expansion -
per the "only if supported by the course materials" instruction, no such
distinction was invented.

**Nothing existing was deleted or rewritten.** `01_Database_Chapter1.txt`
is untouched; `01_Database_Chapter1_chunks.jsonl`'s 73 original lines are
untouched, with 2 new lines appended; `01_Database_Chapter1_enriched.md`'s
existing content (through its "Slide 24 - End of Chapter 1" marker) is
untouched, with a new, clearly-labeled section appended after it.

## Retrieval/reranking fix (Section 3)

Added to `backend/legacy_runtime/rag/query_understanding.py`:

- **`detect_intents(question)`** - classifies a question into zero or more
  of five reusable intents: `definition`, `comparison`, `operation`,
  `example`, `purpose`, from bilingual marker phrases (a question can carry
  more than one, e.g. "what is the difference between a stack and a queue"
  is both definition and comparison).
- **`rerank_by_intent(question, candidates)`** - a conservative, additive
  reranking layer applied on top of a domain module's own BM25 (+ any
  existing domain-specific boosts) score, in the same 15-40 point range as
  the boosts already used in `retrieve_stack`/`retrieve_queue`. It never
  replaces BM25; it only reorders candidates BM25 already scored above
  zero. Signals used (all read from fields already present in every
  domain's chunk schema - `slide_title`, `section_headings`, `text` - so it
  works identically across Stack/Queue/Pointers/all four Database
  chapters with no per-domain configuration):
  - a definition/"what is"/"core idea" heading marker (split into a strong
    tier - `ما هو`, `ما هي`, `what is`, `تعريف`, `معنى`, `الفكرة الأساسية`
    - and a weaker tier - bare `definition`/`define` - since the bare
      English word also appears inside unrelated technical terms like
      "Data Definition Language");
  - **`_contains_definitional_sentence`**: a generic "\<concept\> is/هو a
    ..." copula-proximity check over the chunk's own body text (not just
    headings) - this is what let Pointers Slide 5's real one-line
    definition, buried under a generic "3. الفكرة الأساسية" heading shared
    by every Pointers slide, outrank other same-topic slides. Tightened
    during development to require the copula within 3 tokens *after* the
    concept term (subject-first), because a looser "anywhere on the line"
    version also matched incidental, non-definitional uses of "is"/"هو"
    (verified against real content: "only p is the pointer variable, not
    q" and a Null-Pointer aside both false-matched under the looser check);
  - a **"Possible Student Questions" match**: extracts that section from a
    chunk's own text and checks token overlap with the asked question -
    this is the mechanism, combined with the new chunk's own listed
    question variants, that makes all 11 required Database phrasings rank
    the new chunk first;
  - comparison/operation/example/purpose each get a smaller, similarly
    content-based bonus (heading markers, or question/heading token
    overlap for operation intent).

  **Deliberately does not use slide/part number as a signal.** An earlier
  version boosted `slide_number == 1` for definition intent; checking it
  against real content found this **false** for Stack (slide 1 is an array
  walkthrough, not a definition) - baking in a numbering assumption would
  have made that domain's ranking worse. Every signal is judged from a
  chunk's own title/headings/text instead.

- `backend/legacy_runtime/rag/signbridge_router.py::retrieve_domain()` now
  asks each domain's own retrieve function for a **wider** candidate pool
  (`max(top_k * 4, 8)`) so the reranker has real alternatives to promote,
  then calls `rerank_by_intent()` and truncates back to `top_k`. Intent
  detection and the "Possible Student Questions" match both use the raw,
  un-enriched question, matching the existing rule already followed by
  `retrieve_stack`/`retrieve_queue`'s own definition/comparison heuristics.
- Also fixed the pre-existing `retrieve_stack()` boost that targeted
  `slide_number == 1` for a Stack definition question - corrected to
  `slide_number == 4` (verified to be the actual "Stack Definition and the
  LIFO Principle" slide), with an inline comment explaining why, so a
  future reader does not reintroduce the same mistake.

No query enrichment/alias expansion was changed - `enrich_query_for_retrieval`
still only appends up to 2 canonical terms per domain, so this fix does not
add the "over-expanding a short query" dilution risk the task warned against.

## Index rebuild (Section 4)

Only the Database Chapter 1 BM25 index was rebuilt, using the project's own
existing generation process (no new script):

```
cd backend/legacy_runtime/rag
python 01_Database_Chapter1_rag.py --build
```

Output: `data/vector_store/01_Database_Chapter1_bm25_index.json`, now 75
documents (73 original + 2 new). Verified before rebuilding: no existing
`slide_number == 1` chunk and no `chunk_id` collision (`database-ch1-slide-01-part-01/02`
did not previously exist). Verified after rebuilding: the backend's own
`load_index()` path (`signbridge_router.py`) loads the rebuilt index
without error, and the new chunks are retrievable end-to-end (see live
validation below). No other subject's index was touched.

## Files changed

| File | Change |
|---|---|
| `backend/legacy_runtime/rag/data/processed/01_Database_Chapter1_chunks.jsonl` | Appended 2 new chunks (`database-ch1-slide-01-part-01`, `-02`). 73 existing lines untouched. |
| `backend/legacy_runtime/rag/data/processed/01_Database_Chapter1_enriched.md` | Appended one new, clearly-labeled documentation section after the existing "End of Chapter 1" marker. Nothing above it touched. |
| `backend/legacy_runtime/rag/data/vector_store/01_Database_Chapter1_bm25_index.json` | Rebuilt via the existing `--build` CLI (75 documents). |
| `backend/legacy_runtime/rag/query_understanding.py` | Added `detect_intents`, `rerank_by_intent`, and supporting helpers/marker tables. |
| `backend/legacy_runtime/rag/signbridge_router.py` | `retrieve_domain()` now retrieves a wider pool and applies `rerank_by_intent()`; fixed the Stack slide-number bug in `retrieve_stack()`. |
| `backend/tests/test_rag_routing.py` | Added `AnswerSupportRetrievalTests` and `GroundingRefusalSafetyTests`; kept `RetrievalSmokeTests` with a docstring explaining why it is no longer sufficient alone. |
| `docs/RAG_ANSWER_SUPPORT_RETRIEVAL.md` | This report (new). |

Nothing else was modified.

## Strengthened tests (Section 5)

`AnswerSupportRetrievalTests` (new) replaces "non-empty + correct domain"
with real content checks:

- **`test_database_definition_queries_are_answer_supporting`** - runs all
  11 required exact variants (`ما هي قاعدة البيانات؟`, `ما هي الداتا بيس؟`,
  `ما هي الداتابيس؟`, `شو يعني داتا بيس؟`, `شو الداتا بس؟`,
  `عرف قاعدة البيانات`, `اشرحلي الداتابيس`, `what is a database?`,
  `define database`, `explain DB`, `what is DBMS?`). For each: asserts the
  top 2 sources are **not** only "Database Administrator"/"Database
  Applications Examples" (the exact failure mode), asserts the retrieved
  text contains a store/organize/manage/retrieve concept marker, and
  asserts the new grounded supplemental chunk is the actual top-ranked
  result.
- **`test_stack_definition_question_is_answer_supporting`** - asserts
  `ما هو المكدس؟` ranks Slide 4 (the LIFO definition) first, not Slide 1.
- **`test_queue_definition_question_is_answer_supporting`** /
  **`test_pointer_definition_question_is_answer_supporting`** - asserts the
  retrieved text actually contains FIFO/طابور or the pointer's
  memory-address concept respectively, and (for Pointers) that the exact
  definition slide is ranked first.
- **`test_ambiguous_stack_vs_queue_retrieves_both_domains`** - asserts
  `ما الفرق بين stack و queue؟`'s retrieved *results* (not just routed
  domains) contain both a `"Stack"` and a `"Queue"` labeled source.

`GroundingRefusalSafetyTests` (new) guards Section 1's requirement as an
automated regression:

- **`test_strict_grounding_rules_are_present_and_not_weakened`** - asserts
  the system prompt inside `generate_grounded_response` still contains its
  strict-grounding phrases (so a future edit cannot silently soften them).
- **`test_unrelated_questions_have_no_confident_single_domain`** - asserts
  unrelated questions still get `fallback_all` routing confidence, never a
  confident wrong-domain guess.

All 76 backend tests pass (`python -m unittest discover -s backend/tests -v`),
including the pre-existing 69 and the 7 new ones above.

## Real end-to-end validation (Section 6)

Ran real Groq API calls through `app.rag.service.answer_question()` using
the configured `backend/.env`. The key was never read or printed by any
script in this task - only the application's own existing settings loader
touched it.

| Query | routed_subject | Top sources | Direct answer? | Refused? | Irrelevant sources? |
|---|---|---|---|---|---|
| ما هي الداتابيس؟ | Database Chapter 1 | grounded supplemental chunk (x2), DDL | **Yes** - "قاعدة البيانات هي الجزء في النظام الذي يخزن البيانات..." | No | No |
| ما هي الداتا بيس؟ | Database Chapter 1 | grounded supplemental chunk (x2), DDL | **Yes**, same definition | No | No |
| ما هو المكدس؟ | Stack | Slide 4 (LIFO definition) x2, Slide 1 | **Yes** - LIFO/Push/Pop/Top explained correctly | No | No |
| شو يعني كيو؟ | Queue | Slide 1 (Introduction), 2 others | **Yes** - FIFO/front/rear explained correctly | No | No |
| ما هو البوينتر؟ | Pointers | Slide 5 (definition), Slides 14 & 7 | **Yes** - "الـPointer هو متغير يُخزن عنوانًا في الذاكرة..." | No | No |
| ما الفرق بين stack و queue؟ | Queue + Stack | Queue Slide 1 + 2 others, Stack Slide 4 x3 | **Yes** - correctly compares both LIFO and FIFO | No | No |
| *(bonus)* what is the capital of France? | all 7 domains (safe fallback) | assorted definition/example slides from every subject | N/A | **Yes, correctly** - "I'm sorry, the provided course material does not contain information about the capital of France." | Sources were irrelevant, but Groq correctly did not answer from them |

**"ما هي الداتابيس؟" now returns a direct grounded definition instead of a
refusal, confirmed live.** The bonus unrelated-question check confirms
Section 1's requirement end-to-end: even when retrieval's safe fallback
surfaces irrelevant chunks, Groq's unweakened grounding rules still refuse
rather than fabricate a course-sounding answer.

## Remaining content gaps (honestly reported)

- Chapter 1's new supplemental chunk gives a minimal, honest note about
  DBMS (acronym expansion only) rather than a full definition or an
  explicit Database-vs-DBMS distinction, because no such content exists
  anywhere in Chapter 1's own material. If the course later adds that
  content, it should replace this note rather than have facts invented
  here to fill the gap now.
- Pointers' and Stack's underlying content was not modified (no gap was
  found - both already contain a real definition), so those fixes are
  retrieval-ranking-only. If a future course update reorganizes those
  slides, the `slide_number == 4` (Stack) anchor and the definitional-
  sentence detector (Pointers) should be re-verified against the new
  content rather than assumed still correct.
- This task did not exhaustively re-audit every foundational-question
  category (operation/example/purpose) against every slide in Database
  Chapters 2-4 sentence-by-sentence; the intent-aware reranker now applies
  there too, but only Chapter 1's definition gap and the Stack/Pointers
  ranking bugs were confirmed as real failures and fixed.

## Frozen system constraints - confirmed unmodified

Verified by re-hashing after all changes (SHA-256, unchanged from before
this task):

- `frontend/src/motion-retarget.js`: `E01788316F47903BFFBDED5A1FA1B94905969B85D91ABD826EF4D2705BA56B79`
- `frontend/src/avatar-spatial-diagnostics.js`: `5B421B46ADB82B0868ADE75EB7E12A26BFE0B86D7F8568474E7F993117424806`
- `frontend/public/avatars/avatar_candidate.glb`: `994E083BF8336971B16E07538BAF350C7C66630B7D998E235398D7B272A0CBA5`

No recognition model, dataset, motion file, motion manifest, avatar
calibration/playback code, webcam code, or frontend layout/color file was
opened for writing at any point in this task. No `GROQ_API_KEY` value was
read, logged, or printed by any script written for this task - only the
application's own pre-existing settings loader (`backend/.env` via
`app.rag.service`) touched it, exactly as it already did before this task.
