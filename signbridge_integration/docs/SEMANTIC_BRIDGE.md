# Semantic-planner token bridge — full 58-token report

Recomputed by checking, for every one of the 58 `avatar_sign_lexicon.json` tokens, whether its own token name **or one of its own human-curated aliases** (declared in the lexicon file itself — never invented here) exactly equals, after normalization, a canonical label from one of the three datasets. Arabic text uses `avatar_sign_mapper.py`'s own `normalize()` function, reused verbatim (not reimplemented); English text uses a plain casefold. No fuzzy matching, no new translations.

**This is a semantic-mapping report, not a readiness report.** A bridge existing here means the *concept* has an exact, defensible dataset counterpart — whether the *motion file* is actually `ready` is a separate, always-current fact you get from `GET /api/motions` or `AnswerResponse.resolved_motion_sequence` at request time, never from this table.

## Ready semantic tokens (bridge exists AND the underlying motion is `ready`)

**All 30 bridged tokens below are now `ready`.** The Jordanian IT + KArSL batch made the 27 tokens bridged to those two datasets ready; the subsequent Isharah DTW generation batch (`docs/ISHARAH_ALIGNMENT_INVESTIGATION.md`) made the remaining 3 isharah-only bridges (`INCREMENT`, `ATTRIBUTE`, `READ`) ready as well. Verified live via `motion_manifest.resolve_lexicon_token()` for every entry in `LEXICON_TOKEN_BRIDGE`, and asserted by `backend/tests/test_motion_contract.py::test_full_dataset_generation_does_not_invent_semantic_bridges`. This file states the *mapping and why it's exact*; `GET /api/motions` or `AnswerResponse.resolved_motion_sequence` remain the always-current source for whether a specific request's motion is actually ready at call time.

## Bridged tokens (30 of 58) — dataset selected for each

| Lexicon token | Dataset | Dataset token | How matched |
|---|---|---|---|
| STACK | jordanian_it | STACK | token itself |
| QUEUE | jordanian_it | QUEUE | token itself |
| DATA_TYPE | jordanian_it | DATATYPE | alias "datatype" |
| POINTER | jordanian_it | POINTER | token itself |
| DATABASE | jordanian_it | DATABASE | token itself |
| TOP | jordanian_it | TOP | token itself (also exact-matched isharah gloss "اعلى"; jordanian_it preferred — see selection rule below) |
| FRONT | jordanian_it | FRONT | token itself |
| REAR | jordanian_it | REAR | token itself (also matched KArSL 0280 via alias "back"; jordanian_it preferred) |
| ADD | jordanian_it | ADD | token itself (also matched KArSL 0191 and isharah "اضافه"; jordanian_it preferred) |
| ELEMENT | jordanian_it | ELEMENT | token itself |
| ARRAY | jordanian_it | ARRAY | token itself |
| STORE | jordanian_it | STORE | token itself |
| ZERO | jordanian_it | ZERO | token itself |
| FULL | jordanian_it | FULL | token itself |
| START | jordanian_it | START | token itself — its own exact dataset match, **not** the forbidden START→INITIALIZE guess |
| ADDRESS | jordanian_it | ADDRESS | token itself |
| MEMORY | jordanian_it | MEMORY | token itself (also matched isharah "ذاكره"; jordanian_it preferred) |
| VARIABLE | jordanian_it | VARIABLE | token itself |
| RELATIONSHIP | jordanian_it | RELATION | lexicon's own declared alias "relation" — **not** the forbidden RELATIONSHIP→BINARY_RELATIONSHIP guess |
| PRIMARY_KEY | jordanian_it | PRIMARY_KEY | alias "primary key" |
| FOREIGN_KEY | jordanian_it | FOREIGN_KEY | alias "foreign key" |
| ROW | jordanian_it | ROW | token itself |
| EXTRA | jordanian_it | UNNECESSARY | lexicon's own declared alias "unnecessary" |
| TABLE | jordanian_it | TABLE | token itself (also matched KArSL 0325; jordanian_it preferred) |
| COLUMN | jordanian_it | COLUMN | token itself |
| NOT | karsl | 0069 | alias "لا" == KArSL 0069's `label_ar` exactly (also matched isharah "لا"; karsl preferred) |
| SELECT | karsl | 0185 | alias "يختار" == KArSL 0185's `label_ar` exactly (also matched isharah "اختيار"; karsl preferred) |
| INCREMENT | isharah | زياده | alias "زياده" == an Isharah gloss exactly — generated in the Isharah DTW batch, now `ready` |
| ATTRIBUTE | isharah | صفه | alias "صفه" == an Isharah gloss exactly — generated in the Isharah DTW batch, now `ready` |
| READ | isharah | قراءه | alias "قراءه" == an Isharah gloss exactly — generated in the Isharah DTW batch, now `ready` |

**Selection rule when a token matched more than one dataset**: `jordanian_it` preferred over `karsl` over `isharah` — this was a tie-breaking preference among *equally exact* matches (never a quality judgment), originally chosen because jordanian_it/karsl were the datasets that could be generated first. Now that all three datasets are fully generated, the rule is preserved for consistency/reproducibility rather than because isharah is in any way less available.

## Missing semantic tokens (28 of 58) — no exact match in any dataset

```
DATA_STRUCTURE, LIFO, FIFO, PUSH, POP, ENQUEUE, DEQUEUE, REMOVE, LENGTH,
DECREMENT, EMPTY, MAX_SIZE, VALUE, REFERENCE, SCHEMA, INSTANCE, ENTITY,
NORMALIZATION, FUNCTIONAL_DEPENDENCY, PARTIAL_DEPENDENCY,
TRANSITIVE_DEPENDENCY, ONE_NF, TWO_NF, THREE_NF, CAPACITY, CHANGE,
CANDIDATE_KEY, UNIQUE
```

These are real CS-education concepts with no video/sign clip in any of the three datasets under an exactly-matching label or alias. Recognition-vocabulary coverage and semantic-planner coverage are genuinely different numbers — now that physical motion coverage across the three datasets is 1,341/1,347 (99.6%, see `docs/MOTION_COVERAGE.md`), this list of 28 is unchanged and will stay unchanged regardless of further dataset motion generation, because these concepts simply have no corresponding dataset entry to bridge from, exact-match or otherwise. Closing this gap would require either the dataset owners recording new signs for these specific 28 concepts, or a human linguist confirming a non-exact correspondence — neither of which this integration does automatically.
