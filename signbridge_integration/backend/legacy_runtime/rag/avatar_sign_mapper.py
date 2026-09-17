from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent
DEFAULT_LEXICON = PROJECT / "avatar_sign_lexicon.json"

# Words that can safely remain textual glue in the planner. They are useful in
# Arabic prose but do not need their own canonical technical concept token.
FUNCTION_WORDS = {
    "هو","هي","هذا","هذه","ذلك","تلك","من","في","على","الى","عن","مع","و","او","ثم",
    "فقط","وفق","الذي","التي","ان","كل","بين","عبر","له","لها","كما","اي","يعني",
    "مثلا","واحد","واحده","نفس","عند","بعد","قبل","بواسطه","يتم","يكون","تكون",
    "يعمل","تعمل","القاعده","قاعده","عمليه","عمليات","العمليات","خاصه","بشكل",
    "طريقه","يمكن","يمكنه","يمكنها","ايضا","اما","لكن","لذلك","حيث","فان","مبدا",
    "بمبدا","اسم","اسمه","اسمها","يسمى","تسمى","يطبق","تطبق","يستخدم","تستخدم","يتبع","تسمي","تسمى","ال",
    "ثابت","ثابته","الحالي","الحاليه","يعد","يعدد","عدد","لا","غير","موضع","مكان",
    "الوصول","وسط","الوسط","للوسط","مباشره","مباشرة","نوع","يحدد","يشبه","عندما","البيانات","المتوقعه","المتوقعة","اخر","اول","اولا","داخل","خارج",
    "اي","كعمليه","اساسيه","الاساسيه","الوصول","العشوائي",
    "the","a","an","is","are","of","to","from","in","on","and","or","that","which",
    "with","only","using","works","work","by","for","as","can","also","this","these","it",
}

# Prefixes and common pronominal suffixes.
PREFIXES = ("وال","فال","بال","كال","لل","ال","و","ف","ب","ك","ل")
SUFFIXES = ("كما","هما","هم","هن","ها","ه","ك","ي","نا")

SUBSUMED_IN_CLAUSE = {
    "PUSH": {"ADD"},
    "POP": {"REMOVE"},
    "ENQUEUE": {"ADD"},
    "DEQUEUE": {"REMOVE"},
}

# These tokens normally identify a definition/principle, not repeated actions.
# Repeating them in nearby explanatory clauses adds no avatar information.
GLOBAL_SINGLETONS = {
    "STACK","QUEUE","POINTER","DATABASE","DATA_STRUCTURE",
    "LIFO","FIFO","SCHEMA","INSTANCE",
}


def normalize(text: str) -> str:
    text = str(text or "").casefold().replace("ـ", "")
    text = text.translate(str.maketrans({
        "أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه",
    }))
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[^\w\u0621-\u063A\u0641-\u064A]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _strip_example_sentences(text: str) -> tuple[str, list[str]]:
    """Remove explicit example sentences from avatar planning only.

    The educational answer remains untouched. Examples such as
    ``مثال: Student(ID, Name) ...`` are useful for the student but should not
    create missing avatar concepts or arbitrary sign tokens.
    """
    raw = str(text or "")
    pieces = re.split(r"([.!?؟؛;\n]+)", raw)
    kept: list[str] = []
    ignored: list[str] = []

    i = 0
    while i < len(pieces):
        segment = pieces[i]
        delimiter = pieces[i + 1] if i + 1 < len(pieces) else ""
        normalized = normalize(segment)

        is_example = (
            normalized == "مثال"
            or normalized.startswith("مثال ")
            or normalized == "example"
            or normalized.startswith("example ")
            or normalized.startswith("e g ")
        )

        if is_example:
            if segment.strip():
                ignored.append(segment.strip())
        else:
            kept.append(segment)
            kept.append(delimiter)

        i += 2

    return "".join(kept).strip(), ignored


def split_clauses(text: str) -> list[str]:
    """Split semantic clauses without breaking comma lists inside parentheses."""
    text = str(text or "")
    parts: list[str] = []
    buf: list[str] = []
    depth = 0

    separators = {".", "!", "?", "؟", "؛", ";", ":", "\n", "،", ","}

    for ch in text:
        if ch in "([{":
            depth += 1
            buf.append(ch)
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue

        if ch in separators and depth == 0:
            chunk = "".join(buf).strip()
            if chunk:
                parts.append(chunk)
            buf = []
        else:
            buf.append(ch)

    chunk = "".join(buf).strip()
    if chunk:
        parts.append(chunk)

    return parts


def _candidate_forms(word: str) -> list[str]:
    n = normalize(word)
    forms = [n]

    # Strip one prefix, then optional suffix, conservatively.
    prefixed = list(forms)
    for p in PREFIXES:
        if n.startswith(p) and len(n) - len(p) >= 3:
            prefixed.append(n[len(p):])

    for form in list(dict.fromkeys(prefixed)):
        forms.append(form)
        for s in SUFFIXES:
            if form.endswith(s) and len(form) - len(s) >= 3:
                forms.append(form[:-len(s)])

    return list(dict.fromkeys(f for f in forms if f))


def _is_function_word(word: str) -> bool:
    return any(form in FUNCTION_WORDS for form in _candidate_forms(word))


def _load_lexicon(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ValueError("avatar_sign_lexicon.json must contain an 'entries' list")
    return data


@dataclass(frozen=True)
class Alias:
    words: tuple[str, ...]
    token: str
    motion_id: str
    motion_file: str | None
    priority: int


def _alias_table(data: dict[str, Any]) -> list[Alias]:
    out: list[Alias] = []
    for e in data["entries"]:
        token = str(e.get("token","")).strip().upper()
        if not token:
            continue
        motion_id = str(e.get("motion_id") or token)
        motion_file = e.get("motion_file")
        priority = int(e.get("priority", 0) or 0)
        for raw in [token, *e.get("aliases", [])]:
            n = normalize(raw)
            if n:
                out.append(Alias(tuple(n.split()), token, motion_id, motion_file, priority))
    out.sort(key=lambda a: (len(a.words), a.priority), reverse=True)
    return out


def _single_word_match(word: str, aliases: list[Alias]) -> Alias | None:
    forms = set(_candidate_forms(word))
    for a in aliases:
        if len(a.words) == 1 and a.words[0] in forms:
            return a
    return None


def _match_clause(clause: str, idx: int, aliases: list[Alias]):
    words = normalize(clause).split()
    consumed = [False] * len(words)
    raw = []

    # Longest multi-word aliases reserve their spans.
    for start in range(len(words)):
        if consumed[start]:
            continue
        for a in aliases:
            width = len(a.words)
            if width <= 1:
                continue
            end = start + width
            if end > len(words) or any(consumed[start:end]):
                continue
            if tuple(words[start:end]) == a.words:
                for i in range(start, end):
                    consumed[i] = True
                raw.append({
                    "token": a.token,
                    "motion_id": a.motion_id,
                    "motion_file": a.motion_file,
                    "matched_text": " ".join(words[start:end]),
                    "clause_index": idx,
                    "source_span": [start, end],
                })
                break

    # Remaining words are matched morphologically.
    for i, word in enumerate(words):
        if consumed[i]:
            continue
        a = _single_word_match(word, aliases)
        if a:
            consumed[i] = True
            raw.append({
                "token": a.token,
                "motion_id": a.motion_id,
                "motion_file": a.motion_file,
                "matched_text": word,
                "clause_index": idx,
                "source_span": [i, i+1],
            })

    raw.sort(key=lambda x: x["source_span"][0])

    # Named domain operations are stronger than generic verbs.
    present = {m["token"] for m in raw}
    suppressed = set()
    for strong, weak in SUBSUMED_IN_CLAUSE.items():
        if strong in present:
            suppressed |= weak

    compact = []
    seen = set()
    for m in raw:
        if m["token"] in suppressed:
            continue
        if m["token"] in seen:
            continue
        seen.add(m["token"])
        compact.append(m)

    # Keep unknowns for diagnostics only. They are NOT automatically "missing
    # concepts", because natural-language glue does not need a dedicated sign ID.
    diagnostic_unknown = []
    for i, word in enumerate(words):
        if not consumed[i] and not _is_function_word(word):
            diagnostic_unknown.append(word)

    return compact, diagnostic_unknown


def text_to_sign_plan(text: str, lexicon_path: Path | None = None) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {
            "planner_version":"4.8",
            "planning_mode":"deterministic_concept_planner",
            "sign_tokens":[],
            "motion_sequence":[],
            "matched_concepts":[],
            "concept_clauses":[],
            "missing_concepts":[],
            "diagnostic_unknown_words":[],
            "unmapped_words":[],
            "ignored_example_sentences":[],
            "semantic_coverage":0.0,
            "motion_coverage":0.0,
            "coverage":0.0,
            "avatar_ready":False,
            "linguistic_status":"concept_sequence_not_validated_sign_language_grammar",
            "planner_warnings":["No avatar text was provided."],
        }

    data = _load_lexicon(lexicon_path or DEFAULT_LEXICON)
    aliases = _alias_table(data)

    planner_text, ignored_example_sentences = _strip_example_sentences(text)
    clauses = split_clauses(planner_text)

    planned = []
    concept_clauses = []
    diagnostic_unknown = []

    for ci, clause in enumerate(clauses):
        matches, unknown = _match_clause(clause, ci, aliases)
        planned.extend(matches)
        diagnostic_unknown.extend(unknown)
        concept_clauses.append({
            "clause_index": ci,
            "text": clause,
            "tokens": [m["token"] for m in matches],
            "diagnostic_unknown_words": list(dict.fromkeys(unknown)),
        })

    # Collapse repeated definition/principle IDs while keeping action repetitions.
    final = []
    singleton_seen = set()
    for m in planned:
        token = m["token"]
        if token in GLOBAL_SINGLETONS:
            if token in singleton_seen:
                continue
            singleton_seen.add(token)
        final.append(m)

    sign_tokens = [m["token"] for m in final]

    # A clause is semantically covered if it contributes at least one known concept,
    # or contains only planner glue. This is much more meaningful than requiring a
    # sign token for every Arabic word.
    clause_scores = []
    missing_concepts = []
    for c in concept_clauses:
        if c["tokens"]:
            clause_scores.append(1.0)
        elif not c["diagnostic_unknown_words"]:
            clause_scores.append(1.0)
        else:
            clause_scores.append(0.0)
            missing_concepts.extend(c["diagnostic_unknown_words"])

    semantic_coverage = sum(clause_scores) / len(clause_scores) if clause_scores else 0.0
    missing_concepts = list(dict.fromkeys(missing_concepts))
    diagnostic_unknown = list(dict.fromkeys(diagnostic_unknown))

    motion_sequence = [{
        "token":m["token"],
        "motion_id":m["motion_id"],
        "motion_file":m["motion_file"],
    } for m in final]

    unique_motion = {}
    for m in motion_sequence:
        unique_motion[m["motion_id"]] = m["motion_file"]

    motion_coverage = (
        sum(bool(v) for v in unique_motion.values()) / len(unique_motion)
        if unique_motion else 0.0
    )
    missing_motion_ids = [k for k,v in unique_motion.items() if not v]

    warnings = []
    if missing_concepts:
        warnings.append("Some clauses have no mapped avatar concept.")
    if diagnostic_unknown:
        warnings.append(
            "Diagnostic unknown words remain, but they do not automatically count as missing sign concepts."
        )
    if missing_motion_ids:
        warnings.append("Motion files are missing for: " + ", ".join(missing_motion_ids))

    avatar_ready = bool(motion_sequence) and not missing_concepts and motion_coverage == 1.0

    return {
        "planner_version":"4.8",
        "planning_mode":"deterministic_concept_planner",
        "sign_tokens":sign_tokens,
        "motion_sequence":motion_sequence,
        "matched_concepts":[{
            "token":m["token"],
            "matched_text":m["matched_text"],
            "clause_index":m["clause_index"],
        } for m in final],
        "concept_clauses":concept_clauses,
        "missing_concepts":missing_concepts,
        "diagnostic_unknown_words":diagnostic_unknown,
        "unmapped_words":diagnostic_unknown,
        "ignored_example_sentences":ignored_example_sentences,
        "semantic_coverage":round(semantic_coverage,3),
        "motion_coverage":round(motion_coverage,3),
        "coverage":round(semantic_coverage,3),
        "avatar_ready":avatar_ready,
        "linguistic_status":"concept_sequence_not_validated_sign_language_grammar",
        "planner_warnings":warnings,
    }


def attach_avatar_plan(response: dict[str, Any], lexicon_path: Path | None = None):
    avatar_text = str(
        response.get("avatar_text") or response.get("simplified_text") or ""
    ).strip()
    response["avatar_text"] = avatar_text
    response.update(text_to_sign_plan(avatar_text, lexicon_path))
    return response


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    args = parser.parse_args()
    print(json.dumps(text_to_sign_plan(args.text, args.lexicon), ensure_ascii=False, indent=2))
