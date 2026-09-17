"""Centralized Arabic/bilingual query normalization, domain-alias routing,
and bounded typo tolerance for the SignBridge RAG pipeline.

This is the single place synonym/alias lists live for routing and query
expansion, instead of scattering them across signbridge_router.py and each
domain module. It does not replace any domain module's own local
EXPANSIONS dict (those stay untouched, per-domain, and are still used by
each module's own expanded_query()) - it only adds a normalization layer
and a routing-time alias/fuzzy layer that domain-module retrieval did not
have before.

Nothing here invents educational facts: every alias is a spelling,
transliteration, abbreviation, or colloquial variant of a concept that
already exists in signbridge_router.py's own ROUTING_KEYWORDS or in a
domain module's own EXPANSIONS dict (see the inline comments below for
which existing source grounds each addition).
"""

from __future__ import annotations

import difflib
import re
import unicodedata

# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #

_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = re.compile(r"ـ")
# Arabic punctuation (؟ ، ؛ ـ already handled above) plus common English
# punctuation - normalized to a space so it never glues two words together.
_PUNCTUATION = re.compile(r"[؟،؛!.,?;:()\[\]{}\"'`/\\_+*=~<>|@#%^&\-]")
_WHITESPACE = re.compile(r"\s+")

_CHAR_MAP = str.maketrans(
    {
        # Alef forms -> bare alef
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        # Ya / alef maqsura
        "ى": "ي",
        # Taa marbuta -> haa (safe for search/matching purposes only; never
        # used to alter what is shown back to the user)
        "ة": "ه",
        # Hamza-on-carrier forms -> their plain carrier, so a missing/extra
        # hamza does not block a match
        "ؤ": "و", "ئ": "ي",
    }
)

# Token pattern shared with every domain module's own TOKEN regex
# (Arabic runs, or Latin/digit/technical-symbol runs) so tokenizing the
# same normalized text here and in a domain module agrees.
_TOKEN = re.compile(r"[A-Za-z0-9_+%/-]+|[؀-ۿ]+")


def normalize_text(text: str) -> str:
    """Robust normalization for matching only (never for display).

    Handles: Unicode NFKC, casefold, Arabic diacritics, tatweel, alef
    forms, hamza-on-carrier forms, ya/alef-maqsura, taa marbuta, Arabic and
    English punctuation, and repeated whitespace.
    """
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = _ARABIC_DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = text.translate(_CHAR_MAP)
    text = _PUNCTUATION.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize_text(text))


def strip_definite_article(token: str) -> str:
    """Strip a leading Arabic definite article (ال) for matching only.

    "البيانات" and "بيانات" must be treated as the same concept for
    routing/alias purposes. Only strips when the remainder is still a
    real word (>=2 chars) so short tokens are not mangled.
    """
    if token.startswith("ال") and len(token) - 2 >= 2:
        return token[2:]
    return token


def normalized_token_set(text: str) -> set[str]:
    return {strip_definite_article(t) for t in tokenize(text)}


# --------------------------------------------------------------------------- #
# Domain aliases (routing vocabulary)
# --------------------------------------------------------------------------- #
# Keys match signbridge_router.DOMAIN_FILES exactly. Every entry is grounded
# in an existing concept from signbridge_router.ROUTING_KEYWORDS or a domain
# module's own EXPANSIONS dict - this only adds spelling/transliteration/
# colloquial/abbreviation variants of those same concepts, never a new one.
# The first 1-2 entries per domain are the "canonical" terms used to enrich
# the retrieval query (see enrich_query_for_retrieval below); the rest are
# alternate spellings that route correctly but are not injected verbatim.
DOMAIN_ALIASES: dict[str, list[str]] = {
    "stack": [
        "stack", "مكدس",
        "stacks", "المكدس", "مكدسه", "ستاك", "الستاك", "ستك", "الستك", "ستكات",
        "push", "pop", "peek", "top", "lifo",
    ],
    "queue": [
        "queue", "طابور",
        "queues", "الطابور", "طابوره", "كيو", "الكيو", "كيوز", "كيوات", "كيوه",
        "enqueue", "dequeue", "rear", "front", "fifo",
        "circular queue", "priority queue", "circular", "دائري",
    ],
    "pointers": [
        "pointer", "مؤشر",
        "pointers", "المؤشر", "مؤشرات", "بوينتر", "البوينتر", "بوينترز",
        "shallow copy", "deep copy", "dereference", "memory address",
        "dangling", "memory leak", "عنوان الذاكرة",
    ],
    "database_ch1": [
        "database", "قاعدة بيانات",
        "databases", "dbms", "db", "قواعد بيانات", "قواعد البيانات", "قاعده بيانات",
        "داتا بيس", "الداتا بيس", "داتابيس", "الداتابيس", "داتا بيز", "داته بيس",
        "داتا بيسز", "داتا",
        "schema", "instance", "ddl", "dml", "sql", "database architecture",
    ],
    "database_ch2": [
        "primary key", "مفتاح اساسي",
        "primary keys", "key attribute", "المفتاح الاساسي", "مفتاح أساسي",
        "المفتاح الأساسي", "entity", "كيان", "relationship", "علاقه",
        "entity set", "relationship set", "cardinality", "participation",
        "total participation", "partial participation", "weak entity",
        "er model", "مشاركه كليه", "مشاركه جزئيه",
    ],
    "database_ch3": [
        "foreign key", "مفتاح اجنبي",
        "mapping erd", "map erd", "erd to relational", "erd to relation",
        "many-to-many", "many to many", "one-to-many", "one to many",
        "one-to-one", "m:n", "1:n", "composite key", "تحويل erd",
        "المفتاح الاجنبي",
    ],
    "database_ch4": [
        "normalization", "التطبيع",
        "normal form", "1nf", "2nf", "3nf", "functional dependency",
        "partial dependency", "transitive dependency", "anomaly", "anomalies",
        "اعتماديه وظيفيه", "اعتماد جزئي", "اعتماد انتقالي",
    ],
}


def _normalized_alias_tokens(phrase: str) -> list[str]:
    return [strip_definite_article(t) for t in tokenize(phrase)]


# Pre-normalized alias table, built once at import time: domain ->
# list of (normalized_phrase, [normalized_tokens...]).
_NORMALIZED_ALIASES: dict[str, list[tuple[str, list[str]]]] = {
    domain: [(normalize_text(phrase), _normalized_alias_tokens(phrase)) for phrase in phrases]
    for domain, phrases in DOMAIN_ALIASES.items()
}

# Flat vocabulary of every alias token, used for fuzzy matching, mapped back
# to the domain(s) it belongs to.
_TOKEN_TO_DOMAINS: dict[str, set[str]] = {}
for _domain, _entries in _NORMALIZED_ALIASES.items():
    for _phrase, _tokens in _entries:
        for _tok in _tokens:
            if len(_tok) < 3:
                continue  # too short to fuzzy-match safely
            _TOKEN_TO_DOMAINS.setdefault(_tok, set()).add(_domain)
_ALIAS_VOCABULARY = list(_TOKEN_TO_DOMAINS)


def exact_and_alias_domain_matches(question: str) -> set[str]:
    """Tier 1+2: exact substring match, then token-set alias match (both
    with definite-article stripping) against the normalized question.
    """
    normalized_question = normalize_text(question)
    question_tokens = normalized_token_set(question)
    matched: set[str] = set()

    for domain, entries in _NORMALIZED_ALIASES.items():
        for phrase, tokens in entries:
            if not phrase:
                continue
            # Tier 1: exact normalized substring (fast path, highest priority).
            if phrase in normalized_question:
                matched.add(domain)
                break
            # Tier 2: every token of the alias phrase is present in the
            # question's token set (order-independent, article-insensitive).
            if tokens and all(tok in question_tokens for tok in tokens):
                matched.add(domain)
                break

    return matched


def fuzzy_domain_matches(question: str, cutoff: float = 0.84) -> set[str]:
    """Tier 3: bounded single-token typo tolerance via difflib (stdlib only).

    Only ever called when tiers 1+2 found nothing, and only matches tokens
    of length >= 4 against the alias vocabulary with a conservative cutoff
    (roughly: at most one substituted/missing/extra character on a short
    word). This must never be confident enough to turn an unrelated word
    into a domain match, so short/common tokens are excluded and the cutoff
    is deliberately high.
    """
    matched: set[str] = set()
    for token in normalized_token_set(question):
        if len(token) < 4:
            continue
        close = difflib.get_close_matches(token, _ALIAS_VOCABULARY, n=1, cutoff=cutoff)
        if close:
            matched |= _TOKEN_TO_DOMAINS[close[0]]
    return matched


def route_domains_with_confidence(question: str) -> tuple[list[str], str]:
    """Returns (matched_domains, confidence) where confidence is one of:
    "exact_or_alias", "fuzzy", or "fallback_all" (no evidence for any
    specific domain - the caller should use the existing safe multi-domain
    retrieval fallback, never a blind single-domain guess).
    """
    matched = exact_and_alias_domain_matches(question)
    if matched:
        return sorted(matched), "exact_or_alias"

    matched = fuzzy_domain_matches(question)
    if matched:
        return sorted(matched), "fuzzy"

    return [], "fallback_all"


# --------------------------------------------------------------------------- #
# Intent detection and answer-supporting reranking
# --------------------------------------------------------------------------- #
# Reusable across every domain: these describe the *shape* of a question
# (definition / comparison / operation / example / purpose), never one
# specific sentence or one specific subject.

_STOPWORD_TOKENS = {
    "ما", "هو", "هي", "ماذا", "من", "في", "على", "عن", "يا",
    "is", "a", "an", "the", "of", "do", "does", "to", "for",
}

_DEFINITION_INTENT_MARKERS = [
    "ما هو", "ما هي", "what is", "define", "definition", "عرف",
    "اشرح", "اشرحلي", "شرح", "يعني", "تعريف", "معنى", "explain",
]
_COMPARISON_INTENT_MARKERS = [
    "فرق", "الفرق", "مقارنه", "قارن", "difference", "compare", "vs", "versus",
]
_OPERATION_INTENT_MARKERS = [
    "كيف", "how", "steps", "خطوات", "تعمل", "يعمل", "عمليه", "operation", "operations", "work", "works",
]
_EXAMPLE_INTENT_MARKERS = [
    "مثال", "امثله", "example", "examples",
]
_PURPOSE_INTENT_MARKERS = [
    "لماذا", "ليش", "why", "purpose", "importance", "used for", "usage", "استخدام", "اهميه",
]

# Heading-side markers used by the reranker (kept separate from the
# question-side intent markers above, since a heading is matched as a
# substring of normalized title+headings text, not tokenized).
# Split into "strong" (unambiguous - only ever appears in a genuinely
# definitional heading) and "weak" (a bare word like "definition" that can
# also appear inside an unrelated technical term, e.g. "Data Definition
# Language") so a weak match earns a smaller, conservative bonus.
_DEFINITION_MARKERS_STRONG = [
    normalize_text(m) for m in ("ما هو", "ما هي", "what is", "تعريف", "معنى", "الفكرة الأساسية", "الفكره الاساسيه")
]
_DEFINITION_MARKERS_WEAK = [normalize_text(m) for m in ("definition", "define")]
_COMPARISON_HEADING_MARKERS = [normalize_text(m) for m in ("lifo", "fifo", "تعريف", "ما هو", "ما هي", "introduction", "مقدمه")]
_EXAMPLE_MARKERS = [normalize_text(m) for m in ("example", "examples", "مثال", "امثله")]
_PURPOSE_MARKERS = [normalize_text(m) for m in ("لماذا", "why", "purpose", "importance", "used for", "usage", "استخدام")]

_PSQ_HEADING = re.compile(r"^##[^\n]*possible student questions[^\n]*$", re.IGNORECASE | re.MULTILINE)

# A generic "<concept> is/هو a ..." copula pattern, used by
# _contains_definitional_sentence below to find an actual definitional
# sentence anywhere in a chunk's body text, not just in its heading - which
# is what lets a slide's real one-line definition (e.g. "Pointer هو متغير
# محتواه عنوان في الذاكرة", found buried under a generic "3. الفكرة
# الأساسية" heading rather than a heading that says "definition") outrank a
# same-topic but non-definitional slide that only happens to share the same
# repeating section-heading template.
_DEFINITION_COPULA_TOKENS = {"هو", "هي", "is"}


def _meaningful_tokens(question: str) -> set[str]:
    return {t for t in normalized_token_set(question) if t not in _STOPWORD_TOKENS and len(t) >= 2}


def _extract_possible_questions_block(text: str) -> str:
    match = _PSQ_HEADING.search(text)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(r"^##\s", text[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(text)
    return text[start:end]


def _contains_definitional_sentence(text: str, canonical_terms: set[str]) -> bool:
    """True if some line of `text` names a known course concept as the
    subject of an "is a"/"هو"/"هي" copula shortly afterward (within 3
    tokens, e.g. "Pointer هو متغير ..." or "A stack is a data structure") -
    a generic proxy for "this line actually defines something", independent
    of which heading it happens to sit under.

    Requiring the copula to come shortly *after* the concept term (subject-
    first) rather than merely "anywhere on the same line" was a deliberate
    tightening: the looser version also matched incidental, non-definitional
    uses of "is"/"هو" elsewhere in an unrelated sentence that merely
    mentioned the concept in passing (verified against real Pointers course
    content while building this - e.g. "only p is the pointer variable, not
    q" and "Null Pointer ... هو حالة عدم الإشارة إلى object صالح" both
    matched the looser check without actually defining "pointer").
    """
    for line in text.splitlines():
        tokens = tokenize(line)
        if not tokens:
            continue
        term_positions = [i for i, t in enumerate(tokens) if t in canonical_terms]
        if not term_positions:
            continue
        for term_index in term_positions:
            for offset in range(1, 4):
                copula_index = term_index + offset
                if copula_index >= len(tokens):
                    break
                token = tokens[copula_index]
                if token not in _DEFINITION_COPULA_TOKENS:
                    continue
                if token == "is":
                    if copula_index + 1 < len(tokens) and tokens[copula_index + 1] in {"a", "an", "the"}:
                        return True
                else:
                    return True
    return False


def detect_intents(question: str) -> set[str]:
    """Classify a question into zero or more reusable intents. A question
    can carry more than one intent (e.g. "what is the difference between a
    stack and a queue" is both definition and comparison) - callers should
    not assume exactly one.
    """
    normalized = normalize_text(question)
    intents: set[str] = set()
    if any(normalize_text(marker) in normalized for marker in _DEFINITION_INTENT_MARKERS):
        intents.add("definition")
    if any(normalize_text(marker) in normalized for marker in _COMPARISON_INTENT_MARKERS):
        intents.add("comparison")
    if any(normalize_text(marker) in normalized for marker in _OPERATION_INTENT_MARKERS):
        intents.add("operation")
    if any(normalize_text(marker) in normalized for marker in _EXAMPLE_INTENT_MARKERS):
        intents.add("example")
    if any(normalize_text(marker) in normalized for marker in _PURPOSE_INTENT_MARKERS):
        intents.add("purpose")
    return intents


# --------------------------------------------------------------------------- #
# Retrieval-time query enrichment
# --------------------------------------------------------------------------- #

def _possible_questions_overlap_bonus(question: str, text: str) -> bool:
    block = _extract_possible_questions_block(text)
    if not block:
        return False
    question_tokens = _meaningful_tokens(question)
    if not question_tokens:
        return False
    block_tokens = normalized_token_set(block)
    overlap = question_tokens & block_tokens
    return len(overlap) >= max(2, round(0.6 * len(question_tokens)))


def rerank_by_intent(question: str, candidates: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """A conservative, explainable reranking layer applied on top of a
    domain module's own BM25 (+ any domain-specific boosts) score. It never
    replaces BM25 - it only adds a bounded bonus, on the same order of
    magnitude as the domain-specific boosts already used in
    retrieve_stack/retrieve_queue (20-55 points), to candidates that carry
    a generic, reusable "this is the right kind of chunk for this kind of
    question" signal:

    - definition intent -> prefer a heading that is itself a definition/
      "what is"/"core idea" heading, a chunk whose text contains an actual
      "<concept> is/هو a ..." definitional sentence, and a chunk whose own
      "Possible Student Questions" section closely matches the asked
      question.
    - comparison intent -> mild preference for definition-style headings
      (stack/queue already have their own more specific cross-domain
      comparison boosts in signbridge_router.py; this is the generic
      fallback for domains that do not).
    - operation intent -> prefer chunks whose section headings share a
      content word with the question (e.g. a specific operation name).
    - example intent -> prefer chunks whose title/headings are themselves
      example sections.
    - purpose intent -> prefer why/purpose headings and a matching
      "Possible Student Questions" section.

    Deliberately does NOT assume "slide 1 (or any other fixed slide number)
    is always the domain's introduction/definition slide" - that assumption
    was checked against the real course content while building this
    reranker and found to be **false** for at least one existing domain
    (Stack's slide 1 is "Building a Stack Using an Array", an array-
    implementation walkthrough; the real Stack definition/LIFO explanation
    is on slide 4), so baking in a slide-number shortcut here would have
    made that domain's ranking worse, not better. Every signal here is
    judged from the chunk's own title/headings/text instead, which is also
    why this works identically across every domain (stack, queue, pointers,
    all four database chapters) without any per-domain configuration.
    """
    intents = detect_intents(question)
    if not intents or not candidates:
        return candidates

    canonical_terms = {
        normalize_text(term)
        for domain_aliases in DOMAIN_ALIASES.values()
        for term in domain_aliases[:2]
    }

    reranked: list[tuple[float, dict]] = []
    for score, document in candidates:
        bonus = 0.0
        title_and_headings = normalize_text(
            str(document.get("slide_title", "")) + " " + " ".join(document.get("section_headings", []))
        )
        text = str(document.get("text", ""))

        if "definition" in intents:
            if any(marker in title_and_headings for marker in _DEFINITION_MARKERS_STRONG):
                bonus += 25.0
            elif any(marker in title_and_headings for marker in _DEFINITION_MARKERS_WEAK):
                bonus += 10.0
            if _contains_definitional_sentence(text, canonical_terms):
                bonus += 20.0
            if _possible_questions_overlap_bonus(question, text):
                bonus += 30.0

        if "comparison" in intents:
            if any(marker in title_and_headings for marker in _COMPARISON_HEADING_MARKERS):
                bonus += 15.0

        if "operation" in intents:
            heading_tokens = normalized_token_set(" ".join(document.get("section_headings", [])))
            if _meaningful_tokens(question) & heading_tokens:
                bonus += 20.0

        if "example" in intents:
            if any(marker in title_and_headings for marker in _EXAMPLE_MARKERS):
                bonus += 20.0

        if "purpose" in intents:
            if any(marker in title_and_headings for marker in _PURPOSE_MARKERS):
                bonus += 20.0
            if _possible_questions_overlap_bonus(question, text):
                bonus += 15.0

        reranked.append((score + bonus, document))

    reranked.sort(key=lambda item: item[0], reverse=True)
    return reranked


def enrich_query_for_retrieval(question: str, domains: list[str]) -> str:
    """Appends the canonical (first two, per matched domain) alias terms to
    the question text before it reaches a domain module's own
    tokenize/expanded_query/BM25 pipeline - so a transliteration or
    colloquial phrase the domain module's own local EXPANSIONS dict does not
    recognize still reaches the course-content vocabulary BM25 was built
    from. This is query expansion, not a new retrieval architecture: the
    domain module still runs its own existing BM25 exactly as before, just
    against a slightly enriched query string.
    """
    additions: list[str] = []
    seen: set[str] = set()
    for domain in domains:
        for phrase in DOMAIN_ALIASES.get(domain, [])[:2]:
            if phrase not in seen:
                seen.add(phrase)
                additions.append(phrase)
    if not additions:
        return question
    return question + " " + " ".join(additions)
