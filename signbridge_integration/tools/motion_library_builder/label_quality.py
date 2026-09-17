"""Label quality classification and mojibake detection.

Confirmed by direct inspection (see docs/ENCODING_INVESTIGATION.md): every
source CSV/JSON this project reads is valid UTF-8 and every label this
builder produces round-trips correctly through Python and JSON
(ensure_ascii=False). The mojibake seen in one PowerShell debugging session
was purely a terminal-display artifact, not a data problem. This module
exists so that claim is continuously checked, not just asserted once.
"""

from __future__ import annotations

# Byte-sequence artifacts that appear when UTF-8 bytes are mis-decoded as
# Windows-1252/Latin-1 (the classic "Ø§" pattern). Legitimate Arabic,
# English, or CS-education text essentially never contains these.
MOJIBAKE_MARKERS = ("Ø", "Ù", "Ã", "Â", "â€", "Ã¢")

ARABIC_RANGE = range(0x0600, 0x0700)


def classify_label(label: str | None) -> str:
    """Return one of: empty, numeric_only, arabic_readable, mojibake_suspected, other_non_arabic."""
    text = (label or "").strip()
    if not text:
        return "empty"
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        return "mojibake_suspected"
    if text.isdigit():
        return "numeric_only"
    has_arabic = any(ord(c) in ARABIC_RANGE for c in text)
    non_arabic_non_space = [c for c in text if not c.isspace() and ord(c) not in ARABIC_RANGE]
    if has_arabic and not non_arabic_non_space:
        return "arabic_readable"
    return "other_non_arabic"


def label_report(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        category = classify_label(label)
        counts[category] = counts.get(category, 0) + 1
    return counts


def assert_no_mojibake(labels: dict[str, str]) -> list[str]:
    """Return the keys (tokens) whose label is flagged as mojibake, for the
    caller to raise/report on. Never silently drops or "fixes" a label."""
    return [token for token, label in labels.items() if classify_label(label) == "mojibake_suspected"]
