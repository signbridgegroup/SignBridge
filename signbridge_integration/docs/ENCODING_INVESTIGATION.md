# Arabic text encoding — investigation and conclusion

**Conclusion: every source file this integration reads, and everything it writes, is correct UTF-8. The mojibake seen once (`Ø§`) was a PowerShell terminal/`Get-Content` display artifact only — it never touched any actual file.**

## What was checked, with direct evidence

| Source | Method | Result |
|---|---|---|
| `data/processed/isharah_gloss_clip_library/clips.csv`, row 0 `gloss` field | Python: `open(..., encoding='utf-8-sig')`, then inspected `repr()` and codepoints | `'ا'` → `[0x627]`, a genuine Arabic letter (Alef). Raw file bytes independently decode as valid UTF-8. |
| `data_manifests/motion_manifest.json`, `isharah:GLOSS_0000.label` | Python: `json.loads(path.read_text(encoding='utf-8'))` | Same value, same codepoint — round-trips correctly through the builder's own `json.dumps(..., ensure_ascii=False)` + `.write_text(..., encoding='utf-8')`. |
| `data/external/isharah/annotations/si_1000/train.txt` | Python: raw byte inspection + `.decode('utf-8')` | Valid UTF-8; `سوال هو`, `من هو` etc. decode correctly. |
| `sys.stdout.encoding` in the Python process used throughout this session | `sys.stdout.encoding` | `utf-8` — Python's own output is not the source of the corruption either. |

## Where the mojibake actually came from

Windows PowerShell 5.1's `Get-Content` (and `Write-Host` rendering of PowerShell-native pipeline objects) does not reliably assume UTF-8 for a file without a byte-order mark, and the console's active codepage in this session is not UTF-8. When a UTF-8 multi-byte Arabic character's raw bytes get individually reinterpreted under that codepage, the classic `Ø`/`Ù`/`Ã`/`Â`-prefixed mojibake pattern results. This reproduced identically for two unrelated files (`clips.csv` values printed via `Write-Host`, and `si_1000/train.txt` printed via `Get-Content`) — both files are independently confirmed correct in Python, so the common factor is the PowerShell display path, not the data.

**Practical consequence**: don't trust PowerShell terminal output to judge Arabic text correctness in this project — always verify through Python (as above) or by opening the file in an editor that assumes UTF-8.

## Permanent guard added

`tools/motion_library_builder/label_quality.py` classifies every label the builder produces as `empty`, `numeric_only`, `arabic_readable`, `mojibake_suspected` (flags `Ø`, `Ù`, `Ã`, `Â`, `â€`, `Ã¢`), or `other_non_arabic`. `builder.py`'s `inventory`/`rebuild-manifest` commands **refuse to save the manifest** if any label is flagged `mojibake_suspected`, and `validate` reports the same check on demand.

## Label quality, by dataset (current manifest)

| Dataset | other_non_arabic | arabic_readable | numeric_only | empty | mojibake_suspected |
|---|---|---|---|---|---|
| jordanian_it (165) | 165 | 0 | 0 | 0 | 0 |
| karsl (502) | 502 | 0 | 0 | 0 | 0 |
| isharah (680) | 42 | 638 | 0 | 0 | 0 |

Notes:
- **jordanian_it** labels are English words/symbols from the dataset's own folder names (`Stack`, `Array`, `&`, `C++`, …) — never Arabic, by design; `other_non_arabic` here is expected and correct, not a defect.
- **karsl**: `karsl_sign_mapping.csv` marks many sign IDs `status=new_token` with placeholder `label_ar`/`label_en` values of `"0"`, `"1"`, … These are **not real words** — using them as a label would misrepresent a numeric placeholder as meaningful text. The inventory builder now falls back to an honest `"KArSL <sign_id>"` label (recorded as `quality_metadata.label_source = "sign_id_fallback_no_human_label_assigned"`) instead, and no invented Arabic meaning is attached to any of them.
- **isharah**: 638 of 680 glosses are genuine, readable Arabic; the remaining 42 contain Latin letters, digits, or punctuation mixed into the gloss (e.g. domain terms like `SQL`, `C++`) and are correctly classified as non-Arabic rather than flagged as an error.
