# License and data-usage notes

## Code

No license file existed in the original `C:\SignBridge_Project` repository at the time of this integration. Choose and add a license (e.g. MIT, Apache-2.0) before making the GitHub repository public; until then, treat the code as all-rights-reserved by the project owner.

## Final recognition checkpoint

`data/processed/unified_karsl_multitask_training/best_unified_karsl_multitask.pt` (~18.5 MB) is well under GitHub's 100 MB hard limit. **Before including it in the repository**, confirm redistribution is acceptable given it was trained partly on KArSL and Isharah data whose own licenses may restrict redistribution of derived model weights — check those datasets' license terms (see `docs/DATASETS.md`) before publishing, not just the file size.

## Custom Jordanian IT dataset

Raw videos are **withheld** — do not commit them. Consent/redistribution rights from the signers have not been confirmed as of this integration. `data/processed/final_dataset_audit.csv` and `data/processed/active_length_report.csv` (labels, video counts, detection rates — no video content) are safe to include as manifests.

## KArSL

Not redistributed. Obtain from the original publishers (Sidig, Luqman, Mahmoud, Mohandes — ACM TALLIP 2021) under their license. Only derived metadata (`samples.csv`, `karsl_sign_mapping.csv` — sign IDs, detection rates, no video) is included here.

## Isharah

Not redistributed. Obtain from the dataset's original publishers. Only derived metadata (`clips.csv`, `samples.csv`) is included here.

## RAG course content

The Stack/Queue/Pointers/Database course material (`backend/legacy_runtime/rag/data/`) was authored/compiled by the project team for this course. Confirm with the team whether it may be published in a public repository before doing so; if not, keep the integration repository private or exclude that folder and document how to regenerate it from the original course PDFs (`src/extract_pdf.py`, `src/build_chunks.py`).

## Secrets

No `.env`, `key.env`, or API key of any kind is present in this repository. `.gitignore` excludes `.env` and `key.env` explicitly.
