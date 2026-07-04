---
name: update-html-index
description: Regenerate this repo's root index.html as a table-of-contents page linking every HTML file in the repo, grouped by folder. Use whenever HTML files are added, removed, or moved anywhere in this repo (e.g. after copying in new notebook exports, deleting a section, or reorganizing folders), or when asked to "인덱스 갱신", "목차 다시 만들어", "index 업데이트", "regenerate the index", "update the table of contents". Do NOT use for editing the visual style of index.html itself — that's a manual design change, not a regeneration.
---

# Update HTML Index

This repo publishes a folder of standalone HTML exports (from Jupyter notebooks) via GitHub
Pages. `index.html` at the repo root is the table of contents that makes those files
discoverable — without it, visitors hit a 404 at the root and have to know exact file paths.

Because files get added, removed, or moved over time, `index.html` drifts out of date if
maintained by hand. This skill regenerates it from what's actually on disk, so it's always an
accurate reflection of the current file set.

## How to run it

```bash
python3 .claude/skills/update-html-index/scripts/generate_index.py
```

Run this from anywhere inside the repo — the script walks up from the current directory to
find the repo root (the directory containing `.git`), then scans every subfolder for `*.html`
files (skipping `.git`, `.claude`, `.github`, `node_modules`, and `index.html` itself).

The script:
1. Groups files by their containing folder, in natural sort order (`02-Structures` before
   `10-Foo`, not alphabetical).
2. Derives a readable title per file by stripping a leading numeric prefix (`07-`, `02-05-`)
   and swapping `-`/`_` for spaces — e.g. `06-LangGraph-Agentic-RAG.html` becomes
   "LangGraph Agentic RAG".
3. Derives a folder heading the same way, joined as a breadcrumb — e.g.
   `17-LangGraph/03-Use-Cases` becomes "LangGraph / Use Cases".
4. Overwrites `index.html` with a light/dark-aware page using card-style links, matching the
   look established in this repo's first index page.

## After running

Diff the result (`git diff index.html`) before committing — skim that the new/removed files
you expected are the ones that actually changed. Then stage and commit `index.html` along with
whatever HTML changes triggered the regeneration, in the same commit, so the index and the
content it links to never land in separate commits.

## Adjusting title derivation

The humanization rule (strip numeric prefix, swap separators for spaces) is intentionally
generic so it keeps working as new folders get added. If a specific filename produces an
awkward title, it's simpler to rename the source file to read better than to special-case the
script — the script's job is to be a predictable, mechanical fallback, not to curate wording.
