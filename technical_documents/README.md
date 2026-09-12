# Technical Documents

This folder contains formal Persian technical reports generated from the current
implementation, rather than hand-edited copies of dashboard text.

| File | Purpose |
|---|---|
| `model-design-technical-report-fa.docx` | Ten-page model-design report: Stage 1, Stage 2, thresholds, evaluation, explainability, packaging, and design rationale. |
| `module-architecture-technical-report-fa.docx` | Ten-page module-architecture report: data discovery, extraction, mapping cache, profiles, preprocessing, models, artifact store, dashboard/API, deployment, and tests. |
| `build_technical_documents.py` | Rebuilds both Word documents and their local diagram images from the current documented architecture. |
| `assets/` | Generated architecture diagrams embedded into the Word reports. |

## Rebuild

The regular project virtual environment does not need `python-docx`. Use the
bundled document runtime available in Codex on Windows:

```powershell
& 'C:\Users\hariri_h\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' technical_documents/build_technical_documents.py
```

The script creates the `assets/` directory when needed and overwrites only the
two generated `.docx` files in this folder.
