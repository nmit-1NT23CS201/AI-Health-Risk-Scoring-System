# Documentation Index

AI-Based Health Risk Scoring System for Preventive Healthcare.

> **Research prototype.** Not a clinically validated diagnostic product.

## Table of Contents

| Doc | Contents |
|---|---|
| [architecture.md](architecture.md) | Overview, V1 → V2 evolution, stack, repo layout, current vs proposed architecture, implementation status, phases, next steps |
| [data-and-features.md](data-and-features.md) | NHANES processing, targets, leakage prevention, Mode A/B features, splits |
| [ml-pipeline-and-evaluation.md](ml-pipeline-and-evaluation.md) | Model selection, calibration, thresholds, final test results, Stage 1G audits, artifacts and inference rules |
| [explainability-shap.md](explainability-shap.md) | SHAP methodology, interpretation rules, per-request SHAP plan |
| [api-and-database.md](api-and-database.md) | V1 FastAPI endpoints, schemas, config, SQLite schema, V1 legacy dependencies, V2 integration plan |
| [frontend.md](frontend.md) | Next.js V1 UI, flows, V1 assumptions, V2 UI requirements |

Archived / snapshot material:

- [PROJECT_COMPREHENSIVE_SUMMARY.md](PROJECT_COMPREHENSIVE_SUMMARY.md) — original monolithic snapshot (2026-11-25, commit `769d40c`); not maintained.
- [legacy_v1/README.md](legacy_v1/README.md) — archived V1 documents.
- [../PROJECT_SYSTEM_DESCRIPTION.md](../PROJECT_SYSTEM_DESCRIPTION.md) — original blueprint; may be outdated.

## Status labels used in these docs

| Label | Meaning |
|---|---|
| Implemented & verified | Code/artifact exists and was confirmed by inspection and/or a reproduced audit |
| Implemented, not verified | Code exists; runtime behaviour not executed |
| Partially implemented | Some pieces exist; not end-to-end |
| Planned | Described, not implemented |
| Not started | No implementation or detailed design |

## Documentation Governance

1. **All new documentation goes in `docs/`.** Do not add new documentation files elsewhere in the repository.
2. **Every feature or phase change must update the corresponding doc in `docs/`** in the same change: code, API, schema, UI, data or model changes update the matching file from the table above.
3. Do not describe a feature as implemented unless the code supports it; use the status labels above.
4. Keep verified numbers (metrics, thresholds, counts) in sync with their source-of-truth files, and record discrepancies rather than silently resolving them.
5. The V2 ML research pipeline is frozen. It must not be modified without explicit instruction.
