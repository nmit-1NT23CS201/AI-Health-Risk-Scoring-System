# Documentation Index

AI-Based Health Risk Scoring System for Preventive Healthcare.

> **Research prototype.** Not a clinically validated diagnostic product.

## Table of Contents

| Doc | Contents |
|---|---|
| [architecture.md](architecture.md) | Overview, why V2 exists, stack, repo layout, architecture, implementation status, phases, next steps |
| [data-and-features.md](data-and-features.md) | NHANES processing, targets, leakage prevention, Mode A/B features, splits |
| [ml-pipeline-and-evaluation.md](ml-pipeline-and-evaluation.md) | Model selection, calibration, thresholds, final test results, Stage 1G audits, artifacts and inference rules |
| [explainability-shap.md](explainability-shap.md) | SHAP methodology, interpretation rules, per-request SHAP plan |
| [api-and-database.md](api-and-database.md) | FastAPI application, /v2 endpoints, encoding contract, SQLite schema, tests, what was retired from V1 |
| [frontend.md](frontend.md) | Next.js interface: structure, API usage, views, design system, invariants |

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
