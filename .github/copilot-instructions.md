# Instructions for AI Coding Agents

## Documentation rules

- Place **all new documentation files inside `docs/`**. Do not create documentation elsewhere in the repository.
- Whenever code, APIs, schemas, data or UI change, **update the relevant feature doc(s) in the same change**. Start from `docs/README.md`:
  - `docs/architecture.md` — overview, status, phases
  - `docs/data-and-features.md` — NHANES data, targets, features, splits
  - `docs/ml-pipeline-and-evaluation.md` — models, calibration, thresholds, results, audits
  - `docs/explainability-shap.md` — SHAP
  - `docs/api-and-database.md` — endpoints, schemas, SQLite
  - `docs/frontend.md` — Next.js UI
- Do not describe a feature as implemented unless the code supports it; use the status labels defined in `docs/README.md`.
- Do not invent metrics, feature names or endpoints; keep numbers in sync with their source-of-truth files.

## Frozen V2 ML pipeline

- **Never modify the frozen V2 ML research pipeline without explicit instruction**: `backend/ml/scripts/`, `backend/ml/data/`, `backend/ml/evaluation/`, split files, calibrated models and thresholds, and the root SHAP scripts and artifacts.
- Keep V1 runtime assets (`backend/models/random_forest_model.pkl`, `backend/outputs/metrics.json`, `backend/outputs/confusion_matrix.png`, `dataset/indian_health_risk_dataset.csv`) until V2 is validated.
- `docs/legacy_v1/` is archived; do not edit it.
