# Frontend

> Everything below describes the **V1** frontend. Status: Implemented, not runtime-verified. No V2 frontend code exists.

## 1. Architecture

- Next.js `^16.2.7`, App Router, React 18.3.1, TypeScript 5.5.4. No `pages/` folder.
- Only route: `/` (`frontend/app/page.tsx`), a single `"use client"` component `Home`. No components directory, no auth, no UI/chart/form/HTTP libraries.
- `app/layout.tsx`: Manrope and Space Grotesk fonts; title "AI Health Risk Dashboard".
- `app/globals.css`: plain CSS with custom properties, dark theme.
- API base: `process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000"` (line 5); raw `fetch`.
- Build state: `frontend/node_modules/` and `frontend/.next/` are absent; run `npm install` first.

## 2. Flows

| Flow | Endpoint | Implementation |
|---|---|---|
| Assessment form | — | Sliders/number inputs with HTML min/max only (lines 277-534); presets default/high/low |
| Predict | `POST /predict` | `risk_score.toFixed(2)` as "Score (0-100)", clamped progress bar, risk pill |
| Explanation | in predict response | Contributor chips (names only), insights, three SHAP PNGs |
| Model metrics | `GET /model-metrics` | V1 regression + classification metrics, confusion matrix |
| History | `GET/DELETE /history` | Table with per-row delete and clear-all |

Form ranges: age 18–80, BMI 15–45, systolic 90–200, diastolic 60–130, cholesterol 120–320, HR 50–130, SDNN/RMSSD 10–140, SpO₂ 90–100. There is no validation beyond HTML attributes.

## 3. V1 assumptions to remove in V2

- Continuous 0–100 score and Low/Medium/High Risk strings (`riskPillClass` matches exact strings).
- Single "cardiovascular" model (hero text "Predict cardiovascular risk").
- HRV / SpO₂ inputs.
- Regression metrics panel (MAE/MSE/RMSE/R²).
- One score per history row.

## 4. V2 UI requirements (planned, not started; backend ready)

The V2 backend is implemented (see [api-and-database.md](api-and-database.md)): build forms from `GET /v2/features/{mode}` instead of hard-coding fields, submit to `POST /v2/predict` (only `age`, `gender`, `bmi` are required), render `explanation` per [explainability-shap.md](explainability-shap.md), and use `/v2/history*` for history. V2 routes share the V1 CORS configuration (localhost:3000 only).

- Mode A and Mode B input flows with V2 features (see [data-and-features.md](data-and-features.md)).
- Three disease-specific results: calibrated probability, locked-threshold decision, explanation.
- SHAP in model-output units, never as "+X%" or probability points; never show NHANES precomputed SHAP as the user's explanation.
- Prediction history with a V2 schema (see [api-and-database.md](api-and-database.md)).
- Clear research-prototype / non-diagnostic disclaimer.

`PROJECT_SYSTEM_DESCRIPTION.md` §18 is a **mockup**, not the current UI (it describes a Mode B toggle, percentage probabilities, different colours and `backdrop-filter`, none of which exist in code).
