# Legacy Model V1 Documentation Archive

> **STATUS: DEPRECATED / HISTORICAL REFERENCE ONLY**

The documents stored in this directory (`docs/legacy_v1/`) represent the early **Model V1 proof-of-concept prototype** of the AI Health Risk Scoring System.

---

### Key Historical Characteristics of Model V1 (Superseded):
1. **Dataset**: Used a synthetic 151-row Indian healthcare demonstration dataset (`dataset/indian_health_risk_dataset.csv`).
2. **Model Formulation**: Evaluated a single `RandomForestRegressor` predicting a continuous universal health risk score ($0–100$).
3. **Simulated Wearable Features**: Included synthetic heart rate variability features (`sdnn_hrv`, `rmssd_hrv`, `spo2`).

---

### Canonical Model V2 Replacement:
All active research, clinical modeling, and forthcoming application integration are based strictly on **Model V2**:
- **Dataset**: Official **CDC NHANES August 2021–August 2023** epidemiological cohort ($N = 7,809$ adults aged 20+).
- **Architecture**: Six independent, calibrated binary classification pipelines covering **Cardiovascular Disease**, **Type 2 Diabetes**, and **Hypertension** across two distinct tiers: **Mode A (Non-Invasive)** and **Mode B (Laboratory-Augmented)**.
- **Master Blueprint**: The authoritative system description and architectural specification is documented in [`PROJECT_SYSTEM_DESCRIPTION.md`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/PROJECT_SYSTEM_DESCRIPTION.md).
- **Research Evaluation**: Locked model evaluations and audits are documented under [`backend/ml/evaluation/stage_1f/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1f/) and [`backend/ml/evaluation/stage_1g/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1g/).
