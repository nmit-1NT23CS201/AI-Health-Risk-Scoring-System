# Preventive Risk Assessment (V2)

A research prototype for **preventive screening support**. It estimates, independently, the probability of **cardiovascular disease (CVD)**, **diabetes** and **hypertension** from routinely available information, and shows which inputs contributed most. It is **not** a clinically validated diagnostic product and does not provide medical advice.

- Models trained on **NHANES 2021–2023** adults (US), six configurations: three conditions × **Mode A** (profile, vitals, lifestyle) and **Mode B** (adds laboratory biomarkers).
- Each probability is sigmoid-calibrated and compared with a **locked screening threshold**. There is no combined or 0–100 score.
- Explanations (SHAP) are computed live for the submitted input and shown as relative magnitude, never as probability changes.

Full documentation is in [`docs/`](docs/README.md).

## Repository layout

```text
backend/
├── api/
│   ├── main.py            V1 + V2 composition root (run this)
│   ├── api_server.py      V1 API (legacy, unchanged)
│   ├── app.py             V1 Streamlit dashboard (legacy)
│   └── v2/                V2 inference engine (registry, preprocessing, SHAP, history, routes)
├── database/              SQLite access (healthrisk.db is created locally; git-ignored)
├── ml/                    Frozen V2 research pipeline: data, scripts, models, evaluation reports
├── models/, outputs/      Legacy V1 artifacts (still used by the V1 API)
├── tests/                 pytest suite
├── requirements.txt       Pinned runtime dependencies
└── requirements-dev.txt   Test dependencies
frontend/                  Next.js V2 interface
dataset/                   Legacy V1 dataset (kept for the V1 API)
docs/                      Maintained documentation
```

## Setup

Requires Python 3.13 and Node.js. From the repository root (Windows PowerShell):

```powershell
# Python environment
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

`backend/requirements.txt` pins the versions the models were validated with (notably `scikit-learn==1.9.0`); the saved model files are version-sensitive, so do not upgrade casually.

## Run

Backend (V1 + V2 routes, from the repository root):

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --reload
```

Frontend (second terminal):

```powershell
cd frontend
npm run dev
```

Open <http://localhost:3000>. The API listens on <http://127.0.0.1:8000> (interactive docs at `/docs`). The frontend reads `NEXT_PUBLIC_API_BASE_URL` (default `http://127.0.0.1:8000`); the backend allows CORS only from `localhost:3000` / `127.0.0.1:3000`.

The legacy V1 API alone can still be started with `.\.venv\Scripts\python.exe -m uvicorn backend.api.api_server:app`, but the current frontend uses only the V2 API.

## Test and build

```powershell
# Backend: 126 tests (V2 parity, SHAP, history, V1 non-regression)
.\.venv\Scripts\python.exe -m pytest backend/tests -v

# Frontend
cd frontend
npx tsc --noEmit
npm run build
```

The tests never touch `backend/database/healthrisk.db` (they use a temporary database).

## What the application does

| View | Description |
|---|---|
| Assessment | Choose Mode A or B, enter age, sex and BMI (required) plus anything else you know, and get three independent results: estimated probability, locked screening threshold, decision, key contributing factors, and model notes |
| History | Optional saved assessments, newest first, with details, single delete and clear-all |
| Model Performance | Offline held-out test metrics for all six models with Mode A vs Mode B comparison and the study's limitations |

## Important limitations

- US NHANES 2021–2023 data; no external or clinical validation. Reported metrics are unweighted sample estimates from the held-out test split, which also informed model-family selection.
- The CVD target is a **self-reported prior diagnosis**, not incident risk.
- Mode B improves ranking (ROC-AUC / PR-AUC) but Mode A keeps higher sensitivity at the locked thresholds; neither mode is universally better.
- Some training-data encodings are imperfect (for example the physical-activity category ignores the survey's unit fields); details in [`docs/api-and-database.md`](docs/api-and-database.md).
- Submitted health inputs are stored in a local, unencrypted SQLite file when "Save to history" is on; there is no authentication.

## Documentation

Start at [`docs/README.md`](docs/README.md): architecture, data and features, ML pipeline and evaluation, explainability, API and database, and frontend. All new documentation belongs in `docs/` and must be updated with the code it describes.
