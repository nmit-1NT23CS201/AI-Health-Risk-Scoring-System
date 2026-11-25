# Preventive Risk Assessment

A research prototype for **preventive screening support**. It estimates, independently, the probability of **cardiovascular disease (CVD)**, **diabetes** and **hypertension** from routinely available information, and shows which inputs contributed most. It is **not** a clinically validated diagnostic product and does not provide medical advice.

- Models trained on **NHANES 2021–2023** adults (US), six configurations: three conditions × **Mode A** (profile, vitals, lifestyle) and **Mode B** (adds laboratory biomarkers).
- Each probability is sigmoid-calibrated and compared with a **locked screening threshold**. There is no combined or 0–100 score.
- Explanations (SHAP) are computed live for the submitted input and shown as relative magnitude, never as probability changes.

The repository contains only this V2 system; the earlier V1 prototype (single synthetic-data risk score) has been retired. Full documentation is in [`docs/`](docs/README.md).

## Repository layout

```text
backend/
├── api/
│   ├── main.py            FastAPI application (run this)
│   └── v2/                Inference engine: registry, preprocessing, SHAP, history, routes
├── database/              SQLite access (healthrisk.db is created locally; git-ignored)
├── ml/                    Frozen research pipeline: data, scripts, models, evaluation reports,
│                          Stage 1G SHAP scripts and figures
├── outputs/               Stage 1D / 1E-1 baseline model outputs
├── tests/                 pytest suite
├── requirements.txt       Pinned runtime dependencies
└── requirements-dev.txt   Test dependencies
frontend/                  Next.js interface
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

Backend (from the repository root):

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --reload
```

Frontend (second terminal):

```powershell
cd frontend
npm run dev
```

Open <http://localhost:3000>. The API listens on <http://127.0.0.1:8000> (interactive docs at `/docs`). The frontend reads `NEXT_PUBLIC_API_BASE_URL` (default `http://127.0.0.1:8000`).

### Configuration (optional)

Copy `.env.example` to `.env` (git-ignored; never commit it):

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | Neon / PostgreSQL connection string for saved history | unset → local SQLite file `backend/database/healthrisk.db` |
| `CORS_ORIGINS` | Comma-separated browser origins allowed to call the API | `http://localhost:3000,http://127.0.0.1:3000` |

History is kept per browser: the frontend sends a random `X-Client-ID` (stored in `localStorage`), and the API only ever returns or deletes the caller's own entries. This separates browsers; it is not authentication. `GET /v2/health` reports which database backend is active (`postgresql` or `sqlite`). Tests always run offline against a temporary SQLite database.

## Test and build

```powershell
# Backend: V2 parity, SHAP, history and API tests
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
- Submitted health inputs (and the optional patient name) are stored in the configured database (local SQLite or your Postgres) when "Save to history" is on. History is separated per browser by an anonymous id, but there is no authentication or encryption layer of our own.

## Documentation

Start at [`docs/README.md`](docs/README.md): architecture, data and features, ML pipeline and evaluation, explainability, API and database, and frontend. All new documentation belongs in `docs/` and must be updated with the code it describes.