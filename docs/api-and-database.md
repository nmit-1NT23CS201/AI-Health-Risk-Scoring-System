# Backend API and Database

> The backend is a single V2 FastAPI application (`backend/api/main.py`, engine in `backend/api/v2/`). It is **implemented and verified by the automated suite** (`backend/tests/`, 123 tests passing) and is consumed by the Next.js frontend. The earlier V1 prototype has been retired (see §4).

## 1. Running

| Entry point | Command | Serves |
|---|---|---|
| Application | `.\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --reload` | `GET /`, `GET /health`, `/v2/*` |

Environment: project `.venv` (Python 3.13). Pinned in `backend/requirements.txt` (+ `backend/requirements-dev.txt` for `pytest`, `httpx`): scikit-learn 1.9.0 (matches the version recorded in `stage_1f_results.json`), xgboost 3.4.1, shap 0.54.1, fastapi 0.165.0, pydantic 2.14.4, pandas 3.1.1, numpy 2.5.5, uvicorn 0.54.0, joblib 1.6.0, pyarrow 26.0.1, tzdata 2026.6, matplotlib 3.11.2 (research scripts only). `psycopg[binary]==3.3.9` and `python-dotenv==1.2.4` support Postgres and `.env` loading. Run tests from the repo root: `.venv\Scripts\python -m pytest backend/tests -v`.

`backend/api/main.py` creates the FastAPI app (`title="Preventive Health Risk Assessment API"`, `version="2.0"`) with the `v2_lifespan` startup, CORS (origins from `CORS_ORIGINS`, default `http://localhost:3000,http://127.0.0.1:3000`; headers `Content-Type`, `Accept`, `X-Client-ID`), the `V2Error` handler and the `/v2` router, plus lightweight `GET /` and `GET /health` returning `{"status": "ok", "version": "2.0"}`. **A failed model-registry load never takes the process down**: errors are logged as structured JSON, `/v2/health`, `/v2/predict`, `/v2/model-metrics` and `/v2/features/*` return 503, and `/`, `/health` and the history endpoints (database only) keep working.

## 2. V2 package layout (`backend/api/v2/`)

| Module | Responsibility |
|---|---|
| `config.py` | Paths, constants, disease descriptions, metrics caveats. `V2_RESULTS_JSON` env var overrides the results file (used by degraded-startup tests). No features/thresholds. |
| `registry.py` | Loads `stage_1f_results.json` and the six `*_calibrated.joblib`; startup assertions; SHA-256 logging; builds explainers |
| `preprocessing.py` | Encoding contract mirroring `nhanes_harmonizer.py`; model-row construction |
| `schemas.py` | Pydantic v2 request/response models (`extra="forbid"`) |
| `explain.py` | Live per-request SHAP (see [explainability-shap.md](explainability-shap.md)) |
| `service.py` | encode → `predict_proba` → locked threshold → optional SHAP → optional persistence |
| `history.py` | Maps results to/from `v2_assessments` rows (SQL lives in `database.py`) |
| `router.py` | `/v2` routes; no business logic |
| `errors.py` | Typed errors → HTTP (no `str(exc)` leaks) |
| `tools/build_hypertension_a_background.py` | One-off builder for the SHAP background asset |
| `assets/hypertension_mode_a_shap_background.parquet` | 200 training-split rows (seed 42) for the Hypertension A `LinearExplainer` |

### 2.1 Registry startup (`registry.load_registry`)

Source of truth: `backend/ml/evaluation/stage_1f/stage_1f_results.json` (`features`, order, `selected_threshold`, offline metrics). Nothing model-specific is hard-coded. Hard assertions (any failure → `RegistryLoadError`, V2 unavailable):

- document is Stage `1F` with exactly the six configs (`{cvd,diabetes,hypertension}_mode_{a,b}`);
- per config: `n_features == len(features)`, unique names, no survey/`target_*` columns, every feature has an encoder and a schema field, threshold in (0,1), calibration `sigmoid`;
- artifact path exists (JSON paths are `backend/`-relative with Windows separators and are normalized), deserializes, is a `CalibratedClassifierCV` with one sigmoid calibrator and `classes_ == [0, 1]`, matches `n_features_in_`/`feature_names_in_`, and passes an all-NaN `predict_proba` smoke test.

Soft failures (predictions still served, `/v2/health` returns 200 with `"degraded"`): the per-config explainer cannot be built or fails its additivity self-check (≤ 1e-5). A runtime scikit-learn version differing from the JSON is logged and reported.

## 3. V2 endpoints

| Method | Path | Purpose | Status codes |
|---|---|---|---|
| POST | `/v2/predict` | Disease-specific probabilities, locked-threshold decisions, optional live SHAP; optionally persists | 200; 422 validation; 503 registry unavailable; 500 generic `{"error":"internal_error","request_id"}` |
| GET | `/v2/health` | Registry state, per-config load/explainer status, SHA-256s, runtime vs expected versions | 200 (`ok` or `degraded`); 503 (`unavailable`). Also reports `database_backend`: `"postgresql"` or `"sqlite"` (never the connection string) |
| GET | `/v2/model-metrics` | **Offline research evaluation only**, from `stage_1f_results.json`, with caveats; no per-user data | 200; 503 |
| GET | `/v2/features/{mode}` | Field metadata (labels, units, enums, hard bounds, training range, required flag, which diseases use each field) | 200; 422; 503 |
| GET | `/v2/history?limit&before_id` | Newest-first submissions grouped by `submission_id` (`limit` 1-100, default 20) with `next_before_id` cursor | 200; 422; 503 (DB error) |
| GET | `/v2/history/{submission_id}` | One submission with stored inputs, encoded vector, imputed features | 200; 404 |
| DELETE | `/v2/history/{submission_id}` | Delete one submission | 200; 404 |
| DELETE | `/v2/history?confirm=true` | Delete all V2 history | 200; 422 unless `confirm=true` |

History routes need only the database, so they keep working while the model registry is unavailable.

### 3.1 `POST /v2/predict`

Request: `{ "mode": "A" | "B", "diseases": [...] (default all three, unique, non-empty), "inputs": {...}, "include_explanation": true, "save_to_history": true, "patient_name": null }`. `patient_name` is an optional label (max 120 characters; surrounding whitespace is trimmed and a blank value becomes `null`); it is stored with the history entry only and is never used by the models. `inputs` is `ModeAInputs` (13 fields) or `ModeBInputs` (+16 labs), discriminated by `mode`; unknown fields and lab fields under Mode A are 422.

- **Required:** only `age` (20-100; >80 is top-coded to 80 and reported in `input_adjustments`), `gender` (`male`/`female`), `bmi`. Everything else is optional and sent to the model as `NaN` when absent (in-distribution: BP, waist and labs are 24-61% missing in the training data). Hard plausibility bounds reject impossible values; values outside the NHANES training range produce per-disease `warnings`.
- **Per disease** the response returns `probability` (calibrated `predict_proba[:, 1]`, never `predict()`, never a second sigmoid), `threshold` (locked), `predicted_positive` (`probability >= threshold`), `decision_label` (`screen_positive`/`screen_negative`), `imputed_features`, `ignored_inputs`, `warnings`, `model_version` (`stage1f/<config_key>`), `artifact_sha256`, and `explanation` (or `explanation_unavailable_reason`).
- **`ignored_inputs`** makes leakage exclusions explicit: Hypertension A/B ignore `mean_sbp`/`mean_dbp`; Diabetes B ignores `hba1c`/`fasting_glucose`.
- **No aggregate or 0-100 score exists.** The response is `submission_id`, `created_at` (UTC), `mode`, `results[]`, `history_saved`, `input_adjustments`, `warnings[]`, `disclaimer`.
- **Persistence policy:** with `save_to_history=true` all rows of the submission are inserted in **one transaction after every prediction succeeded**. If persistence fails the endpoint still returns **200** with `history_saved=false` and an entry in the top-level `warnings` (no internal detail). `save_to_history=false` stores nothing.

### 3.2 Encoding contract (`preprocessing.py`, mirrors `nhanes_harmonizer.py`)

| Field | Encoding |
|---|---|
| `gender` | male → 1.0, female → 2.0 |
| `smoking_status` | never 0, former 1, current 2 |
| `alcohol_frequency` | never 0, former 1, current 2 (a status, despite the name) |
| `physical_activity_level` | **high 1, moderate 2, low 3** |
| `age` | > 80 → 80 (adjustment noted) |
| other fields | pass-through in NHANES conventional units; absent → `NaN` (imputation happens only inside the artifacts: median imputers for RF/LR, native NaN for HGB/XGB) |

Known encoding limitations (properties of the frozen training data; fixing them requires retraining):

1. **`alcohol_frequency` is a 3-level status, not a frequency.** Raw `ALQ121` has categorical codes 0-10 plus 77/99, but the harmonizer treats only 777/999 as missing, so 5 participants coded 77/99 became "current drinker".
2. **`physical_activity_level` ignores NHANES unit columns.** `PAD790Q`/`PAD810Q` are treated as days/week although their companion unit columns (`PAD790U`/`PAD810U`) are never read; the label is "reported quantity ≥ 3 vigorous or ≥ 5 moderate regardless of unit". The API accepts the already-binned level and the UI must describe the same definition.
3. **BP/pulse are means of up to three oscillometric (BPXO) readings**; single home readings approximate this.
4. **Lab units are NHANES conventional units** (e.g. mg/dL); no conversion code exists.
5. **`bmi` is required by policy but is missing in ~23.5% of NHANES rows**, so such rows cannot be sent through the API (they are covered by encoded-level parity tests).
6. CVD means "self-reported prior diagnosis", not incident risk; training cohort is US NHANES 2021-2023.

## 4. Retired V1 (removed in Phase 5)

The original prototype (a single RandomForestRegressor 0–100 score on a small synthetic dataset, with SHAP plots, rule-based insights, a Streamlit dashboard and an `assessments` table) has been **removed**: `backend/api/api_server.py`, `backend/api/app.py`, `dataset/`, `backend/models/`, `backend/outputs/metrics.json` and `confusion_matrix.png`, the V1 helper modules in `backend/ml/` (`feature_engineering`, `risk_interpreter`, `shap_explainer`, `data_loader`, `train_model`, `preprocessing`, `predict`, `model_evaluation`, `visualization`, `utils`), the V1 database functions and the V1 non-regression tests. The code remains available in git history. `streamlit` was dropped from `backend/requirements.txt`; `matplotlib` is kept because retained research scripts (`backend/ml/scripts/benchmark_*.py`, `backend/ml/evaluation/stage_1g/shap_plot_figure3.py`) import it (the API does not).

Leftovers to be aware of:

- `backend/ml/scripts/main.py` is a V1 end-to-end runner that imports the deleted helpers and reads the deleted dataset and model. It sits in the protected research-scripts directory and was deliberately left untouched; it no longer runs and can be deleted with explicit approval.
- An existing local `backend/database/healthrisk.db` may still contain the old V1 `assessments` table and rows. The code no longer creates or reads it and nothing was dropped; delete the file (it is git-ignored) if the old data is not wanted.
- Historical research reports under `backend/ml/data/` and `backend/ml/evaluation/` still mention the V1 dataset; they are frozen records.
## 5. Database (`backend/database/database.py`)

### 5.0 Backends and configuration

`backend/database/database.py` picks a backend per call from `DATABASE_URL`:

| `DATABASE_URL` | Backend | Notes |
|---|---|---|
| starts with `postgres://` or `postgresql://` | **PostgreSQL** (e.g. Neon) via `psycopg` | `%s` placeholders, `BIGSERIAL` key, `DOUBLE PRECISION` for probabilities, `dict_row` results |
| unset or empty | **SQLite** `backend/database/healthrisk.db` | `?` placeholders, `INTEGER PRIMARY KEY AUTOINCREMENT`; automatic fallback |

Configuration lives in a git-ignored `.env` at the repository root, loaded at startup by `backend/api/main.py` (`python-dotenv`; real environment variables take precedence). `.env.example` documents it: `DATABASE_URL` (optional) and `CORS_ORIGINS` (optional, comma-separated, default `http://localhost:3000,http://127.0.0.1:3000`). `.gitignore` ignores `.env` and `.env.*` except `.env.example`. The connection string is never logged, returned or committed; error logs record only the exception type. No ORM, no migration tool. The local `.db` file is git-ignored too (`*.db`) because it holds user-submitted health data.

Tests always run offline against a temporary SQLite file: `conftest.py` sets `DATABASE_URL` to an empty value before importing the app (so `.env` cannot inject the Neon URL), removes it for every test, redirects `DB_PATH` to a temp file and asserts the local file is never modified.

### 5.0.1 Per-user history (`X-Client-ID`)

History is scoped per browser. Every request may carry an `X-Client-ID` header (the frontend sends a UUID kept in `localStorage`). The API trims it; missing or blank falls back to `"anonymous"` (so `curl` and Swagger keep working); more than 128 characters is a 422. All history reads and deletes are filtered with `WHERE client_id = ...` in the database layer, so one client cannot list, fetch or delete another client's submissions (they get `404` / `0` rows), and `DELETE /v2/history?confirm=true` clears only the caller's rows. The id is a separation key, **not authentication**: anyone who knows another client's id could use it, and `"anonymous"` is shared by all header-less callers. It is never included in responses.

### 5.1 Table `v2_assessments`

Created by `create_v2_tables()` (`CREATE TABLE IF NOT EXISTS`, idempotent) from the V2 **lifespan** in `backend/api/main.py`, never at import time. One row per disease result; all rows of a submission share `submission_id`.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | history is ordered by `id DESC` |
| `submission_id` | TEXT NOT NULL | UUID4 grouping the diseases of one request; indexed |
| `created_at` | TEXT NOT NULL | UTC ISO-8601 with `Z`, millisecond precision; indexed |
| `disease` | TEXT NOT NULL | CHECK in `cvd`, `diabetes`, `hypertension` |
| `mode` | TEXT NOT NULL | CHECK in `A`, `B` |
| `config_key` | TEXT NOT NULL | e.g. `diabetes_mode_b` |
| `inputs_json` | TEXT NOT NULL | validated request inputs as submitted (human enums, absent fields omitted) |
| `model_input_json` | TEXT NOT NULL | exact encoded feature vector in model order; `NaN` stored as `null` |
| `probability` | REAL NOT NULL | calibrated `predict_proba[:, 1]` |
| `threshold` | REAL NOT NULL | locked threshold used |
| `predicted_positive` | INTEGER NOT NULL | CHECK 0/1 |
| `imputed_json` | TEXT | JSON list of features sent as missing |
| `model_version` | TEXT NOT NULL | `stage1f/<config_key>` |
| `artifact_sha256` | TEXT NOT NULL | SHA-256 of the model file used |
| `schema_version` | INTEGER NOT NULL DEFAULT 1 | for future migrations |
| `client_id` | TEXT NOT NULL DEFAULT `'anonymous'` | per-browser owner (see §5.0.1); indexed with `id DESC` (`idx_v2_assessments_client_created`) |
| `patient_name` | TEXT (nullable) | optional label from the request; returned by `GET /v2/history` and `GET /v2/history/{id}` as `patient_name` (`null` when omitted) |

**Automatic upgrade:** `create_v2_tables()` inspects the existing columns (`PRAGMA table_info` on SQLite, `information_schema.columns` on PostgreSQL) and runs `ALTER TABLE ... ADD COLUMN` for `patient_name` and `client_id TEXT NOT NULL DEFAULT 'anonymous'` when missing, so existing databases are upgraded in place on the next startup without losing rows (idempotent; covered by a test). Pre-existing rows belong to `"anonymous"`.

Functions: `create_v2_tables`, `save_v2_submission` (single transaction; all rows or none), `list_v2_history(limit, before_id)` (groups by submission, positioned by each submission's highest row id), `get_v2_submission`, `delete_v2_submission`, `delete_all_v2_history`. SHAP explanations are **not** stored (recomputable). There are no users or foreign keys. The old V1 `assessments` table is no longer created or used (see §4).

## 6. Tests (`backend/tests/`)

| File | Covers |
|---|---|
| `conftest.py` | Offline SQLite only (`DATABASE_URL` neutralised), temp-DB redirect before app import, local-DB integrity guard, shared fixtures |
| `test_v2_schemas.py`, `test_v2_preprocessing.py` | Validation, encoding table, leakage exclusions, training-range table equals processed data |
| `test_v2_registry.py` | Six configs/thresholds, tampered JSON, missing/corrupt artifacts, app stays up with V2 503 |
| `test_v2_predict_parity.py` | Golden parity vs independent offline load (max diff 7.8e-16), full-test-split aggregate metrics vs `stage_1f_results.json` within 5e-5, threshold boundary, no double sigmoid |
| `test_v2_shap.py` | Additivity, payload semantics, HTN-A background hygiene and premise check, concurrency, no pyplot, latency |
| `test_v2_api.py`, `test_v2_history.py` | Endpoints, persistence, atomicity, failure policy, history CRUD, `v2_assessments` created by the lifespan (not at import), per-client isolation (list / fetch / delete / delete-all / pagination), backend selection, CORS `X-Client-ID` and `CORS_ORIGINS` |
| `test_v2_api.py` (also) | `GET /`, `GET /health`, OpenAPI contains only the V2 route set, CORS allows only the frontend origins |
