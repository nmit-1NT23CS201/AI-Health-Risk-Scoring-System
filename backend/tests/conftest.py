"""Shared fixtures. The V1 SQLite DB is redirected to a temp file BEFORE any app import so that
``backend/database/healthrisk.db`` (tracked in git) is never touched by the test suite."""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.database.database as _db  # noqa: E402  (must precede any api_server import)

_REAL_DB = Path(_db.DB_PATH)
_TMP_DIR = Path(tempfile.mkdtemp(prefix="hrs_test_db_"))
_db.DB_PATH = _TMP_DIR / "healthrisk_test.db"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from backend.api.v2 import config  # noqa: E402

PROCESSED = config.SPLITS_DIR.parent
SPLIT_FILES = {"cvd": "cvd_splits.csv", "diabetes": "diabetes_splits.csv", "hypertension": "hypertension_splits.csv"}


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def pytest_configure(config):  # noqa: F811  (pytest requires this exact parameter name)
    config.addinivalue_line("markers", "artifacts: needs the frozen Stage 1F artifacts / processed NHANES data")


@pytest.fixture(scope="session", autouse=True)
def _real_db_untouched():
    before = _sha(_REAL_DB)
    assert Path(_db.DB_PATH) != _REAL_DB
    yield
    assert _sha(_REAL_DB) == before, "tests modified backend/database/healthrisk.db"


@pytest.fixture(scope="session", autouse=True)
def _v2_tables_in_temp_db():
    _db.create_v2_tables()  # service-level tests persist history without starting the app lifespan
    yield


@pytest.fixture(scope="session")
def registry():
    from backend.api.v2.registry import load_registry

    if not config.DEFAULT_RESULTS_JSON.exists():
        pytest.skip("stage_1f_results.json missing")
    return load_registry(config.DEFAULT_RESULTS_JSON)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from backend.api.main import app

    with TestClient(app) as c:  # runs lifespan -> loads V2 registry
        yield c


@pytest.fixture
def preserve_v2_state(client):
    """Tests that restart the lifespan with a broken registry share ``app.state``; restore it afterwards."""
    from backend.api.main import app

    saved = (app.state.v2_registry, app.state.v2_startup_errors)
    yield
    app.state.v2_registry, app.state.v2_startup_errors = saved


_cache: dict[str, pd.DataFrame] = {}


def load_split_frame(config_key: str, split: str = "test") -> pd.DataFrame:
    """Processed parquet rows (SEQN + target + features) for one split, missing targets dropped."""
    import json

    key = (config_key, split)
    if str(key) in _cache:
        return _cache[str(key)]
    rec = next(r for r in json.loads(config.DEFAULT_RESULTS_JSON.read_text())["results"] if r["config_key"] == config_key)
    disease = rec["target"]
    df = pd.read_parquet(PROCESSED / f"nhanes_{config_key}.parquet").dropna(subset=[f"target_{disease}"])
    sp = pd.read_csv(config.SPLITS_DIR / SPLIT_FILES[disease])
    out = df[df["SEQN"].isin(sp.loc[sp["split"] == split, "SEQN"])].reset_index(drop=True)
    _cache[str(key)] = out
    return out


def row_to_inputs(row: pd.Series, features: list[str]) -> dict:
    """Reverse-map one processed row (model codes) to an API `inputs` payload (human enums)."""
    gender = {1.0: "male", 2.0: "female"}
    smoking = {0.0: "never", 1.0: "former", 2.0: "current"}
    alcohol = {0.0: "never", 1.0: "former", 2.0: "current"}
    activity = {1.0: "high", 2.0: "moderate", 3.0: "low"}
    out: dict = {}
    for f in features:
        v = row[f]
        if pd.isna(v):
            continue
        if f == "gender":
            out[f] = gender[float(v)]
        elif f == "smoking_status":
            out[f] = smoking[float(v)]
        elif f == "alcohol_frequency":
            out[f] = alcohol[float(v)]
        elif f == "physical_activity_level":
            out[f] = activity[float(v)]
        elif f == "education_level":
            out[f] = int(v)
        else:
            out[f] = float(v)
    return out


def sample_rows(config_key: str, n: int = 50, seed: int = 42) -> pd.DataFrame:
    """n random test rows (seed 42) plus the 3 rows with the most missing predictors."""
    import json

    rec = next(r for r in json.loads(config.DEFAULT_RESULTS_JSON.read_text())["results"] if r["config_key"] == config_key)
    df = load_split_frame(config_key)
    feats = rec["features"]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    sparse = df[feats].isna().sum(axis=1).sort_values(ascending=False).index[:3]
    keep = list(dict.fromkeys([*idx.tolist(), *sparse.tolist()]))
    return df.loc[keep].reset_index(drop=True)

