"""Phase 2 controlled experiments (see MODEL_PERFORMANCE_INVESTIGATION.md §8).

Protocol
- Development set = train ∪ validation (frozen split files). 5-fold × 3-repeat stratified CV, seed 42.
- Every fitted component (imputers, models, calibrators) is fit inside the training fold only.
  Feature engineering is row-local (no fitted state), so it is applied before CV without leakage.
- Stage 1 (screening): uncalibrated OOF ROC-AUC / PR-AUC and train−val gap for all candidates.
- Stage 2 (full): per-fold calibration fit on inner 3-fold OOF predictions of the training fold
  (sigmoid / isotonic / beta / temperature), 10 metrics at a threshold chosen on pooled OOF
  calibrated predictions (sens ≥ 0.70, spec ≥ 0.60, max Youden J), B4 lab-masking shift.
- Final: the test split is read ONLY in `final_test()` and only for selected candidates, using the
  Stage 1F protocol (base on train, calibrator on validation, threshold fixed from Stage 2 CV).
Nothing outside backend/ml/experiments/ is written.

Usage: .venv/Scripts/python.exe backend/ml/experiments/scripts/phase2_experiments.py [stage1|stage2|final|all]
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path, PureWindowsPath

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[4]
BACKEND = ROOT / "backend"
PROC = BACKEND / "ml/data/processed/nhanes_2021_2023"
RES_DIR = BACKEND / "ml/experiments/results"
RES_DIR.mkdir(parents=True, exist_ok=True)
STAGE1F = {r["config_key"]: r for r in json.loads((BACKEND / "ml/evaluation/stage_1f/stage_1f_results.json").read_text())["results"]}
SEED = 42
N_SPLITS, N_REPEATS = 5, 3

LEAKAGE = {  # raw columns that must never feed a model (directly or via a derived feature)
    "cvd": {"MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "DIQ050", "DIQ070", "BPQ101D"},
    "diabetes": {"hba1c", "fasting_glucose", "DIQ010", "DIQ050", "DIQ070", "BPQ101D"},
    "hypertension": {"mean_sbp", "mean_dbp", "BPQ020", "BPQ101D"},
}
META = {"SEQN", "WTINT2YR", "WTMEC2YR", "WTSAF2YR", "SDMVSTRA", "SDMVPSU"}
LABS = ["hba1c", "fasting_glucose", "total_cholesterol", "hdl_cholesterol", "triglycerides", "ldl_cholesterol",
        "serum_creatinine", "blood_urea_nitrogen", "serum_uric_acid", "alt_enzyme", "ast_enzyme", "hemoglobin",
        "wbc_count", "platelet_count", "rdw", "serum_albumin"]
CBC_LIVER = ["hemoglobin", "wbc_count", "platelet_count", "rdw", "alt_enzyme", "ast_enzyme", "serum_albumin"]

# derived feature -> raw source columns (for the leakage guard)
DERIVED = {
    "non_hdl": ["total_cholesterol", "hdl_cholesterol"], "tg_hdl": ["triglycerides", "hdl_cholesterol"],
    "tc_hdl": ["total_cholesterol", "hdl_cholesterol"], "egfr": ["serum_creatinine", "age", "gender"],
    "pulse_pressure": ["mean_sbp", "mean_dbp"], "map": ["mean_sbp", "mean_dbp"],
    "smoke_never": ["smoking_status"], "smoke_former": ["smoking_status"], "smoke_current": ["smoking_status"],
    "alc_never": ["alcohol_frequency"], "alc_former": ["alcohol_frequency"], "alc_current": ["alcohol_frequency"],
    "miss_exam_bp": ["mean_pulse"], "miss_body": ["bmi"], "miss_lab": ["hemoglobin", "serum_creatinine"],
    "miss_fasting": ["triglycerides"],
}


# ----------------------------------------------------------------------------- data
def load_dev(key):
    rec = STAGE1F[key]
    d = rec["target"]
    df = pd.read_parquet(PROC / f"nhanes_{key}.parquet").dropna(subset=[f"target_{d}"])
    sp = pd.read_csv(PROC / "splits" / f"{d}_splits.csv")[["SEQN", "split"]]
    df = df.merge(sp, on="SEQN", how="inner")
    dev = df[df["split"].isin(["train", "validation"])].reset_index(drop=True)
    assert "test" not in set(dev["split"])
    return dev, f"target_{d}"


def egfr(df):
    scr, age, female = df["serum_creatinine"], df["age"], df["gender"] == 2
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    r = scr / k
    v = 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.2 * 0.9938 ** age * np.where(female, 1.012, 1.0)
    return v.where(scr.notna())


def make_features(raw, key, spec):
    """Row-local feature engineering. spec: set of flags."""
    rec = STAGE1F[key]
    feats = list(rec["features"])
    X = raw[feats].copy()
    mode_b = key.endswith("_b")
    disease = rec["target"]
    if mode_b and ("A1" in spec or "A2" in spec):
        if "ldl_cholesterol" in X:
            X = X.drop(columns=["ldl_cholesterol"])
        if "A1" in spec and "preventive" not in spec:
            X["non_hdl"] = raw["total_cholesterol"] - raw["hdl_cholesterol"]
    if mode_b and "A2" in spec:
        X["tg_hdl"] = raw["triglycerides"] / raw["hdl_cholesterol"]
        if "preventive" not in spec:
            X["tc_hdl"] = raw["total_cholesterol"] / raw["hdl_cholesterol"]
    if mode_b and "A3" in spec:
        X = X.drop(columns=[c for c in CBC_LIVER if c in X])
    if mode_b and "preventive" in spec:
        X = X.drop(columns=[c for c in ("total_cholesterol", "ldl_cholesterol", "non_hdl", "tc_hdl") if c in X])
    if mode_b and "A7" in spec:
        X["egfr"] = egfr(raw)
    if "A5" in spec and disease in ("cvd", "diabetes"):
        X["pulse_pressure"] = raw["mean_sbp"] - raw["mean_dbp"]
        X["map"] = raw["mean_dbp"] + (raw["mean_sbp"] - raw["mean_dbp"]) / 3
    if "A8" in spec:
        for col, pre in (("smoking_status", "smoke"), ("alcohol_frequency", "alc")):
            v = raw[col]
            for code, name in ((0, "never"), (1, "former"), (2, "current")):
                X[f"{pre}_{name}"] = (v == code).astype(float).where(v.notna())
            X = X.drop(columns=[col])
    if "A9" in spec:
        X["miss_exam_bp"] = raw["mean_pulse"].isna().astype(float)
        X["miss_body"] = raw["bmi"].isna().astype(float)
        if mode_b:
            X["miss_lab"] = (raw["hemoglobin"].isna() & raw["serum_creatinine"].isna()).astype(float)
            X["miss_fasting"] = raw["triglycerides"].isna().astype(float)
    guard(X.columns, disease)
    return X.astype("float64")


def guard(cols, disease):
    bad = LEAKAGE[disease]
    for c in cols:
        assert c not in META and not c.startswith("target_"), c
        srcs = DERIVED.get(c, [c])
        assert not (set(srcs) & bad), f"LEAKAGE: {c} uses {set(srcs) & bad} for {disease}"


# ----------------------------------------------------------------------------- models
def unwrap(m):
    cc = m.calibrated_classifiers_[0]
    est = getattr(cc, "estimator", None) or cc.base_estimator
    while type(est).__name__ == "FrozenEstimator":
        est = est.estimator
    return est


def frozen_base(key):
    rec = STAGE1F[key]
    return clone(unwrap(joblib.load(BACKEND.joinpath(*PureWindowsPath(rec["artifact_paths"]["joblib"]).parts))))


def rf(leaf, depth, ms, cw=None, n=500, mf=0.5, imputer="median"):
    imp = {"median": SimpleImputer(strategy="median"),
           "median_ind": SimpleImputer(strategy="median", add_indicator=True),
           "iterative": IterativeImputer(max_iter=10, random_state=SEED)}[imputer]
    return Pipeline([("imp", imp), ("clf", RandomForestClassifier(
        n_estimators=n, max_depth=depth, min_samples_leaf=leaf, max_samples=ms, max_features=mf,
        class_weight=cw, n_jobs=-1, random_state=SEED))])


def hgb(depth, leaf, l2, lr=0.05, it=300, cw=None):
    return HistGradientBoostingClassifier(max_depth=depth, min_samples_leaf=leaf, l2_regularization=l2,
                                          learning_rate=lr, max_iter=it, class_weight=cw, random_state=SEED)


def xgb(depth, mcw, lam, lr=0.03, n=400, spw=1.0):
    return XGBClassifier(max_depth=depth, min_child_weight=mcw, reg_lambda=lam, learning_rate=lr, n_estimators=n,
                         subsample=0.7, colsample_bytree=0.7, scale_pos_weight=spw, n_jobs=-1,
                         random_state=SEED, eval_metric="logloss")


def lr(C=1.0, cw=None, imputer="median_ind", l1_ratio=None):
    imp = SimpleImputer(strategy="median", add_indicator=(imputer == "median_ind")) if imputer != "iterative" \
        else IterativeImputer(max_iter=10, random_state=SEED)
    kw = dict(C=C, class_weight=cw, max_iter=5000)
    if l1_ratio is not None:
        kw.update(solver="saga", l1_ratio=l1_ratio)
    return Pipeline([("imp", imp), ("sc", StandardScaler()), ("clf", LogisticRegression(**kw))])


class SplineLR(Pipeline):
    pass


def spline_lr(spline_cols, n_cols, C=1.0, cw=None):
    others = [i for i in range(n_cols) if i not in spline_cols]
    ct = ColumnTransformer([("spl", SplineTransformer(n_knots=4, degree=3), spline_cols)], remainder="passthrough")
    _ = others
    return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("ct", ct),
                     ("sc", StandardScaler()), ("clf", LogisticRegression(C=C, class_weight=cw, max_iter=5000))])


def with_class_weight(model):
    m = clone(model)
    clf = m[-1] if isinstance(m, Pipeline) else m
    if isinstance(clf, XGBClassifier):
        clf.set_params(scale_pos_weight=-1.0)  # placeholder: set per fold from data
    else:
        clf.set_params(class_weight="balanced")
    return m


def fit(model, X, y):
    m = clone(model)
    clf = m[-1] if isinstance(m, Pipeline) else m
    if isinstance(clf, XGBClassifier) and clf.get_params()["scale_pos_weight"] == -1.0:
        clf.set_params(scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
    return m.fit(X, y)


def p1(m, X):
    return m.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------- metrics
def ece(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def binm(y, p, t):
    pr = p >= t
    tp, fp = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum())
    fn, tn = int((~pr & (y == 1)).sum()), int((~pr & (y == 0)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec, npv = tp / max(tp + fp, 1), tn / max(tn + fn, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "npv": npv, "f1": f1,
            "balanced_accuracy": (sens + spec) / 2}


def full_metrics(y, p, t):
    lo = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    cal = LogisticRegression(C=1e6).fit(lo.reshape(-1, 1), y)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "ece": ece(y, p), "cal_slope": float(cal.coef_[0][0]), "cal_intercept": float(cal.intercept_[0]),
            **binm(y, p, t)}


def choose_threshold(y, p):
    grid = np.round(np.arange(0.02, 0.905, 0.005), 3)
    rows = []
    for t in grid:
        b = binm(y, p, t)
        rows.append((t, b["sensitivity"], b["specificity"], b["sensitivity"] + b["specificity"] - 1))
    ok = [r for r in rows if r[1] >= 0.70 and r[2] >= 0.60]
    if not ok:
        ok = [r for r in rows if r[1] >= 0.70] or rows
    return float(max(ok, key=lambda r: r[3])[0])


def nb_ci(diffs, n_tr, n_te):
    """Nadeau–Bengio corrected repeated-CV t interval."""
    d = np.asarray(diffs)
    k = len(d)
    var = d.var(ddof=1) * (1 / k + n_te / n_tr)
    h = stats.t.ppf(0.975, k - 1) * np.sqrt(var)
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - h), float(d.mean() + h)],
            "significant": bool(d.mean() - h > 0 or d.mean() + h < 0)}


# ----------------------------------------------------------------------------- calibration
class Cal:
    def __init__(self, method):
        self.m = method

    def fit(self, p, y):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        if self.m == "sigmoid":
            self.f = LogisticRegression(C=1e5).fit(z.reshape(-1, 1), y)
        elif self.m == "temperature":
            self.f = LogisticRegression(C=1e5, fit_intercept=False).fit(z.reshape(-1, 1), y)
        elif self.m == "beta":
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            self.f = LogisticRegression(C=1e5).fit(np.c_[np.log(pc), -np.log(1 - pc)], y)
        elif self.m == "isotonic":
            self.f = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def __call__(self, p):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        if self.m in ("sigmoid", "temperature"):
            return self.f.predict_proba(z.reshape(-1, 1))[:, 1]
        if self.m == "beta":
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            return self.f.predict_proba(np.c_[np.log(pc), -np.log(1 - pc)])[:, 1]
        return self.f.predict(p)


CAL_METHODS = ["sigmoid", "isotonic", "beta", "temperature"]


# ----------------------------------------------------------------------------- CV engines
def folds(y):
    return list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(np.zeros(len(y)), y))


def stage1_eval(model, X, y, F):
    aucs, prs, gaps, oof = [], [], [], np.zeros((N_REPEATS, len(y)))
    for k, (tr, te) in enumerate(F):
        m = fit(model, X.iloc[tr], y[tr])
        pte = p1(m, X.iloc[te])
        oof[k // N_SPLITS, te] = pte
        a = roc_auc_score(y[te], pte)
        aucs.append(a)
        prs.append(average_precision_score(y[te], pte))
        gaps.append(roc_auc_score(y[tr], p1(m, X.iloc[tr])) - a)
    return {"auc": aucs, "pr": prs, "gap": gaps}, oof


def summarize(r):
    return {k: [float(np.mean(v)), float(np.std(v))] for k, v in r.items()}


# ----------------------------------------------------------------------------- stage 1
def candidates_trackC(key, X):
    fam = type(frozen_base(key)[-1] if isinstance(frozen_base(key), Pipeline) else frozen_base(key)).__name__
    C = {"C0_frozen_clone": frozen_base(key),
         "C1_rf_l20_d6_ms07": rf(20, 6, 0.7), "C1_rf_l40_d6_ms06": rf(40, 6, 0.6),
         "C1_rf_l80_d4_ms05": rf(80, 4, 0.5), "C1_rf_l40_d4_ms07": rf(40, 4, 0.7),
         "C2_hgb_d3_l50": hgb(3, 50, 10.0), "C2_hgb_d4_l100": hgb(4, 100, 20.0),
         "C2_xgb_d3_mcw50": xgb(3, 50, 10.0), "C2_xgb_d4_mcw30": xgb(4, 30, 5.0),
         "C4_lr_medind": lr(1.0)}
    if key == "hypertension_mode_a":
        cols = list(X.columns)
        C["C4_spline_lr"] = spline_lr([cols.index(c) for c in ("age", "bmi", "waist_circumference")], len(cols))
        C["C4_elasticnet_lr"] = lr(1.0, l1_ratio=0.5)
    return C, fam


def run_stage1(key):
    t0 = time.time()
    dev, tcol = load_dev(key)
    y = dev[tcol].astype(int).values
    F = folds(y)
    out = {"n_dev": len(dev), "prevalence": float(y.mean()), "trackA": {}, "trackB": {}, "trackC": {}, "combo": {}}
    oofs = {}
    base_X = make_features(dev, key, set())
    # ---- Track C (models on frozen features)
    C, fam = candidates_trackC(key, base_X)
    out["frozen_family"] = fam
    for name, model in C.items():
        r, oof = stage1_eval(model, base_X, y, F)
        out["trackC"][name] = {"raw": r, **summarize(r)}
        oofs[name] = oof
        print(f"  [{key}] C {name}: AUC {np.mean(r['auc']):.4f} gap {np.mean(r['gap']):.3f}", flush=True)
    # ensemble (C6): mean of best linear and best GBDT OOF ranks per repeat
    lin = max((n for n in out["trackC"] if "lr" in n), key=lambda n: out["trackC"][n]["auc"][0])
    gb = max((n for n in out["trackC"] if ("hgb" in n or "xgb" in n)), key=lambda n: out["trackC"][n]["auc"][0])
    ens = {"auc": [], "pr": [], "gap": []}
    for k, (tr, te) in enumerate(F):
        r_ = k // N_SPLITS
        a = pd.Series(oofs[lin][r_, te]).rank().values + pd.Series(oofs[gb][r_, te]).rank().values
        ens["auc"].append(roc_auc_score(y[te], a))
        ens["pr"].append(average_precision_score(y[te], a))
        ens["gap"].append(np.nan)
    out["trackC"][f"C6_ensemble[{lin}+{gb}]"] = {"raw": ens, **summarize(ens)}
    # ---- Track B (imputation; frozen features)
    B = {"B0_median(frozen-style RF)": rf(40, 6, 0.6, imputer="median"),
         "B1_median+indicator RF": rf(40, 6, 0.6, imputer="median_ind"),
         "B2_iterative RF": rf(40, 6, 0.6, imputer="iterative"),
         "B1_median+indicator LR": lr(1.0, imputer="median_ind"),
         "B0_median LR": lr(1.0, imputer="median"),
         "B3_native_nan HGB": hgb(3, 50, 10.0)}
    for name, model in B.items():
        r, _ = stage1_eval(model, base_X, y, F)
        out["trackB"][name] = {"raw": r, **summarize(r)}
        print(f"  [{key}] B {name}: AUC {np.mean(r['auc']):.4f}", flush=True)
    # ---- Track A (feature sets) with reference model = best Track C non-ensemble model
    best_c = max((n for n in out["trackC"] if not n.startswith("C6")), key=lambda n: out["trackC"][n]["auc"][0])
    ref = C[best_c]
    out["trackA_reference_model"] = best_c
    mode_b = key.endswith("_b")
    dis = STAGE1F[key]["target"]
    specs = {"A0_frozen_features": set(), "A5_pulse_pressure_map": {"A5"}, "A8_onehot_smoke_alcohol": {"A8"},
             "A9_component_missing_flags": {"A9"}}
    if dis == "hypertension":
        specs.pop("A5_pulse_pressure_map")
    if mode_b:
        specs.update({"A1_drop_ldl_add_nonhdl": {"A1"}, "A2_lipid_ratios": {"A1", "A2"}, "A3_drop_cbc_liver": {"A3"},
                      "A7_egfr": {"A7"}})
        if dis in ("cvd", "diabetes"):
            specs["A4_preventive_no_tc_ldl"] = {"preventive", "A1"}
    combo = {"A9", "A8"} | ({"A5"} if dis != "hypertension" else set())
    if mode_b:
        combo |= {"A1", "A2", "A3", "A7"}
    specs["A_combo"] = combo
    if mode_b and dis in ("cvd", "diabetes"):
        specs["A4_preventive_combo"] = (combo - {"A1"}) | {"preventive", "A2"}
    out["feature_specs"] = {k: sorted(v) for k, v in specs.items()}
    for name, spec in specs.items():
        Xs = make_features(dev, key, spec)
        model = ref
        if "spline" in best_c:
            cols = list(Xs.columns)
            model = spline_lr([cols.index(c) for c in ("age", "bmi", "waist_circumference")], len(cols))
        r, _ = stage1_eval(model, Xs, y, F)
        out["trackA"][name] = {"raw": r, **summarize(r), "n_features": Xs.shape[1], "features": list(Xs.columns)}
        print(f"  [{key}] A {name} ({Xs.shape[1]}f): AUC {np.mean(r['auc']):.4f}", flush=True)
    # paired deltas vs frozen clone on frozen features (same folds)
    n_te = len(y) / N_SPLITS
    n_tr = len(y) - n_te
    ref_auc = out["trackC"]["C0_frozen_clone"]["raw"]["auc"]
    ref_pr = out["trackC"]["C0_frozen_clone"]["raw"]["pr"]
    for track in ("trackA", "trackB", "trackC"):
        for name, v in out[track].items():
            v["delta_auc_vs_frozen"] = nb_ci(np.array(v["raw"]["auc"]) - np.array(ref_auc), n_tr, n_te)
            v["delta_pr_vs_frozen"] = nb_ci(np.array(v["raw"]["pr"]) - np.array(ref_pr), n_tr, n_te)
    out["seconds"] = time.time() - t0
    (RES_DIR / f"stage1_{key}.json").write_text(json.dumps(out, default=float))
    return out


# ----------------------------------------------------------------------------- stage 2
def build_model(key, name, X):
    C, _ = candidates_trackC(key, X)
    if name in C:
        return C[name]
    raise KeyError(name)


def stage2_eval(model, X, y, F, raw_dev, key, spec, mask_labs):
    """Full protocol with inner-OOF calibration; returns per-method pooled OOF + fold info."""
    per = {m: np.zeros((N_REPEATS, len(y))) for m in CAL_METHODS}
    gaps, shifts_mean, shifts_abs = [], [], []
    for k, (tr, te) in enumerate(F):
        Xtr, ytr = X.iloc[tr], y[tr]
        base = fit(model, Xtr, ytr)
        inner = np.zeros(len(tr))
        for itr, ite in StratifiedKFold(3, shuffle=True, random_state=SEED + k).split(Xtr, ytr):
            inner[ite] = p1(fit(model, Xtr.iloc[itr], ytr[itr]), Xtr.iloc[ite])
        pte = p1(base, X.iloc[te])
        cals = {m: Cal(m).fit(inner, ytr) for m in CAL_METHODS}
        for m in CAL_METHODS:
            per[m][k // N_SPLITS, te] = cals[m](pte)
        gaps.append(roc_auc_score(ytr, p1(base, Xtr)) - roc_auc_score(y[te], pte))
        if mask_labs:
            rawm = raw_dev.iloc[te].copy()
            rawm[[c for c in LABS if c in rawm]] = np.nan
            Xm = make_features(rawm, key, spec)
            pm = cals["sigmoid"](p1(base, Xm))
            po = per["sigmoid"][k // N_SPLITS, te]
            shifts_mean.append(float((pm - po).mean()))
            shifts_abs.append(float(np.abs(pm - po).mean()))
    return per, gaps, shifts_mean, shifts_abs


def summarize_stage2(per, gaps, y, F, shifts_mean, shifts_abs):
    res = {"gap": [float(np.mean(gaps)), float(np.std(gaps))]}
    for m in CAL_METHODS:
        pooled = np.concatenate([per[m][r] for r in range(N_REPEATS)])
        ypool = np.tile(y, N_REPEATS)
        t = choose_threshold(ypool, pooled)
        fm = [full_metrics(y[te], per[m][k // N_SPLITS, te], t) for k, (tr, te) in enumerate(F)]
        agg = {kk: [float(np.mean([f[kk] for f in fm])), float(np.std([f[kk] for f in fm]))] for kk in fm[0]}
        res[m] = {"threshold": t, "folds": fm, "mean_sd": agg}
    if shifts_mean:
        res["B4_shift_mean"] = [float(np.mean(shifts_mean)), float(np.std(shifts_mean))]
        res["B4_shift_abs"] = [float(np.mean(shifts_abs)), float(np.std(shifts_abs))]
    return res


def pick_best(s1, key):
    """Best stage-1 candidate across Track C models × Track A feature sets (combo runs)."""
    return s1["best"]


def run_combo_and_select(key, s1):
    """Combine best Track C model with the best Track A feature sets; pick the overall best."""
    dev, tcol = load_dev(key)
    y = dev[tcol].astype(int).values
    F = folds(y)
    dis = STAGE1F[key]["target"]
    model_names = [n for n in s1["trackC"] if not n.startswith("C6")]
    top_models = sorted(model_names, key=lambda n: -s1["trackC"][n]["auc"][0])[:2]
    a_sorted = sorted((n for n in s1["trackA"] if "preventive" not in n), key=lambda n: -s1["trackA"][n]["auc"][0])
    top_fs = list(dict.fromkeys(["A0_frozen_features", *a_sorted[:2]]))
    specs = {k: set(v) for k, v in s1["feature_specs"].items()}
    n_te = len(y) / N_SPLITS
    n_tr = len(y) - n_te
    ref_auc = s1["trackC"]["C0_frozen_clone"]["raw"]["auc"]
    ref_pr = s1["trackC"]["C0_frozen_clone"]["raw"]["pr"]
    combo = {}
    for mn in top_models:
        for fs in top_fs:
            Xs = make_features(dev, key, specs[fs])
            model = build_model(key, mn, Xs)
            r, _ = stage1_eval(model, Xs, y, F)
            combo[f"{mn} × {fs}"] = {"raw": r, **summarize(r), "model": mn, "fs": fs,
                                    "delta_auc_vs_frozen": nb_ci(np.array(r["auc"]) - np.array(ref_auc), n_tr, n_te),
                                    "delta_pr_vs_frozen": nb_ci(np.array(r["pr"]) - np.array(ref_pr), n_tr, n_te)}
            print(f"  [{key}] combo {mn} × {fs}: AUC {np.mean(r['auc']):.4f}", flush=True)
    best = max(combo, key=lambda n: (round(combo[n]["auc"][0], 3), combo[n]["pr"][0]))
    s1["combo"] = combo
    s1["best"] = {"model": combo[best]["model"], "fs": combo[best]["fs"]}
    if dis in ("cvd", "diabetes") and key.endswith("_b"):
        prev_fs = max((n for n in s1["trackA"] if "preventive" in n), key=lambda n: s1["trackA"][n]["auc"][0])
        s1["preventive"] = {"model": combo[best]["model"], "fs": prev_fs}
    (RES_DIR / f"stage1_{key}.json").write_text(json.dumps(s1, default=float))
    return s1


def run_stage2(key):
    t0 = time.time()
    s1 = json.loads((RES_DIR / f"stage1_{key}.json").read_text())
    if "best" not in s1:
        s1 = run_combo_and_select(key, s1)
    dev, tcol = load_dev(key)
    y = dev[tcol].astype(int).values
    F = folds(y)
    specs = {k: set(v) for k, v in s1["feature_specs"].items()}
    mode_b = key.endswith("_b")
    cands = {"frozen": ("C0_frozen_clone", "A0_frozen_features", False),
             "best": (s1["best"]["model"], s1["best"]["fs"], False),
             "best_balanced": (s1["best"]["model"], s1["best"]["fs"], True)}
    if "preventive" in s1:
        cands["preventive"] = (s1["preventive"]["model"], s1["preventive"]["fs"], False)
    out = {"candidates": {}}
    for cname, (mn, fs, bal) in cands.items():
        Xs = make_features(dev, key, specs[fs])
        model = build_model(key, mn, Xs)
        if bal:
            model = with_class_weight(model)
        per, gaps, sm, sa = stage2_eval(model, Xs, y, F, dev, key, specs[fs], mode_b)
        res = summarize_stage2(per, gaps, y, F, sm, sa)
        res.update({"model": mn, "feature_set": fs, "balanced": bal, "n_features": Xs.shape[1],
                    "features": list(Xs.columns)})
        out["candidates"][cname] = res
        print(f"  [{key}] stage2 {cname}: sigmoid AUC {res['sigmoid']['mean_sd']['roc_auc'][0]:.4f} "
              f"Brier {res['sigmoid']['mean_sd']['brier'][0]:.4f}", flush=True)
    # paired deltas (sigmoid) vs frozen for every metric
    n_te = len(y) / N_SPLITS
    n_tr = len(y) - n_te
    fz = out["candidates"]["frozen"]
    for cname, res in out["candidates"].items():
        res["calibration_choice"] = choose_cal(res)
        m = res["calibration_choice"]
        res["delta_vs_frozen"] = {kk: nb_ci([a[kk] - b[kk] for a, b in zip(res[m]["folds"], fz["sigmoid"]["folds"])], n_tr, n_te)
                                  for kk in ("roc_auc", "pr_auc", "brier", "ece", "sensitivity", "specificity",
                                             "precision", "npv", "f1", "balanced_accuracy")}
    out["seconds"] = time.time() - t0
    (RES_DIR / f"stage2_{key}.json").write_text(json.dumps(out, default=float))
    return out


def choose_cal(res):
    """Sigmoid unless another method improves mean Brier by > 0.001 and ECE by > 0.005."""
    sb, se = res["sigmoid"]["mean_sd"]["brier"][0], res["sigmoid"]["mean_sd"]["ece"][0]
    best = "sigmoid"
    for m in ("isotonic", "beta", "temperature"):
        b, e = res[m]["mean_sd"]["brier"][0], res[m]["mean_sd"]["ece"][0]
        if b < sb - 0.001 and e < se - 0.005:
            best, sb, se = m, b, e
    return best


# ----------------------------------------------------------------------------- final (test, once)
def final_test():
    rows = {}
    for key in STAGE1F:
        s2 = json.loads((RES_DIR / f"stage2_{key}.json").read_text())
        s1 = json.loads((RES_DIR / f"stage1_{key}.json").read_text())
        specs = {k: set(v) for k, v in s1["feature_specs"].items()}
        rec = STAGE1F[key]
        d = rec["target"]
        df = pd.read_parquet(PROC / f"nhanes_{key}.parquet").dropna(subset=[f"target_{d}"])
        sp = pd.read_csv(PROC / "splits" / f"{d}_splits.csv")[["SEQN", "split"]]
        df = df.merge(sp, on="SEQN", how="inner")
        tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "validation", "test"))
        tcol = f"target_{d}"
        # selected experimental candidate = better of best / best_balanced by CV (AUC then Brier)
        c = s2["candidates"]
        pick = max(("best", "best_balanced"), key=lambda n: (round(c[n][c[n]["calibration_choice"]]["mean_sd"]["roc_auc"][0], 3),
                                                            -c[n][c[n]["calibration_choice"]]["mean_sd"]["brier"][0]))
        to_eval = {"experimental": pick}
        if "preventive" in c:
            to_eval["preventive"] = "preventive"
        rows[key] = {"published_frozen": {**rec["test_metrics"], **rec["test_binary_at_optimized_threshold"]}}
        for label, cname in to_eval.items():
            cc = c[cname]
            spec = specs[cc["feature_set"]]
            Xtr, Xva, Xte = (make_features(z, key, spec) for z in (tr, va, te))
            model = build_model(key, cc["model"], Xtr)
            if cc["balanced"]:
                model = with_class_weight(model)
            base = fit(model, Xtr, tr[tcol].astype(int).values)            # Stage 1F protocol: base on train
            cal = Cal(cc["calibration_choice"]).fit(p1(base, Xva), va[tcol].astype(int).values)  # calibrator on val
            p = cal(p1(base, Xte))
            t = cc[cc["calibration_choice"]]["threshold"]                   # threshold fixed by CV beforehand
            yte = te[tcol].astype(int).values
            m = full_metrics(yte, p, t)
            m["threshold"] = t
            rows[key][label] = {"candidate": cname, "model": cc["model"], "feature_set": cc["feature_set"],
                                "calibration": cc["calibration_choice"], "balanced": cc["balanced"], **m}
            joblib.dump({"base": base, "calibrator": cal, "features": list(Xtr.columns), "threshold": t},
                        BACKEND / f"ml/experiments/models/{key}_{label}.joblib")
            print(f"[TEST] {key} {label}: AUC {m['roc_auc']:.4f} PR {m['pr_auc']:.4f} sens {m['sensitivity']:.3f}", flush=True)
    (RES_DIR / "final_test.json").write_text(json.dumps(rows, default=float, indent=1))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    (BACKEND / "ml/experiments/models").mkdir(parents=True, exist_ok=True)
    keys = sys.argv[2:] or list(STAGE1F)
    if what in ("stage1", "all"):
        for k in keys:
            print(f"== stage1 {k}", flush=True)
            s1 = run_stage1(k)
            run_combo_and_select(k, s1)
    if what in ("stage2", "all"):
        for k in keys:
            print(f"== stage2 {k}", flush=True)
            run_stage2(k)
    if what == "final":
        final_test()
