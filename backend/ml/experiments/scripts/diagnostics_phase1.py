"""Phase 1 read-only diagnostics for the frozen Stage 1F models.

TEST-SET QUARANTINE: this script reads only rows whose split is "train" or "validation".
Rows in the "test" split are dropped immediately after loading and never touched.
Nothing under backend/ml/data, scripts, evaluation, api or frontend is written.

Usage (repo root):
    .\\.venv\\Scripts\\python.exe backend/ml/experiments/scripts/diagnostics_phase1.py
Writes backend/ml/experiments/results/phase1_diagnostics.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path, PureWindowsPath

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[4]
BACKEND = ROOT / "backend"
PROC = BACKEND / "ml/data/processed/nhanes_2021_2023"
RESULTS = json.loads((BACKEND / "ml/evaluation/stage_1f/stage_1f_results.json").read_text())["results"]
OUT = BACKEND / "ml/experiments/results/phase1_diagnostics.json"
SEED = 42


def unwrap(m):
    cc = m.calibrated_classifiers_[0]
    est = getattr(cc, "estimator", None) or cc.base_estimator
    while type(est).__name__ == "FrozenEstimator":
        est = est.estimator
    return est


def load(rec):
    disease = rec["target"]
    df = pd.read_parquet(PROC / f"nhanes_{rec['config_key']}.parquet").dropna(subset=[f"target_{disease}"])
    sp = pd.read_csv(PROC / "splits" / f"{disease}_splits.csv")[["SEQN", "split"]]
    df = df.merge(sp, on="SEQN", how="inner")
    df = df[df["split"].isin(["train", "validation"])].reset_index(drop=True)  # quarantine test
    assert "test" not in set(df["split"])
    return df, f"target_{disease}"


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def cal_slope_intercept(y, p):
    lo = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6).fit(lo.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def metrics(y, p):
    s, i = cal_slope_intercept(y, p)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "ece": ece(y, p), "cal_slope": s, "cal_intercept": i, "mean_p": float(p.mean()), "prevalence": float(y.mean())}


def op_point(y, p, t):
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {"t": round(float(t), 3), "sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1), "flag_rate": float(pred.mean())}


def paired_boot(y, p1, p2, n=1000):
    rng = np.random.RandomState(SEED)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p2[i]) - roc_auc_score(y[i], p1[i]))
    d = np.array(d)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "sd": float(d.std())}


def vif(X):
    Xc = X.dropna()
    Z = ((Xc - Xc.mean()) / Xc.std(ddof=0)).values
    out = {}
    for j, c in enumerate(X.columns):
        others = np.delete(Z, j, axis=1)
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Z)), others], Z[:, j], rcond=None)
        r2 = 1 - ((Z[:, j] - np.c_[np.ones(len(Z)), others] @ beta) ** 2).sum() / (Z[:, j] ** 2).sum()
        out[c] = float(1 / max(1 - r2, 1e-9))
    return out, len(Xc)


report = {"note": "train/validation only; test split quarantined", "configs": {}}
val_probs = {}

for rec in RESULTS:
    key = rec["config_key"]
    feats = rec["features"]
    thr = rec["selected_threshold"]
    df, tcol = load(rec)
    tr, va = df[df.split == "train"], df[df.split == "validation"]
    Xtr, ytr, Xva, yva = tr[feats], tr[tcol].astype(int).values, va[feats], va[tcol].astype(int).values
    model = joblib.load(BACKEND.joinpath(*PureWindowsPath(rec["artifact_paths"]["joblib"]).parts))
    base = unwrap(model)
    clf = base[-1] if isinstance(base, Pipeline) else base

    pb_tr, pb_va = base.predict_proba(Xtr)[:, 1], base.predict_proba(Xva)[:, 1]
    pc_va = model.predict_proba(Xva)[:, 1]
    val_probs[key] = (yva, pb_va, va["SEQN"].values)
    r = {"model": rec["model_name"], "base_class": type(clf).__name__, "n_features": len(feats), "threshold": thr,
         "n_train": len(tr), "n_val": len(va),
         "base_train": metrics(ytr, pb_tr), "base_val": metrics(yva, pb_va),
         "calibrated_val_in_sample": metrics(yva, pc_va)}
    r["gap_roc_auc"] = r["base_train"]["roc_auc"] - r["base_val"]["roc_auc"]
    r["gap_pr_auc"] = r["base_train"]["pr_auc"] - r["base_val"]["pr_auc"]
    try:
        r["hyperparams"] = {k: v for k, v in clf.get_params().items() if k in (
            "n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features", "class_weight",
            "learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization", "reg_lambda", "reg_alpha",
            "subsample", "colsample_bytree", "min_child_weight", "C", "penalty", "scale_pos_weight")}
        r["hyperparams"] = json.loads(json.dumps(r["hyperparams"], default=str))
    except Exception:
        pass

    # operating-point sensitivity around the locked threshold (validation, calibrated)
    r["threshold_sweep_val"] = [op_point(yva, pc_va, t) for t in np.round(np.arange(thr - 0.05, thr + 0.0501, 0.01), 3) if 0 < t < 1]
    r["prob_quantiles_val"] = {q: float(np.quantile(pc_va, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
    r["share_val_within_0.03_of_threshold"] = float((np.abs(pc_va - thr) <= 0.03).mean())

    # permutation importance on validation (base model, ROC-AUC) — no retraining
    pi = permutation_importance(base, Xva, yva, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1)
    r["perm_importance_val"] = {f: {"mean": float(m), "sd": float(s)} for f, m, s in
                                sorted(zip(feats, pi.importances_mean, pi.importances_std), key=lambda z: -z[1])}

    # missingness: rate on train/val, target rate observed vs missing (train), indicator-only AUC
    miss = {}
    for f in feats:
        m_tr = Xtr[f].isna().values
        if m_tr.sum() < 20:
            continue
        miss[f] = {"miss_train": float(m_tr.mean()), "miss_val": float(Xva[f].isna().mean()),
                   "target_rate_missing": float(ytr[m_tr].mean()), "target_rate_observed": float(ytr[~m_tr].mean())}
    r["missingness"] = miss
    Mtr, Mva = Xtr.isna().astype(int), Xva.isna().astype(int)
    keep = [c for c in feats if Mtr[c].sum() >= 20]
    if keep:
        li = LogisticRegression(max_iter=2000).fit(Mtr[keep], ytr)
        r["missing_indicators_only_val_auc"] = float(roc_auc_score(yva, li.predict_proba(Mva[keep])[:, 1]))
        if isinstance(base, Pipeline):  # do residual errors on val correlate with missingness? (median-imputed models)
            resid = yva - pc_va
            r["val_residual_by_any_lab_missing"] = {
                "mean_resid_rows_with_missing": float(resid[Mva[keep].sum(1).values > 0].mean()) if (Mva[keep].sum(1) > 0).any() else None,
                "mean_resid_rows_complete": float(resid[Mva[keep].sum(1).values == 0].mean()) if (Mva[keep].sum(1) == 0).any() else None}

    # mutual information (train, median-imputed for MI only) and collinearity (train)
    mi = mutual_info_classif(Xtr.fillna(Xtr.median()), ytr, random_state=SEED)
    r["mutual_info_train"] = dict(sorted(((f, float(v)) for f, v in zip(feats, mi)), key=lambda z: -z[1]))
    cont = [f for f in feats if Xtr[f].nunique() > 6]
    corr = Xtr[cont].corr(method="spearman")
    pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(cont) for b in cont[i + 1:] if abs(corr.loc[a, b]) >= 0.6]
    r["collinear_pairs_train_spearman_ge_0.6"] = sorted(pairs, key=lambda z: -abs(z[2]))
    v, ncc = vif(Xtr[cont])
    r["vif_train_complete_case"] = {"n_rows": ncc, "values": dict(sorted(v.items(), key=lambda z: -z[1]))}

    # encoding-quirk checks (train)
    for f in ("physical_activity_level", "alcohol_frequency", "smoking_status"):
        if f in feats:
            g = tr.groupby(tr[f].fillna(-1))[tcol].agg(["count", "mean"])
            r.setdefault("category_target_rates_train", {})[f] = {str(k): {"n": int(c), "rate": float(m)} for k, (c, m) in g.iterrows()}
    report["configs"][key] = r
    print(f"done {key}")

# Mode B vs A paired bootstrap on validation (base models; same participants)
report["mode_b_minus_a_val_auc"] = {}
for d in ("cvd", "diabetes", "hypertension"):
    ya, pa, sa = val_probs[f"{d}_mode_a"]
    yb, pb, sb = val_probs[f"{d}_mode_b"]
    assert (sa == sb).all() and (ya == yb).all()
    report["mode_b_minus_a_val_auc"][d] = paired_boot(ya, pa, pb)

# Lab-group ablation for Mode B via permutation of feature GROUPS on validation (no retraining)
LAB_GROUPS = {
    "glycaemic": ["hba1c", "fasting_glucose"],
    "lipids": ["total_cholesterol", "hdl_cholesterol", "triglycerides", "ldl_cholesterol"],
    "renal": ["serum_creatinine", "blood_urea_nitrogen", "serum_uric_acid"],
    "liver": ["alt_enzyme", "ast_enzyme", "serum_albumin"],
    "cbc": ["hemoglobin", "wbc_count", "platelet_count", "rdw"],
}
report["mode_b_group_permutation_val"] = {}
for rec in RESULTS:
    if not rec["config_key"].endswith("_b"):
        continue
    df, tcol = load(rec)
    va = df[df.split == "validation"]
    X, y = va[rec["features"]].copy(), va[tcol].astype(int).values
    base = unwrap(joblib.load(BACKEND.joinpath(*PureWindowsPath(rec["artifact_paths"]["joblib"]).parts)))
    ref = roc_auc_score(y, base.predict_proba(X)[:, 1])
    rng = np.random.RandomState(SEED)
    out = {"ref_auc": ref}
    for g, cols in list(LAB_GROUPS.items()) + [("all_labs", sum(LAB_GROUPS.values(), []))]:
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = []
        for _ in range(5):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].values[perm]
            drops.append(ref - roc_auc_score(y, base.predict_proba(Xp)[:, 1]))
        out[g] = {"auc_drop_mean": float(np.mean(drops)), "auc_drop_sd": float(np.std(drops))}
    report["mode_b_group_permutation_val"][rec["config_key"]] = out

# Published Stage 1F static baselines (no recomputation on test)
report["stage_1f_published"] = {r["config_key"]: {"test_metrics": r["test_metrics"],
                                                  "test_binary": r["test_binary_at_optimized_threshold"],
                                                  "val_at_threshold": r["validation_metrics_at_opt_threshold"],
                                                  "calibration_comparison": r["validation_calibration_comparison"]}
                                for r in RESULTS}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=1, default=float))
print("wrote", OUT)
