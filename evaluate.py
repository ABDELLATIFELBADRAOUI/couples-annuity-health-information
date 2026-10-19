"""
evaluate.py -- out-of-sample evaluation of the nested couples models on the held-out couples.

Targets  : husband, wife, joint life (first death), last survivor (second death)
Models   : M0 independence, M1 + partner covariates, M2 + widowhood, M3 + frailty (Bayesian urn models),
           MD (urns with dependence, no covariates) if its chains exist,
           Cox (own covariates) and Cox + partner covariates, both combined under independence,
           Gompertz margins alone and with a Frank copula (if copula_baseline.py has been run),
           DeepSurv and DeepHit with the same two sets of covariates (if deep_models.py has been run)
Metrics  : integrated Brier score over years 1-10 and Brier score at 10 years (inverse-probability-of-
           censoring weights), time-dependent AUC at 5 and 10 years, Uno's C at 10 years (scikit-survival),
           calibration at 10 years by tertile of predicted risk against Kaplan-Meier.
Intervals: paired bootstrap of the held-out couples for the difference with M0. The bootstrap uses the
           metrics of metrics_util.metrics_own (same definitions as scikit-survival, see check_metrics.py),
           which accept every resample.
"""
import glob
import json
import os

import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from metrics_util import TIMES, truncate, metrics, metrics_own, censoring_survival, calibration
from pipeline import (load_couples, split_indices, fit_scaler, make_data, prior_arrays, age_max_of,
                      outcomes, predict_states, load_draws, SPLIT_SEED)

RES = os.environ.get("CARUP_RESULTS_DIR", "results_eval")          # where the chains are, and the output
BAYES = os.environ.get("CARUP_MODELS", "M0,M1,M2,M3").split(",")
BAYES += [v for v in ("MD",) if glob.glob(f"{RES}/{v}_train_*.npz") and v not in BAYES]
HORIZON, N_DRAWS, N_BOOT = 12, 400, int(os.environ.get("CARUP_N_BOOT", "1000"))
TARGETS = ["husband", "wife", "joint_life", "last_survivor"]

d = load_couples("cohort_w10.csv")
tr, te = split_indices(d)
dtr, dte = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)
sc = fit_scaler(dtr)
data_tr, data_te = make_data(dtr, sc), make_data(dte, sc)
ages, pstar, _ = prior_arrays("prior_table.csv", age_max_of(d))


o_tr_full = outcomes(dtr)                                # full follow-up, used to train the Cox models
o_tr = {k: truncate(v) for k, v in o_tr_full.items()}
o_te_full = outcomes(dte)
o_te = {k: truncate(v) for k, v in o_te_full.items()}

# ------------------------------------------------------------------ predictions
pred = {}
for v in BAYES:
    files = sorted(glob.glob(f"{RES}/{v}_train_*.npz"))
    assert len(files) == 2, f"two chain files are expected for {v} in {RES}, found {len(files)}"
    pred[v] = predict_states(load_draws(files, n_draws=N_DRAWS), data_te, int(ages[0]), HORIZON)


def cox_curves(with_partner):
    curves = []
    for s, name in enumerate(["husband", "wife"]):
        def design(data):
            cols = [data.age0[:, s:s + 1].astype(float), data.x_own[:, s, :]]
            if with_partner:
                cols.append(data.x_cross[:, s, :])
            return np.concatenate(cols, axis=1)
        t, e = o_tr_full[name]
        model = CoxPHSurvivalAnalysis().fit(design(data_tr), Surv.from_arrays(e, t))
        fns = model.predict_survival_function(design(data_te))
        grid = np.arange(1, HORIZON + 1).astype(float)
        lo, hi = fns[0].domain
        curves.append(np.array([fn(np.clip(grid, lo, hi)) for fn in fns]))
    h, w = curves
    return {"husband": h, "wife": w, "joint_life": h * w, "last_survivor": h + w - h * w}


pred["Cox"] = cox_curves(False)
pred["Cox+partner"] = cox_curves(True)
MODELS = BAYES + ["Cox", "Cox+partner"]
# Gompertz margins and Frank copula, if copula_baseline.py has been run for this split
if os.path.exists(f"{RES}/copula_predictions.npz"):
    z = np.load(f"{RES}/copula_predictions.npz")
    assert int(z["split_seed"]) == SPLIT_SEED, "copula_predictions.npz belongs to another split"
    for key in [k for k in z.files if "|" in k]:
        name, tg = key.split("|")
        pred.setdefault(name, {})[tg] = z[key]
    MODELS += [m for m in ["Gompertz", "Copula", "Copula+partner"] if m in pred]
# DeepSurv and DeepHit, if deep_models.py has been run for this split
if os.path.exists(f"{RES}/deep_predictions.npz"):
    z = np.load(f"{RES}/deep_predictions.npz")
    assert int(z["split_seed"]) == SPLIT_SEED, "deep_predictions.npz belongs to another split"
    for key in [k for k in z.files if "|" in k]:
        name, tg = key.split("|")
        pred.setdefault(name, {})[tg] = z[key]
    MODELS += [m for m in ["DeepSurv", "DeepSurv+partner", "DeepHit", "DeepHit+partner"] if m in pred]


# ------------------------------------------------------------------ metrics
out = {"setting": dict(train=len(dtr), test=len(dte), horizon_years=10, posterior_draws=N_DRAWS, bootstrap=N_BOOT),
       "targets": {}}
rng = np.random.default_rng(2026)
boot_idx = [rng.integers(0, len(dte), len(dte)) for _ in range(N_BOOT)]
boot_all = {}
for tg in TARGETS:
    y_train = Surv.from_arrays(o_tr[tg][1], o_tr[tg][0])
    t_te, e_te = o_te[tg]
    y_test = Surv.from_arrays(e_te, t_te)
    res = {m: metrics(y_train, y_test, pred[m][tg]) for m in MODELS}
    for m in MODELS:
        res[m]["calibration"] = calibration(t_te, e_te, pred[m][tg][:, 9])
    # paired bootstrap: the same resamples of the held-out couples for every model (kept in evaluation_boot.npz,
    # so that any two models can be compared), and the differences with M0
    G = censoring_survival(*o_tr[tg])
    boot = {m: {k: np.zeros(N_BOOT) for k in ("ibs", "brier10", "auc5", "auc10")} for m in MODELS}
    for b, idx in enumerate(boot_idx):
        for m in MODELS:
            mm = metrics_own(G, t_te[idx], e_te[idx], pred[m][tg][idx])
            for k in mm:
                boot[m][k][b] = mm[k]
    for m in MODELS:
        boot_all.update({f"{tg}|{m}|{k}": v for k, v in boot[m].items()})
        if m == "M0":
            continue
        for k in ("ibs", "auc10"):
            x = boot[m][k] - boot["M0"][k]
            res[m][f"d_{k}_vs_M0"] = dict(mean=float(x.mean()), q025=float(np.percentile(x, 2.5)),
                                          q975=float(np.percentile(x, 97.5)), n_boot=int(x.size))
    n_events = int(o_te_full[tg][1].sum())                 # over the whole follow-up
    out["targets"][tg] = dict(events=n_events, events_by_10=int((e_te & (t_te <= 10)).sum()), models=res)
    print(f"\n{tg}: {n_events} events in the held-out couples ({int((e_te & (t_te <= 10)).sum())} within 10 years)")
    print(f"{'model':16s} {'IBS':>7s} {'Brier10':>8s} {'AUC5':>6s} {'AUC10':>6s} {'C':>6s} {'calib.err':>9s}   dIBS vs M0 [95%]            dAUC10 vs M0 [95%]")
    for m in MODELS:
        r = res[m]
        line = (f"{m:16s} {r['ibs']:7.4f} {r['brier10']:8.4f} {r['auc5']:6.3f} {r['auc10']:6.3f} {r['c_uno']:6.3f} "
                f"{r['calibration']['mean_abs_error']:9.3f}")
        if m != "M0":
            a, b = r["d_ibs_vs_M0"], r["d_auc10_vs_M0"]
            line += f"   {a['mean']:+.4f} [{a['q025']:+.4f} ; {a['q975']:+.4f}]   {b['mean']:+.3f} [{b['q025']:+.3f} ; {b['q975']:+.3f}]"
        print(line)
json.dump(out, open(f"{RES}/evaluation.json", "w"), indent=2)
np.savez_compressed(f"{RES}/evaluation_boot.npz", **boot_all)
np.savez_compressed(f"{RES}/predictions_10y.npz", **{f"{m}|{tg}": pred[m][tg][:, :10] for m in MODELS for tg in TARGETS},
                    **{f"time|{tg}": o_te[tg][0] for tg in TARGETS}, **{f"event|{tg}": o_te[tg][1] for tg in TARGETS})
