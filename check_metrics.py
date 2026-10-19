"""check_metrics.py -- the metrics written in metrics_util.py against scikit-survival.

On the held-out couples of the main split, for every model and target: integrated Brier score, Brier
score at 10 years and AUC at 5 and 10 years computed by metrics_util.metrics_own and by scikit-survival.
Then the same on bootstrap resamples that scikit-survival accepts, and the number of resamples it refuses.
Output: results_eval/check_metrics.txt
"""
import glob

import numpy as np
from sksurv.util import Surv

from metrics_util import truncate, metrics, metrics_own, censoring_survival
from pipeline import load_couples, split_indices, fit_scaler, make_data, prior_arrays, age_max_of, outcomes, predict_states, load_draws

L = []


def say(msg):
    print(msg, flush=True)
    L.append(msg)


d = load_couples("cohort_w10.csv")
tr, te = split_indices(d)
dtr, dte = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)
sc = fit_scaler(dtr)
data_te = make_data(dte, sc)
ages, pstar, _ = prior_arrays("prior_table.csv", age_max_of(d))
o_tr = {k: truncate(v) for k, v in outcomes(dtr).items()}
o_te = {k: truncate(v) for k, v in outcomes(dte).items()}
pred = {v: predict_states(load_draws(sorted(glob.glob(f"results_eval/{v}_train_*.npz")), n_draws=100), data_te, int(ages[0]), 12) for v in ("M0", "M3")}
z = np.load("results_eval/deep_predictions.npz")
for key in z.files:
    if "|" not in key:                       # bookkeeping entries (e.g. split_seed), not predictions
        continue
    name, tg = key.split("|")
    pred.setdefault(name, {})[tg] = z[key]
worst = 0.0
for tg in ("husband", "wife", "joint_life", "last_survivor"):
    y_train = Surv.from_arrays(o_tr[tg][1], o_tr[tg][0])
    t, e = o_te[tg]
    G = censoring_survival(*o_tr[tg])
    for m in pred:
        a, b = metrics(y_train, Surv.from_arrays(e, t), pred[m][tg]), metrics_own(G, t, e, pred[m][tg])
        worst = max(worst, max(abs(a[k] - b[k]) for k in b))
say(f"1. Échantillon complet, {len(pred)} modèles x 4 cibles x 4 mesures : écart maximal avec scikit-survival {worst:.1e}")
rng = np.random.default_rng(2026)
for tg in ("joint_life", "last_survivor"):
    y_train = Surv.from_arrays(o_tr[tg][1], o_tr[tg][0])
    t, e = o_te[tg]
    G = censoring_survival(*o_tr[tg])
    refused, worst, n = 0, 0.0, 200
    for _ in range(n):
        idx = rng.integers(0, len(t), len(t))
        b = metrics_own(G, t[idx], e[idx], pred["M3"][tg][idx])
        try:
            a = metrics(y_train, Surv.from_arrays(e[idx], t[idx]), pred["M3"][tg][idx])
            worst = max(worst, max(abs(a[k] - b[k]) for k in b))
        except ValueError:
            refused += 1
    say(f"2. {tg}, {n} rééchantillonnages : {refused} refusés par scikit-survival ; écart maximal sur les autres {worst:.1e} ; "
        f"plus petit temps observé dans l'échantillon de test {t.min():g}")
open("results_eval/check_metrics.txt", "w", encoding="utf-8").write("\n".join(L) + "\n")
