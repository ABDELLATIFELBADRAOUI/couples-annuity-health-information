"""
summarize_sens.py -- the full model under other prior choices (fits of fit_sens.py, results_sens/).

For each setting: convergence, the dependence parameters and two health effects, the distance of the
baseline mortality and of the predicted 10-year probabilities from the reference fit, and the
out-of-sample accuracy on the held-out couples (integrated Brier score and AUC at 10 years).

Usage: python summarize_sens.py      -> results_sens/sens_summary.json, results_sens/RESULTATS_LOIS_A_PRIORI.md
"""
import json
import os

import numpy as np
import arviz as az
from sksurv.util import Surv

from fit_sens import SETTINGS
from metrics_util import truncate, metrics
from pipeline import load_couples, split_indices, fit_scaler, make_data, prior_arrays, age_max_of, outcomes, predict_states, load_draws

TARGETS = ["husband", "wife", "joint_life", "last_survivor"]
LABEL = {"base": "Référence", "c_quarter": "Force des urnes divisée par 4", "c_times4": "Force des urnes multipliée par 4",
         "centre_low": "Mortalité a priori × 0,7", "centre_high": "Mortalité a priori × 1,3",
         "scale05": "Effets : Normale(0 ; 0,5)", "scale2": "Effets : Normale(0 ; 2)", "student": "Effets : Student à 3 degrés de liberté",
         "kappa05": "Fragilité : demi-normale(0,5)", "kappa2": "Fragilité : demi-normale(2)"}
N_DRAWS = 400
fr = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")


def interval(x, transform=lambda v: v):
    flat = np.asarray(x).reshape(-1)
    return dict(mean=float(transform(flat).mean()), median=float(np.median(transform(flat))),
                q025=float(transform(np.percentile(flat, 2.5))), q975=float(transform(np.percentile(flat, 97.5))))


if __name__ == "__main__":
    d = load_couples("cohort_w10.csv")
    tr, te = split_indices(d)
    dtr, dte = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)
    sc = fit_scaler(dtr)
    data_te = make_data(dte, sc)
    ages, _, _ = prior_arrays("prior_table.csv", age_max_of(d))
    o_tr = {k: truncate(v) for k, v in outcomes(dtr).items()}
    o_te = {k: truncate(v) for k, v in outcomes(dte).items()}
    shown = np.array([60, 70, 80, 90]) - int(ages[0])
    out, pred, p0_mean = {}, {}, {}
    for name in SETTINGS:
        path = f"results_sens/{name}.npz"
        if not os.path.exists(path):
            continue
        z = np.load(path)
        worst_rhat, min_ess = 0.0, np.inf
        for k in ("p0", "beta", "gamma", "omega", "kappa"):
            flat = z[k].reshape(z[k].shape[0], z[k].shape[1], -1)
            for j in range(flat.shape[2]):
                worst_rhat = max(worst_rhat, float(az.rhat(flat[:, :, j])))
                min_ess = min(min_ess, float(az.ess(flat[:, :, j], method="bulk")))
        pred[name] = predict_states(load_draws([path], n_draws=N_DRAWS), data_te, int(ages[0]), 10)
        p0_mean[name] = z["p0"].reshape((-1,) + z["p0"].shape[2:]).mean(axis=0)
        res = dict(setting=SETTINGS[name], chains=int(z["kappa"].shape[0]), draws=int(z["kappa"].shape[1]), divergences=int(z["diverging"].sum()),
                   max_rhat=worst_rhat, min_ess_bulk=min_ess, minutes=float(z["elapsed"]) / 60,
                   widowhood_hr=[interval(z["omega"][:, :, s], np.exp) for s in range(2)], kappa=interval(z["kappa"]),
                   smoker_hr=[interval(z["beta"][:, :, s, 1], np.exp) for s in range(2)],
                   wealth_hr=[interval(z["beta"][:, :, s, 5], np.exp) for s in range(2)],
                   p0_shown=p0_mean[name][:, shown].tolist(), metrics={})
        for tg in TARGETS:
            m = metrics(Surv.from_arrays(o_tr[tg][1], o_tr[tg][0]), Surv.from_arrays(o_te[tg][1], o_te[tg][0]), pred[name][tg])
            res["metrics"][tg] = dict(ibs=m["ibs"], auc10=m["auc10"])
        out[name] = res
    if "base" in out:
        for name, res in out.items():
            res["p0_relative_difference"] = float(np.abs(p0_mean[name] / p0_mean["base"] - 1.0)[:, : 91 - int(ages[0])].mean())     # ages 50 to 90
            res["survival10_abs_difference"] = {tg: dict(mean=float(np.abs(pred[name][tg][:, 9] - pred["base"][tg][:, 9]).mean()),
                                                         max=float(np.abs(pred[name][tg][:, 9] - pred["base"][tg][:, 9]).max())) for tg in TARGETS}
    # the setting "base" repeats the first two chains of the reference fit (same model, data and seed): the draws must coincide
    same = None
    if "base" in out and os.path.exists("results_eval/M3_train_a.npz"):
        a, b = np.load("results_sens/base.npz"), np.load("results_eval/M3_train_a.npz")
        same = float(max(np.abs(a[k] - b[k]).max() for k in ("p0", "beta", "gamma", "omega", "kappa")))
    json.dump(dict(ages_shown=[60, 70, 80, 90], base_against_reference_fit_max_abs_difference=same, settings=out),
              open("results_sens/sens_summary.json", "w"), indent=1)

    hr = lambda r: f"{fr(r['median'], 2)} [{fr(r['q025'], 2)} ; {fr(r['q975'], 2)}]"
    L = ["# Sensibilité du modèle complet aux lois a priori", "",
         "Modèle complet réajusté sur les 3 550 couples d'estimation, un choix modifié à la fois (2 chaînes, 1 000 + 1 000 itérations). "
         "Rapports de risque : médiane a posteriori [intervalle à 95 %] ; variance de la fragilité : moyenne a posteriori [intervalle à 95 %].", "",
         "| Choix | Divergences | R̂ max | ESS min | Veuvage, maris | Veuvage, femmes | Variance de la fragilité | Fumeur, maris | Fumeuse, femmes |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, r in out.items():
        k = r["kappa"]
        L.append(f"| {LABEL[name]} | {r['divergences']} | {fr(r['max_rhat'])} | {r['min_ess_bulk']:.0f} | {hr(r['widowhood_hr'][0])} | {hr(r['widowhood_hr'][1])} | "
                 f"{fr(k['mean'])} [{fr(k['q025'])} ; {fr(k['q975'])}] | {hr(r['smoker_hr'][0])} | {hr(r['smoker_hr'][1])} |")
    L += ["", "Écart à l'ajustement de référence et précision hors échantillon (1 522 couples mis de côté) :", "",
          "| Choix | Mortalité de référence : écart relatif moyen (50-90 ans) | Survie à 10 ans, premier décès : écart absolu moyen ; maximal | "
          "Second décès : moyen ; maximal | IBS mari | IBS femme | IBS premier décès | IBS second décès | AUC 10 ans premier décès |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, r in out.items():
        dj, dl = r.get("survival10_abs_difference", {}).get("joint_life"), r.get("survival10_abs_difference", {}).get("last_survivor")
        m = r["metrics"]
        L.append(f"| {LABEL[name]} | {fr(100 * r.get('p0_relative_difference', float('nan')), 1)} % | "
                 + (f"{fr(100 * dj['mean'], 2)} ; {fr(100 * dj['max'], 2)} pt | {fr(100 * dl['mean'], 2)} ; {fr(100 * dl['max'], 2)} pt | " if dj else "— | — | ")
                 + f"{fr(m['husband']['ibs'], 4)} | {fr(m['wife']['ibs'], 4)} | {fr(m['joint_life']['ibs'], 4)} | {fr(m['last_survivor']['ibs'], 4)} | "
                 f"{fr(m['joint_life']['auc10'])} |")
    if same is not None:
        L += ["", f"Contrôle : le choix « Référence » refait les deux premières chaînes de l'ajustement de référence (même modèle, mêmes données, "
              f"même graine) ; plus grand écart entre les tirages : {same:.1e}."]
    open("results_sens/RESULTATS_LOIS_A_PRIORI.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
