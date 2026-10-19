"""summarize_real.py -- posterior summary of the fit on the real cohort (chains from fit_real.py)."""
import glob
import json

import numpy as np
import arviz as az

from fit_real import load
from dyadic_mortality import build_arrays

ARVIZ1 = int(az.__version__.split(".")[0]) >= 1


def ess_tail(x):
    return float(az.ess(x, method="tail", prob=(0.05, 0.95))) if ARVIZ1 else float(az.ess(x, method="tail"))


files = sorted(glob.glob("results_real/chains_*.npz"))
parts = [np.load(f) for f in files]
post = {k: np.concatenate([p[k] for p in parts], axis=0) for k in ["p0", "beta", "gamma", "omega", "kappa"]}
n_chains, n_draws = post["kappa"].shape
n_div = int(sum(p["diverging"].sum() for p in parts))
steps = np.concatenate([p["num_steps"].ravel() for p in parts])
ages, pstar, c = parts[0]["ages"], parts[0]["pstar"], parts[0]["c"]

rows = []
def add(label, draws):                                   # draws: (chains, draws), log-hazard-ratio scale
    flat = draws.reshape(-1)
    lo, hi = np.percentile(flat, [2.5, 97.5])
    rows.append(dict(parameter=label, mean=float(flat.mean()), sd=float(flat.std()), q025=float(lo), q975=float(hi),
                     hazard_ratio=float(np.exp(np.median(flat))), hr_q025=float(np.exp(lo)), hr_q975=float(np.exp(hi)),
                     prob_positive=float((flat > 0).mean()), r_hat=float(az.rhat(draws)),
                     ess_bulk=float(az.ess(draws, method="bulk")), ess_tail=ess_tail(draws)))

own = ["BMI (+1 écart-type)", "fumeur", "hypertension", "diabète"]
cross = ["BMI du conjoint (+1 écart-type)", "conjoint fumeur"]
for s, sex in enumerate(["mari", "femme"]):
    for j, nm in enumerate(own):
        add(f"{sex} : {nm}", post["beta"][:, :, s, j])
    for j, nm in enumerate(cross):
        add(f"{sex} : {nm}", post["gamma"][:, :, s, j])
    add(f"{sex} : veuvage", post["omega"][:, :, s])

k = post["kappa"]
kflat = k.reshape(-1)
kappa = dict(mean=float(kflat.mean()), q025=float(np.percentile(kflat, 2.5)), q975=float(np.percentile(kflat, 97.5)),
             r_hat=float(az.rhat(k)), ess_bulk=float(az.ess(k, method="bulk")), ess_tail=ess_tail(k),
             kendall_tau_mean=float((kflat / (kflat + 2)).mean()))

# baseline: prior guess versus posterior, with the cohort's person-years and deaths at each age
data, ages2, _, _, _ = load("cohort_w10.csv", "prior_table.csv")
arr = build_arrays(data, int(ages[0]), int(ages[-1]))
at_risk, deaths = np.zeros_like(pstar), np.zeros_like(pstar)
for s in range(2):
    idx = arr["age_idx"][:, s, :]
    np.add.at(at_risk[s], idx[(arr["surv_mask"] + arr["event_mask"])[:, s, :] > 0], 1.0)
    np.add.at(deaths[s], idx[arr["event_mask"][:, s, :] > 0], 1.0)
p = post["p0"].reshape(-1, *post["p0"].shape[2:])
plo, phi = np.percentile(p, [2.5, 97.5], axis=0)
idata = az.from_dict({"posterior": {"p0": post["p0"]}}) if ARVIZ1 else az.from_dict(posterior={"p0": post["p0"]})
rhat_p0 = np.asarray(az.rhat(idata, var_names=["p0"])["p0"])
ess_p0 = np.asarray(az.ess(idata, var_names=["p0"], method="bulk")["p0"])
baseline = []
for s, sex in enumerate(["hommes", "femmes"]):
    for age in range(55, 96, 10):
        j = age - int(ages[0])
        baseline.append(dict(sex=sex, age=age, at_risk=int(at_risk[s, j]), deaths=int(deaths[s, j]), prior=float(pstar[s, j]),
                             prior_strength=float(c[s, j]), posterior_mean=float(p[:, s, j].mean()),
                             q025=float(plo[s, j]), q975=float(phi[s, j])))

out = dict(setting=dict(couples=int(parts[0]["n_couples"]), ages=[int(ages[0]), int(ages[-1])], chains=int(n_chains),
                        warmup=1000, draws_per_chain=int(n_draws), divergences=n_div,
                        leapfrog_steps_mean=float(steps.mean()), leapfrog_steps_max=int(steps.max()),
                        minutes_per_process=[round(float(p_["elapsed"]) / 60, 1) for p_ in parts]),
           effects=rows, kappa=kappa,
           baseline=dict(table=baseline, max_r_hat=float(rhat_p0.max()), min_ess_bulk=float(ess_p0.min())))
json.dump(out, open("results_real/posterior_summary.json", "w"), indent=2, ensure_ascii=False)

L = ["| Effet | Rapport de risque (médiane) | IC 95 % | P(effet > 0) | R̂ | ESS bulk / tail |", "|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['parameter']} | {r['hazard_ratio']:.2f} | [{r['hr_q025']:.2f} ; {r['hr_q975']:.2f}] | "
             f"{r['prob_positive']:.3f} | {r['r_hat']:.3f} | {r['ess_bulk']:.0f} / {r['ess_tail']:.0f} |")
L += ["", f"Variance de la fragilité partagée κ : {kappa['mean']:.2f} [{kappa['q025']:.2f} ; {kappa['q975']:.2f}], "
          f"R̂ = {kappa['r_hat']:.3f}, ESS = {kappa['ess_bulk']:.0f} / {kappa['ess_tail']:.0f} ; "
          f"tau de Kendall impliqué par la fragilité seule : {kappa['kendall_tau_mean']:.2f}.", "",
      "| Sexe | Âge | Personnes-années | Décès | Prior p* (poids c) | Postérieur (moyenne) | IC 95 % |", "|---|---|---|---|---|---|---|"]
for b in baseline:
    L.append(f"| {b['sex']} | {b['age']} | {b['at_risk']} | {b['deaths']} | {b['prior']:.4f} ({b['prior_strength']:.0f}) | "
             f"{b['posterior_mean']:.4f} | [{b['q025']:.4f} ; {b['q975']:.4f}] |")
s_ = out["setting"]
L += ["", f"Hasard de base : max R̂ = {out['baseline']['max_r_hat']:.3f}, min ESS = {out['baseline']['min_ess_bulk']:.0f}.",
      f"{s_['couples']} couples, âges {s_['ages'][0]}-{s_['ages'][1]} ; {s_['chains']} chaînes × {s_['draws_per_chain']} tirages après "
      f"{s_['warmup']} de chauffe ; divergences : {n_div} ; pas de leapfrog : {s_['leapfrog_steps_mean']:.0f} en moyenne, "
      f"{s_['leapfrog_steps_max']} au maximum ; durée : {s_['minutes_per_process']} min par processus."]
open("results_real/posterior_summary.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
