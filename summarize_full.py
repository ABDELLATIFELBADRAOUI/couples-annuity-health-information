"""summarize_full.py -- posterior summary of the full model (M3) fitted on all couples, with
education and wealth among the covariates (chains: results_eval/M3_full_*.npz)."""
import glob
import json

import numpy as np
import arviz as az

ARVIZ1 = int(az.__version__.split(".")[0]) >= 1


def ess_tail(x):
    return float(az.ess(x, method="tail", prob=(0.05, 0.95))) if ARVIZ1 else float(az.ess(x, method="tail"))


parts = [np.load(f) for f in sorted(glob.glob("results_eval/M3_full_*.npz"))]
post = {k: np.concatenate([p[k] for p in parts], axis=0) for k in ["p0", "beta", "gamma", "omega", "kappa"]}
n_chains, n_draws = post["kappa"].shape
n_div = int(sum(p["diverging"].sum() for p in parts))

rows = []
def add(label, draws):
    flat = draws.reshape(-1)
    lo, hi = np.percentile(flat, [2.5, 97.5])
    rows.append(dict(parameter=label, hazard_ratio=float(np.exp(np.median(flat))), hr_q025=float(np.exp(lo)),
                     hr_q975=float(np.exp(hi)), prob_positive=float((flat > 0).mean()), r_hat=float(az.rhat(draws)),
                     ess_bulk=float(az.ess(draws, method="bulk")), ess_tail=ess_tail(draws)))

own = ["BMI (+1 écart-type)", "fumeur", "hypertension", "diabète", "éducation (+1 écart-type)", "patrimoine (+1 écart-type)"]
cross = ["BMI du conjoint (+1 écart-type)", "conjoint fumeur"]
for s, sex in enumerate(["mari", "femme"]):
    for j, nm in enumerate(own):
        add(f"{sex} : {nm}", post["beta"][:, :, s, j])
    for j, nm in enumerate(cross):
        add(f"{sex} : {nm}", post["gamma"][:, :, s, j])
    add(f"{sex} : veuvage", post["omega"][:, :, s])
k = post["kappa"]
kf = k.reshape(-1)
kappa = dict(mean=float(kf.mean()), q025=float(np.percentile(kf, 2.5)), q975=float(np.percentile(kf, 97.5)),
             r_hat=float(az.rhat(k)), ess_bulk=float(az.ess(k, method="bulk")), ess_tail=ess_tail(k))
idata = az.from_dict({"posterior": {"p0": post["p0"]}}) if ARVIZ1 else az.from_dict(posterior={"p0": post["p0"]})
base = dict(max_r_hat=float(np.asarray(az.rhat(idata, var_names=["p0"])["p0"]).max()),
            min_ess_bulk=float(np.asarray(az.ess(idata, var_names=["p0"], method="bulk")["p0"]).min()))
out = dict(setting=dict(couples=int(parts[0]["n_couples"]), chains=int(n_chains), draws_per_chain=int(n_draws),
                        warmup=1000, divergences=n_div,
                        minutes_per_process=[round(float(p["elapsed"]) / 60, 1) for p in parts]),
           effects=rows, kappa=kappa, baseline=base)
json.dump(out, open("results_eval/full_model_summary.json", "w"), indent=2, ensure_ascii=False)

L = ["| Effet | Rapport de risque (médiane) | IC 95 % | P(effet > 0) | R̂ | ESS bulk / tail |", "|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['parameter']} | {r['hazard_ratio']:.2f} | [{r['hr_q025']:.2f} ; {r['hr_q975']:.2f}] | "
             f"{r['prob_positive']:.3f} | {r['r_hat']:.3f} | {r['ess_bulk']:.0f} / {r['ess_tail']:.0f} |")
L += ["", f"Variance de la fragilité partagée κ : {kappa['mean']:.3f} [{kappa['q025']:.3f} ; {kappa['q975']:.3f}], "
          f"R̂ = {kappa['r_hat']:.3f}, ESS = {kappa['ess_bulk']:.0f} / {kappa['ess_tail']:.0f}.",
      f"Hasard de base : max R̂ = {base['max_r_hat']:.3f}, min ESS = {base['min_ess_bulk']:.0f}.",
      f"{out['setting']['couples']} couples ; {n_chains} chaînes × {n_draws} tirages après 1000 de chauffe ; divergences : {n_div}."]
open("results_eval/full_model_summary.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
