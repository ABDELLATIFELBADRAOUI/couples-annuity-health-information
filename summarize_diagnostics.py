"""
summarize_diagnostics.py -- convergence diagnostics of every fit of the couples model on the main split
(rank-normalised split R-hat and effective sample sizes of Vehtari et al. 2021, through ArviZ).

Usage: python summarize_diagnostics.py    -> results_eval/diagnostics.json, results_eval/DIAGNOSTICS.md
"""
import glob
import json

import numpy as np
import arviz as az

FITS = {"M0": "M0_train", "M1": "M1_train", "M2": "M2_train", "M3": "M3_train", "MD": "MD_train", "M3 (tous les couples)": "M3_full"}
ARVIZ1 = int(az.__version__.split(".")[0]) >= 1
tail = lambda x: float(az.ess(x, method="tail", prob=(0.05, 0.95))) if ARVIZ1 else float(az.ess(x, method="tail"))

if __name__ == "__main__":
    out = {}
    for label, stem in FITS.items():
        parts = [np.load(f) for f in sorted(glob.glob(f"results_eval/{stem}_*.npz"))]
        if not parts:
            continue
        res = dict(couples=int(parts[0]["n_couples"]), chains=int(sum(p["p0"].shape[0] for p in parts)), draws=int(parts[0]["p0"].shape[1]),
                   divergences=int(sum(p["diverging"].sum() for p in parts)), parameters=0, groups={},
                   mean_leapfrog_steps=float(np.mean([p["num_steps"].mean() for p in parts])),
                   max_leapfrog_steps=int(max(p["num_steps"].max() for p in parts)), minutes=float(sum(float(p["elapsed"]) for p in parts)) / 60)
        for k in ("p0", "beta", "gamma", "omega", "kappa"):
            if k not in parts[0].files:
                continue
            x = np.concatenate([p[k] for p in parts], axis=0)
            flat = x.reshape(x.shape[0], x.shape[1], -1)
            rh = [float(az.rhat(flat[:, :, j])) for j in range(flat.shape[2])]
            eb = [float(az.ess(flat[:, :, j], method="bulk")) for j in range(flat.shape[2])]
            et = [tail(flat[:, :, j]) for j in range(flat.shape[2])]
            res["groups"][k] = dict(n=len(rh), max_rhat=max(rh), min_ess_bulk=min(eb), min_ess_tail=min(et))
            res["parameters"] += len(rh)
        res.update(max_rhat=max(g["max_rhat"] for g in res["groups"].values()), min_ess_bulk=min(g["min_ess_bulk"] for g in res["groups"].values()),
                   min_ess_tail=min(g["min_ess_tail"] for g in res["groups"].values()))
        out[label] = res
    json.dump(out, open("results_eval/diagnostics.json", "w"), indent=1)
    L = ["# Diagnostics de convergence des ajustements (découpage principal)", "",
         "| Modèle | Couples | Chaînes × tirages | Paramètres | Divergences | R̂ maximal | ESS minimal (bulk) | ESS minimal (tail) | Pas de saute-mouton (moyenne ; max) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for label, r in out.items():
        L.append(f"| {label} | {r['couples']} | {r['chains']} × {r['draws']} | {r['parameters']} | {r['divergences']} | {r['max_rhat']:.3f} | "
                 f"{r['min_ess_bulk']:.0f} | {r['min_ess_tail']:.0f} | {r['mean_leapfrog_steps']:.0f} ; {r['max_leapfrog_steps']} |")
    open("results_eval/DIAGNOSTICS.md", "w", encoding="utf-8").write("\n".join(L).replace(".", ",") + "\n")
    print("\n".join(L))
