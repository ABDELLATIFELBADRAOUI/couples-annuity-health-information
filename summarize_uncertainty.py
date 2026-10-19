"""
summarize_uncertainty.py -- losses of the decision analysis when the truth is a posterior draw of the
mortality model (results of run_uncertainty.py).

Usage: python summarize_uncertainty.py   -> results_decision/uncertainty_summary.json and .md
"""
import json

import numpy as np

LABEL = {"PP": "Connaît le modèle complet estimé, pas ses vrais paramètres (loi prédictive)", "MARG": "Ignore la dépendance seule (MARG)",
         "M0": "Santé sans dépendance (M0)", "MD": "Dépendance sans santé (MD)", "POP": "Âge et sexe seulement (POP)"}
# Every household keeps the decisions and the consumption policy of the main analysis; none knows the draw.
fr = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")

if __name__ == "__main__":
    z = np.load("results_decision/uncertainty.npz")
    index = z["index"]
    base = np.load("results_decision/base_limit.npz")
    from decision_dp import Params, wealth_equivalent_loss
    par = Params()
    main = {m: base[f"true|{m}"] for m in ("MARG", "M0", "MD", "POP")}
    out = {"draws": int(z["draws"].size), "couples": int(index.size), "seconds": float(z["seconds"]), "beliefs": {}}
    means = {}
    for m in LABEL:
        x = z[f"loss|{m}"]                                    # (draws, couples), in %
        means[m] = x.mean(axis=1)
        r = dict(mean=float(means[m].mean()), sd=float(means[m].std(ddof=1)), min=float(means[m].min()), max=float(means[m].max()))
        if m in main:                                         # the main analysis (truth: posterior predictive law), same couples
            r["main_analysis_same_couples"] = float(100 * wealth_equivalent_loss(main[m][index], base["true|TRUE"][index], par).mean())
        out["beliefs"][m] = r
    order = ["POP", "MD", "M0", "MARG"]
    out["share_of_draws"] = {f"{a}>{b}": float(np.mean(means[a] > means[b])) for i, a in enumerate(order) for b in order[i + 1:]}
    ratio = means["POP"] / means["M0"]
    out["ratio_POP_to_M0"] = dict(mean=float(ratio.mean()), min=float(ratio.min()), max=float(ratio.max()))
    json.dump(out, open("results_decision/uncertainty_summary.json", "w"), indent=1)
    L = ["# Incertitude du modèle de mortalité dans l'analyse de décision", "",
         f"La vérité est tour à tour chacun de {out['draws']} tirages a posteriori des paramètres du modèle complet ; {out['couples']} couples mis de côté, "
         "cas de base. Chaque ménage garde la décision et la politique de consommation de l'analyse principale : aucun ne connaît le tirage. "
         "Perte moyenne en % des ressources initiales : moyenne sur les tirages, écart-type entre tirages, plus petite et plus grande valeur.", "",
         "| Ménage | Perte moyenne | Écart-type | Minimum ; maximum | Analyse principale, mêmes couples |", "|---|---|---|---|---|"]
    for m, r in out["beliefs"].items():
        L.append(f"| {LABEL[m]} | {fr(r['mean'])} | {fr(r['sd'])} | {fr(r['min'])} ; {fr(r['max'])} | "
                 f"{fr(r['main_analysis_same_couples']) if 'main_analysis_same_couples' in r else '—'} |")
    L += ["", "Part des tirages où la perte du premier ménage dépasse celle du second : "
          + " ; ".join(f"{k.replace('>', ' > ')} : {fr(100 * v, 0)} %" for k, v in out["share_of_draws"].items()) + ".", "",
          f"Rapport perte POP / perte M0 : {fr(out['ratio_POP_to_M0']['mean'], 1)} "
          f"(de {fr(out['ratio_POP_to_M0']['min'], 1)} à {fr(out['ratio_POP_to_M0']['max'], 1)})."]
    open("results_decision/RESULTATS_INCERTITUDE.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
