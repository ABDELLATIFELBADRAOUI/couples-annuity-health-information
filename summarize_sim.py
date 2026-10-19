"""
summarize_sim.py -- summary of the replicated simulation study (sim_study.py).

For each scenario and each group of parameters: bias of the posterior mean, root mean squared error,
coverage of the 95 % credible interval, over the replications. For the dependence parameters (widowhood
effects omega, frailty variance kappa): the same quantities one by one, the share of replications whose
interval for omega excludes zero, and the link between the two sources of dependence (posterior
correlation within a replication, correlation of the estimation errors across replications). Convergence
of the fits (split R-hat over every parameter, effective sample sizes of omega and kappa) and accuracy of
the predicted 10-year probabilities are reported. The scenarios "base_ref" and "indep_ref" refit the same
data sets with the protocol of the real data: the two protocols are compared replication by replication.

Usage: python summarize_sim.py     -> results_sim/sim_summary.json, results_sim/RESULTATS_SIMULATION.md
"""
import glob
import json
import os

import numpy as np

from pipeline import OWN, CROSS
from sim_study import SCENARIOS, AGES_SHOWN

GROUPS = {"p0": "Mortalité de référence (4 âges × 2 sexes)", "beta": "Effets des caractéristiques propres (12)",
          "gamma": "Effets des caractéristiques du conjoint (4)", "omega": "Effet du veuvage (2)", "kappa": "Variance de la fragilité"}
LABEL = {"base": "Valeurs estimées", "indep": "Aucune dépendance", "frailty": "Fragilité 0,5, sans effet de veuvage",
         "widow": "Veuvage 1,5, sans fragilité", "both": "Fragilité 0,5 et veuvage 1,5", "small": "Valeurs estimées, 1 000 couples",
         "short": "Valeurs estimées, suivi limité à 6 ans", "base_ref": "Valeurs estimées, protocole de référence",
         "indep_ref": "Aucune dépendance, protocole de référence"}
TARGETS = ["husband", "wife", "joint_life", "last_survivor"]
fr = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")


def load(scenario):
    return {int(os.path.basename(f)[:-4].split("_")[-1]): np.load(f) for f in sorted(glob.glob(f"results_sim/{scenario}_[0-9][0-9][0-9].npz"))}


def group_stats(reps, k):
    true = np.array([r[f"{k}_true"] for r in reps], dtype=float).reshape(len(reps), -1)
    mean = np.array([r[f"{k}_mean"] for r in reps]).reshape(len(reps), -1)
    med = np.array([r[f"{k}_q50"] for r in reps]).reshape(len(reps), -1)
    lo = np.array([r[f"{k}_q025"] for r in reps]).reshape(len(reps), -1)
    hi = np.array([r[f"{k}_q975"] for r in reps]).reshape(len(reps), -1)
    sd = np.array([r[f"{k}_sd"] for r in reps]).reshape(len(reps), -1)
    err = mean - true
    inside = (lo <= true) & (true <= hi)
    interior = true[0] != 0 if k == "kappa" else np.ones(true.shape[1], dtype=bool)      # kappa = 0 is on the boundary: no coverage
    n_cov = int(inside[:, interior].size)
    cov = float(inside[:, interior].mean()) if interior.any() else None
    out = dict(components=int(true.shape[1]), bias=err.mean(axis=0).tolist(), rmse=np.sqrt((err ** 2).mean(axis=0)).tolist(),
               sd_of_estimates=(err.std(axis=0, ddof=1) if len(reps) > 1 else np.full(true.shape[1], np.nan)).tolist(),
               mean_posterior_sd=sd.mean(axis=0).tolist(), mean_posterior_median=med.mean(axis=0).tolist(), width=(hi - lo).mean(axis=0).tolist(),
               truth=true[0].tolist(), mean_abs_bias=float(np.abs(err.mean(axis=0)).mean()), max_abs_bias=float(np.abs(err.mean(axis=0)).max()),
               mean_rmse=float(np.sqrt((err ** 2).mean(axis=0)).mean()),
               coverage_by_component=np.where(interior, inside.mean(axis=0), np.nan).tolist(), coverage=cov,
               coverage_se=None if cov is None else float(np.sqrt(cov * (1 - cov) / n_cov)),             # as if the n_cov intervals were independent
               excludes_zero=((lo > 0) | (hi < 0)).mean(axis=0).tolist(), mean_q975=hi.mean(axis=0).tolist(),
               max_rhat=float(max(np.max(r[f"{k}_rhat"]) for r in reps)), min_ess=float(min(np.min(r[f"{k}_ess"]) for r in reps)))
    if k == "p0":                                                                         # relative errors: the levels differ by two orders of magnitude
        rel = err / true
        out.update(mean_abs_relative_bias=float(np.abs(rel.mean(axis=0)).mean()), max_abs_relative_bias=float(np.abs(rel.mean(axis=0)).max()))
    return out


if __name__ == "__main__":
    out, runs = {}, {}
    for sc in SCENARIOS:
        byrep = load(sc)
        if not byrep:
            continue
        runs[sc] = byrep
        reps = list(byrep.values())
        worst = np.array([max(max(float(np.max(r[f"{k}_rhat"])) for k in ("beta", "gamma", "omega", "kappa")), float(r["p0_rhat_max"])) for r in reps])
        s = dict(replications=len(reps), couples=int(reps[0]["n_couples"]), deaths_mean=np.mean([r["deaths"] for r in reps], axis=0).tolist(),
                 both_died_mean=float(np.mean([r["both_died"] for r in reps])), person_years_mean=float(np.mean([r["person_years"] for r in reps])),
                 divergences=int(sum(int(r["divergences"]) for r in reps)), replications_with_divergences=int(sum(int(r["divergences"]) > 0 for r in reps)),
                 seconds_mean=float(np.mean([r["seconds"] for r in reps])), chains=int(reps[0]["chains"]), warmup=int(reps[0]["warmup"]),
                 draws=int(reps[0]["draws"]), groups={k: group_stats(reps, k) for k in GROUPS}, max_rhat=float(worst.max()),
                 share_rhat_above_1_01=float(np.mean(worst > 1.01)), share_rhat_above_1_05=float(np.mean(worst > 1.05)))
        c = np.array([r["corr_omega_kappa"] for r in reps])
        s["posterior_corr_omega_kappa"] = c.mean(axis=0).tolist()
        e_om = np.array([r["omega_mean"] - r["omega_true"] for r in reps])
        e_ka = np.array([float(r["kappa_mean"]) - float(r["kappa_true"]) for r in reps])
        s["corr_errors_omega_kappa"] = [float(np.corrcoef(e_om[:, j], e_ka)[0, 1]) if len(reps) > 2 else None for j in range(2)]
        s["prediction"] = {tg: dict(bias=float(np.mean([r[f"pred_bias_{tg}"] for r in reps])), mae=float(np.mean([r[f"pred_mae_{tg}"] for r in reps])))
                           for tg in TARGETS}
        out[sc] = s
    # the two protocols on the same data sets
    protocol = {}
    for ref, short in (("base_ref", "base"), ("indep_ref", "indep")):
        common = sorted(set(runs.get(ref, {})) & set(runs.get(short, {})))
        if common:
            d = {}
            for k in ("omega", "kappa", "beta"):
                a = np.array([np.ravel(runs[ref][r][f"{k}_mean"]) for r in common])
                b = np.array([np.ravel(runs[short][r][f"{k}_mean"]) for r in common])
                sd = np.array([np.ravel(runs[ref][r][f"{k}_sd"]) for r in common])
                d[k] = dict(max_abs_difference=float(np.abs(a - b).max()), max_difference_in_posterior_sd=float((np.abs(a - b) / sd).max()))
            protocol[short] = dict(replications=len(common), **d)
    json.dump(dict(own=OWN, cross=CROSS, ages_shown=AGES_SHOWN.tolist(), scenarios=out, protocol_check=protocol),
              open("results_sim/sim_summary.json", "w"), indent=1)

    first = next(iter(out.values()))
    L = ["# Étude de simulation répétée du modèle de couples", "",
         "Des vies sont simulées avec des paramètres connus pour les couples d'estimation (âges et caractéristiques réels ; durée de suivi "
         "réelle, ou 11 ans, la fin administrative, quand les deux conjoints sont décédés dans les données réelles), puis le modèle complet "
         f"est réajusté avec les mêmes lois a priori que sur les données réelles ({first['chains']} chaînes, {first['warmup']} itérations "
         f"d'adaptation et {first['draws']} tirages chacune, sauf mention contraire). « Couverture » : part des intervalles de crédibilité à 95 % qui "
         "contiennent la vraie valeur, sur les réplications et les paramètres du groupe.", "",
         "## Convergence et taille", "", "| Scénario | Réplications | Chaînes × tirages | Couples | Décès (maris ; femmes ; deux) | Divergences | R̂ maximal | "
         "Réplications avec R̂ > 1,01 | > 1,05 | ESS minimal (veuvage ; fragilité) | Durée moyenne |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc, s in out.items():
        L.append(f"| {LABEL[sc]} | {s['replications']} | {s['chains']} × {s['draws']} | {s['couples']} | {s['deaths_mean'][0]:.0f} ; {s['deaths_mean'][1]:.0f} ; "
                 f"{s['both_died_mean']:.0f} | {s['divergences']} | {fr(s['max_rhat'])} | {fr(100 * s['share_rhat_above_1_01'], 0)} % | "
                 f"{fr(100 * s['share_rhat_above_1_05'], 0)} % | {s['groups']['omega']['min_ess']:.0f} ; {s['groups']['kappa']['min_ess']:.0f} | {s['seconds_mean']:.0f} s |")
    L += ["", "## Paramètres de dépendance", "",
          "| Scénario | Paramètre | Vraie valeur | Moyenne des estimations | Écart-type des estimations | Écart-type a posteriori moyen | "
          "Couverture | Intervalle excluant 0 |", "|---|---|---|---|---|---|---|---|"]
    for sc, s in out.items():
        for k, names in (("omega", ["veuvage, maris", "veuvage, femmes"]), ("kappa", ["variance de la fragilité"])):
            g = s["groups"][k]
            for j, name in enumerate(names):
                cov = g["coverage_by_component"][j]
                L.append(f"| {LABEL[sc]} | {name} | {fr(g['truth'][j])} | {fr(g['truth'][j] + g['bias'][j])} | {fr(g['sd_of_estimates'][j])} | "
                         f"{fr(g['mean_posterior_sd'][j])} | {'—' if cov is None or np.isnan(cov) else fr(100 * cov, 0) + ' %'} | "
                         f"{fr(100 * g['excludes_zero'][j], 0) + ' %' if k == 'omega' else '—'} |")
    L += ["", "Quand la vraie variance de la fragilité est nulle, elle est au bord du domaine : la couverture n'a pas de sens ; la moyenne "
          "des estimations indique alors ce que le modèle attribue à tort à la fragilité (la loi a priori demi-normale pousse l'estimation "
          "au-dessus de zéro).", "",
          "## Tous les groupes de paramètres", "",
          "| Scénario | Groupe | Biais absolu moyen | Biais absolu maximal | Erreur quadratique moyenne (racine) | Couverture (erreur standard) |",
          "|---|---|---|---|---|---|"]
    for sc, s in out.items():
        for k, label in GROUPS.items():
            g = s["groups"][k]
            cov = "—" if g["coverage"] is None else f"{fr(100 * g['coverage'], 1)} % ({fr(100 * g['coverage_se'], 1)})"
            if k == "p0":
                L.append(f"| {LABEL[sc]} | {label} | {fr(100 * g['mean_abs_relative_bias'], 1)} % (relatif) | {fr(100 * g['max_abs_relative_bias'], 1)} % (relatif) | — | {cov} |")
            else:
                L.append(f"| {LABEL[sc]} | {label} | {fr(g['mean_abs_bias'])} | {fr(g['max_abs_bias'])} | {fr(g['mean_rmse'])} | {cov} |")
    L += ["", "L'erreur standard de la couverture suppose les intervalles indépendants : elle est optimiste, les paramètres d'une même réplication étant liés.", "",
          "## Séparation du veuvage et de la fragilité", "",
          "| Scénario | Corrélation a posteriori (veuvage maris, fragilité) | (veuvage femmes, fragilité) | Corrélation des erreurs entre réplications "
          "(maris) | (femmes) |", "|---|---|---|---|---|"]
    for sc, s in out.items():
        ce = s["corr_errors_omega_kappa"]
        L.append(f"| {LABEL[sc]} | {fr(s['posterior_corr_omega_kappa'][0], 2)} | {fr(s['posterior_corr_omega_kappa'][1], 2)} | "
                 f"{'—' if ce[0] is None or np.isnan(ce[0]) else fr(ce[0], 2)} | {'—' if ce[1] is None or np.isnan(ce[1]) else fr(ce[1], 2)} |")
    L += ["", "## Probabilités de survie à 10 ans prédites (300 couples), écart à la vérité en points de pourcentage", "",
          "| Scénario | Mari : biais ; erreur absolue | Femme | Premier décès | Second décès |", "|---|---|---|---|---|"]
    for sc, s in out.items():
        L.append(f"| {LABEL[sc]} | " + " | ".join(f"{fr(100 * s['prediction'][tg]['bias'], 2)} ; {fr(100 * s['prediction'][tg]['mae'], 2)}" for tg in TARGETS) + " |")
    if protocol:
        L += ["", "## Protocole court contre protocole de référence, mêmes jeux de données", "",
              "Plus grand écart entre les moyennes a posteriori des deux protocoles, en valeur et en écarts-types a posteriori.", "",
              "| Scénario | Réplications | Veuvage | Fragilité | Effets des caractéristiques propres |", "|---|---|---|---|---|"]
        for sc, p in protocol.items():
            L.append(f"| {LABEL[sc]} | {p['replications']} | " + " | ".join(
                f"{fr(p[k]['max_abs_difference'])} ({fr(p[k]['max_difference_in_posterior_sd'], 2)} é.-t.)" for k in ("omega", "kappa", "beta")) + " |")
    open("results_sim/RESULTATS_SIMULATION.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
