"""summarize_seeds.py -- the evaluation repeated on five random 70/30 splits of the couples.

Reads results_eval/evaluation.json (main split) and results_seeds/seed_<k>/evaluation.json.
Writes results_seeds/splits_summary.json and results_seeds/RESULTATS_DECOUPAGES.md.
"""
import glob
import json
import os

import numpy as np
import arviz as az

TARGETS = ["husband", "wife", "joint_life", "last_survivor"]
LABEL = {"husband": "Mari", "wife": "Femme", "joint_life": "Premier décès", "last_survivor": "Second décès"}
dirs = {"principal": "results_eval"}
dirs.update({f"graine {k}": f"results_seeds/seed_{k}" for k in (1, 2, 3, 4) if os.path.exists(f"results_seeds/seed_{k}/evaluation.json")})
ev = {name: json.load(open(f"{path}/evaluation.json")) for name, path in dirs.items()}
models = [m for m in ev["principal"]["targets"]["husband"]["models"] if all(m in e["targets"]["husband"]["models"] for e in ev.values())]
ARVIZ1 = int(az.__version__.split(".")[0]) >= 1


def convergence(path, variant):
    parts = [np.load(f) for f in sorted(glob.glob(f"{path}/{variant}_train_*.npz"))]
    worst = 0.0
    for k in ["p0", "beta", "gamma", "omega", "kappa"]:
        if k in parts[0].files:
            x = np.concatenate([p[k] for p in parts], axis=0)                        # (chains, draws, ...)
            flat = x.reshape(x.shape[0], x.shape[1], -1)
            worst = max(worst, max(float(az.rhat(flat[:, :, j])) for j in range(flat.shape[2])))
    return dict(chains=int(sum(p["p0"].shape[0] for p in parts)), divergences=int(sum(p["diverging"].sum() for p in parts)),
                max_r_hat=worst)


out = {"splits": list(dirs), "models": models,
       "convergence": {name: {v: convergence(path, v) for v in ("M0", "M3")} for name, path in dirs.items()},
       "metrics": {}, "differences_vs_M0": {}}
lines = [f"# Évaluation répétée sur {len(dirs)} découpages 70/30", "",
         "Chaque découpage : modèles réajustés sur 3 550 couples, évalués sur les 1 522 autres. "
         "Modèles présents dans tous les découpages : " + ", ".join(models) + ".", "",
         "## Convergence des ajustements", "", "| Découpage | Modèle | Chaînes | Divergences | R̂ max |", "|---|---|---|---|---|"]
for name in dirs:
    for v in ("M0", "M3"):
        c = out["convergence"][name][v]
        lines.append(f"| {name} | {v} | {c['chains']} | {c['divergences']} | {c['max_r_hat']:.3f} |")

lines += ["", "## Métriques : moyenne sur les découpages [minimum ; maximum]", ""]
for tg in TARGETS:
    lines += [f"### {LABEL[tg]}", "", "| Modèle | IBS (10 ans) | AUC à 10 ans | Erreur de calibration |", "|---|---|---|---|"]
    out["metrics"][tg] = {}
    for m in models:
        row = {}
        for key, getter in (("ibs", lambda r: r["ibs"]), ("auc10", lambda r: r["auc10"]),
                            ("calibration", lambda r: r["calibration"]["mean_abs_error"])):
            x = np.array([getter(e["targets"][tg]["models"][m]) for e in ev.values()])
            row[key] = dict(mean=float(x.mean()), min=float(x.min()), max=float(x.max()), by_split=x.tolist())
        out["metrics"][tg][m] = row
        lines.append(f"| {m} | {row['ibs']['mean']:.4f} [{row['ibs']['min']:.4f} ; {row['ibs']['max']:.4f}] | "
                     f"{row['auc10']['mean']:.3f} [{row['auc10']['min']:.3f} ; {row['auc10']['max']:.3f}] | "
                     f"{row['calibration']['mean']:.3f} [{row['calibration']['min']:.3f} ; {row['calibration']['max']:.3f}] |")
    lines.append("")

lines += ["## Différences avec M0 (indépendance)", "",
          "Différence moyenne sur les découpages. Entre parenthèses : nombre de découpages où l'intervalle bootstrap à 95 % "
          "exclut 0, en faveur du modèle (+) ou de M0 (−). IBS : négatif = meilleur que M0. AUC : positif = meilleur que M0.", ""]
for tg in TARGETS:
    lines += [f"### {LABEL[tg]}", "", "| Modèle | Δ IBS | Δ AUC à 10 ans |", "|---|---|---|"]
    out["differences_vs_M0"][tg] = {}
    for m in models:
        if m == "M0":
            continue
        cell, row = [], {}
        for key, better_if_negative in (("d_ibs_vs_M0", True), ("d_auc10_vs_M0", False)):
            rs = [e["targets"][tg]["models"][m][key] for e in ev.values()]
            mean = float(np.mean([r["mean"] for r in rs]))
            neg = sum(r["q975"] < 0 for r in rs)
            pos = sum(r["q025"] > 0 for r in rs)
            wins, losses = (neg, pos) if better_if_negative else (pos, neg)
            row[key] = dict(mean=mean, by_split=[r["mean"] for r in rs], splits_better=int(wins), splits_worse=int(losses))
            cell.append(f"{mean:+.4f} (+{wins} / −{losses})")
        out["differences_vs_M0"][tg][m] = row
        lines.append(f"| {m} | {cell[0]} | {cell[1]} |")
    lines.append("")
json.dump(out, open("results_seeds/splits_summary.json", "w"), indent=2)
open("results_seeds/RESULTATS_DECOUPAGES.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
