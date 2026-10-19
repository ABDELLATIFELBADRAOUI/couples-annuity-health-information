"""
evaluate_agents.py -- exact gap between the learned policies and the optimum, on the held-out couples.

For every agent saved in the results folder (weights of rl_apg.py and rl_ppo.py):
    - the annuity choice of the agent for each held-out couple;
    - the exact value of its policy under the full model (followed along every history);
    - the gap to the optimum of dynamic programming, as a share of the initial resources;
    - the part of the gap due to the annuity choice alone (the agent's choice with the best consumption).
Reference lines on the same couples: households that decide with a simpler mortality model, and three
simple rules. Everything uses the laws with the limiting age of 110 (results_decision/base_limit.npz).
The held-out couples are not used during training: the agents are followed on training couples.

Usage: python evaluate_agents.py          (folder: results_rl, or the environment variable CARUP_RL_OUT)
The environment variable CARUP_AGENTS (comma-separated names, e.g. apg_seed0,ppo_seed0) restricts the
evaluation to these agents; by default every agent of the folder is evaluated.
Writes <folder>/agents_summary.json, <folder>/agents_per_couple.npz and <folder>/RESULTATS_AGENTS.md.
"""
import glob
import json
import os

import numpy as np
import jax

from decision_setup import law, held_out, horizon
from decision_dp import Params, wealth_equivalent_loss, value_at_actions
from rl_core import build, init_policy, load_params, make_exact_evaluator, MENU, AGE_LIMIT, T_MAX, RULE

OUT = os.environ.get("CARUP_RL_OUT", "results_rl")
KIND = {"apg": "Gradient à travers le simulateur", "ppoa": "PPO, bruit décroissant", "ppo": "PPO, bruit appris"}
par = Params()
S = build(par)
test = S["test"]
laws = {b: law(b, AGE_LIMIT) for b in ("TRUE", "POP")}
both, only = laws["TRUE"]
_, dte, *_ = held_out()
H = horizon(dte)
N = len(H)
ref = np.load("results_decision/base_limit.npz")
best = ref["true|TRUE"]
evaluate = make_exact_evaluator(par)
fr = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")
sp = lambda n: f"{int(n):,}".replace(",", " ")


def stats(x, seed=7):
    idx = np.random.default_rng(seed).integers(0, x.size, (2000, x.size))
    m = x[idx].mean(axis=1)
    return dict(mean=float(x.mean()), ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], median=float(np.median(x)),
                q90=float(np.percentile(x, 90)), q99=float(np.percentile(x, 99)), max=float(x.max()), min=float(x.min()),
                share_above_0_1=float(np.mean(x > 0.1)), share_above_1=float(np.mean(x > 1.0)))


# ------------------------------------------------------------------ agents
agents = {}
selected = set(filter(None, os.environ.get("CARUP_AGENTS", "").split(",")))
for kind in KIND:
    for path in sorted(glob.glob(f"{OUT}/{kind}_seed*.npz")):
        name = os.path.basename(path)[:-4]
        if selected and name not in selected:
            continue
        width = int(np.load(path)["width"])
        params = load_params(path, init_policy(jax.random.PRNGKey(0), hidden=(width, width)))
        alpha, k, val = evaluate(params, test, both, only)
        agents[name] = dict(kind=kind, alpha=alpha, phi=MENU[k], value=val, info=json.load(open(path[:-4] + ".json")))
names = list(agents)

# the best consumption at imposed annuity choices: each agent's choice, and two simple choices
alpha_cols = [agents[a]["alpha"] for a in names] + [np.ones(N), np.zeros(N)]
phi_cols = [agents[a]["phi"] for a in names] + [np.full(N, 2.0 / 3.0), np.full(N, MENU[0])]
at_action = value_at_actions(par, laws, H, np.stack(alpha_cols, 1), np.stack(phi_cols, 1))
loss = lambda v: 100 * wealth_equivalent_loss(v, best, par)

out = {"couples": int(N), "rule": RULE, "agents": {}, "references": {}}
per_couple = dict(alpha_opt=ref["alpha|TRUE"], phi_opt=ref["phi|TRUE"])
buy = ref["alpha|TRUE"] > 0
for j, a in enumerate(names):
    g, g_choice = loss(agents[a]["value"]), loss(at_action[:, j])
    per_couple.update({f"{a}|alpha": agents[a]["alpha"], f"{a}|phi": agents[a]["phi"], f"{a}|gap": g, f"{a}|gap_choice": g_choice})
    info = agents[a]["info"]
    out["agents"][a] = dict(
        kind=agents[a]["kind"], seed=info["seed"], seconds=info["seconds"], lives=info["curve"][-1]["lives"],
        gap=stats(g), gap_choice=stats(g_choice), gap_consumption_mean=float((g - g_choice).mean()),
        gap_training_monitoring=info["curve"][-1]["gap_mean"],
        alpha_mean=float(agents[a]["alpha"].mean()), alpha_abs_diff=float(np.abs(agents[a]["alpha"] - ref["alpha|TRUE"]).mean()),
        same_share=float(np.mean(np.isclose(agents[a]["phi"], ref["phi|TRUE"])[buy])), curve=info["curve"])

# reference lines
for m, label in (("POP", "Ménage qui décide avec l'âge et le sexe seulement (POP)"),
                 ("M0", "Ménage qui décide avec le modèle d'indépendance (M0)"),
                 ("MARG", "Ménage qui ignore la dépendance seule (MARG)")):
    out["references"][m] = dict(label=label, gap=stats(loss(ref[f"true|{m}"])))
y = (1.0 - par.load) / (np.asarray(test["aB"]) + (2.0 / 3.0) * np.asarray(test["aA"]))      # everything annuitised, 2/3 to the survivor
disc = par.beta ** np.arange(T_MAX)
u = lambda c: c ** (1.0 - par.g) / (1.0 - par.g)
alone = only.sum(axis=(1, 2))[:, :T_MAX]
rule = (disc[None, :] * (both[:, :T_MAX] * 2.0 * u(y / par.n)[:, None] + alone * u(2.0 / 3.0 * y)[:, None])).sum(axis=1)
out["references"]["rule_income"] = dict(label="Règle : tout en rente avec réversion 2/3, et consommer son revenu", gap=stats(loss(rule)))
out["references"]["rule_all_in"] = dict(label="Règle : tout en rente avec réversion 2/3, puis consommation optimale", gap=stats(loss(at_action[:, len(names)])))
out["references"]["no_annuity"] = dict(label="Aucune rente, puis consommation optimale", gap=stats(loss(at_action[:, len(names) + 1])))
np.savez(f"{OUT}/agents_per_couple.npz", **per_couple)

# ------------------------------------------------------------------ report
L = ["# Agents de décision : écart à l'optimum exact", "",
     f"{sp(N)} couples mis de côté, cas de base, lois avec âge limite de 110 ans. Écart : part des ressources initiales que le ménage "
     "céderait pour passer de la politique de l'agent à la politique optimale (en %). La politique de l'agent est suivie le long de "
     "toutes les histoires du couple : l'écart est exact, sans bruit de simulation. Intervalles : bootstrap sur les couples. "
     "Pendant l'entraînement, les agents sont suivis sur 300 couples d'estimation ; les couples mis de côté ne servent qu'ici.", "",
     "| Agent | Graine | Vies simulées | Durée | Écart moyen [IC 95 %] | Médiane | 90ᵉ centile | Maximum | Dû au choix de la rente | Dû à la consommation | "
     "Part mise en rente (moyenne) | Même réversion que l'optimum | Écart sur les 300 couples d'estimation suivis |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for a in names:
    r = out["agents"][a]
    L.append(f"| {KIND[r['kind']]} | {r['seed']} | {sp(r['lives'])} | {r['seconds'] / 60:.0f} min | "
             f"{fr(r['gap']['mean'])} [{fr(r['gap']['ci'][0])} ; {fr(r['gap']['ci'][1])}] | {fr(r['gap']['median'])} | {fr(r['gap']['q90'])} | "
             f"{fr(r['gap']['max'], 2)} | {fr(r['gap_choice']['mean'])} | {fr(r['gap_consumption_mean'])} | {fr(100 * r['alpha_mean'], 1)} % | "
             f"{fr(100 * r['same_share'], 1)} % | {fr(r['gap_training_monitoring'])} |")
out["alpha_opt_mean"] = float(ref["alpha|TRUE"].mean())
L += ["", f"À l'optimum, la part mise en rente est de {fr(100 * out['alpha_opt_mean'], 1)} % en moyenne. « Dû au choix de la rente » : écart "
      "du choix de l'agent suivi de la meilleure consommation ; « dû à la consommation » : le reste.", "",
      "Les agents apprennent sur des vies simulées des couples d'estimation et corrigent la même règle simple de consommation (rl_core). "
      "PPO n'utilise ni l'équation de budget ni le modèle de mortalité : il n'apprend que des récompenses. L'agent à gradient utilise "
      "l'équation de budget et essaie les quatre taux de réversion sur chaque vie.", "",
      "Par type d'agent et par budget de vies, sur les graines :", "",
      "| Agent | Vies simulées | Graines | Écart moyen : moyenne des graines | Minimum | Maximum | Écart-type entre graines |", "|---|---|---|---|---|---|---|"]
out["by_kind"] = {}
for kind in KIND:
    for lives in sorted({out["agents"][a]["lives"] for a in names if out["agents"][a]["kind"] == kind}):
        g = [out["agents"][a]["gap"]["mean"] for a in names if out["agents"][a]["kind"] == kind and out["agents"][a]["lives"] == lives]
        out["by_kind"][f"{kind}|{lives}"] = dict(seeds=len(g), mean=float(np.mean(g)), min=float(min(g)), max=float(max(g)),
                                                  sd=float(np.std(g, ddof=1)) if len(g) > 1 else None)
        L.append(f"| {KIND[kind]} | {sp(lives)} | {len(g)} | {fr(np.mean(g))} | {fr(min(g))} | {fr(max(g))} | "
                 f"{fr(np.std(g, ddof=1)) if len(g) > 1 else '—'} |")
L += ["", "Repères sur les mêmes couples :", "", "| Politique | Écart moyen [IC 95 %] | Médiane | Maximum |", "|---|---|---|---|"]
for key in ("MARG", "M0", "POP", "rule_all_in", "rule_income", "no_annuity"):
    r = out["references"][key]
    L.append(f"| {r['label']} | {fr(r['gap']['mean'])} [{fr(r['gap']['ci'][0])} ; {fr(r['gap']['ci'][1])}] | "
             f"{fr(r['gap']['median'])} | {fr(r['gap']['max'], 2)} |")
L += ["", "Courbe d'apprentissage (écart moyen exact sur les 300 couples d'estimation suivis, en %) :", ""]
for kind in KIND:
    for lives_end in sorted({out["agents"][a]["lives"] for a in names if out["agents"][a]["kind"] == kind}):
        runs = [out["agents"][a] for a in names if out["agents"][a]["kind"] == kind and out["agents"][a]["lives"] == lives_end]
        lives = [c["lives"] for c in runs[0]["curve"]]
        keep = [i for i in range(len(lives)) if i % max(1, len(lives) // 8) == 0 or i == len(lives) - 1]
        L += [f"{KIND[kind]}, {sp(lives_end)} vies :", "", "| Vies simulées | " + " | ".join(f"graine {r['seed']}" for r in runs) + " |", "|---|" + "---|" * len(runs)]
        for i in keep:
            L.append("| " + sp(lives[i]) + " | " + " | ".join(fr(r["curve"][i]["gap_mean"]) if i < len(r["curve"]) else "—" for r in runs) + " |")
        L.append("")
json.dump(out, open(f"{OUT}/agents_summary.json", "w"), indent=2)
open(f"{OUT}/RESULTATS_AGENTS.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
