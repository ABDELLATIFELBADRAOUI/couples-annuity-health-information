"""summarize_decision.py -- tables of the decision analysis, from results_decision/<scenario>.npz.

Usage: python summarize_decision.py          scenarios without limiting age (histories cut at the horizon)
       python summarize_decision.py limit    reference scenarios, with the limiting age of 110
Writes results_decision/decision_summary[_limit].json and results_decision/RESULTATS_DECISION[_LIMITE].md.
The intervals are bootstrap intervals over the couples (2,000 resamples): they describe the sampling
of the couples, not the uncertainty of the mortality model, which is held at its posterior predictive.
"""
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

from decision_setup import law, held_out, horizon
from decision_dp import Params, annuity_due, conditional_probabilities, make_solver, wealth_equivalent_loss
from run_decision import SCENARIOS

TITLES = {
    "base": "Cas de base",
    "pre50": "Moitié des ressources déjà en rente (réversion 2/3)",
    "freephi": "Taux de réversion libre entre 40 % et 100 % (pas de 5 points)",
    "g2": "Aversion au risque 2", "g5": "Aversion au risque 5",
    "load0": "Rente sans chargement", "load20": "Chargement de 20 %",
}
NAMES = {"MARG": "MARG : mêmes survies individuelles, conjoints indépendants", "M0": "M0 : modèle d'indépendance",
         "M1": "M1 : M0 + covariables du conjoint", "M2": "M2 : M1 + effet de veuvage",
         "MD": "MD : dépendance sans santé (âge, sexe, veuvage, fragilité)",
         "POP": "POP : âge et sexe seulement", "TRUE": "Modèle complet (vérité)"}
ORDER = ["MARG", "M0", "M1", "M2", "MD", "POP"]
SUFFIX = "_" + sys.argv[1] if len(sys.argv) > 1 else ""
AGE_LIMIT = 110 if SUFFIX == "_limit" else None
path_of = lambda name: f"results_decision/{name}{SUFFIX}.npz"
_, dte, *_ = held_out()
H = horizon(dte)
N = len(dte)


def fr(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def load(name):
    z = dict(np.load(path_of(name)))
    if name == "base" and not SUFFIX and os.path.exists("results_decision/base_marg.npz"):
        extra = np.load("results_decision/base_marg.npz")
        assert np.allclose(extra["true|TRUE"], z["true|TRUE"], rtol=1e-12, atol=0), "the two base runs disagree on the truth"
        z.update({k: extra[k] for k in extra.files if k.endswith("|MARG")})
    z.setdefault("index", np.arange(N))
    g, n_eq, r, beta, load_, xi = z["params"]
    par = Params(g=float(g), n=float(n_eq), r=float(r), beta=float(beta), load=float(load_), xi=float(xi), phis=tuple(z["phis"]))
    return z, par, [m for m in ORDER if f"true|{m}" in z]


def loss_of(z, par, m, key="true"):
    return 100 * wealth_equivalent_loss(z[f"{key}|{m}"], z["true|TRUE"], par)


def stats(x, seed=7):
    idx = np.random.default_rng(seed).integers(0, x.size, (2000, x.size))
    m = x[idx].mean(axis=1)
    return dict(n=int(x.size), mean=float(x.mean()), ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                median=float(np.median(x)), q90=float(np.percentile(x, 90)), q99=float(np.percentile(x, 99)), max=float(x.max()),
                min=float(x.min()), share_above_0_1=float(np.mean(x > 0.1)), share_above_1=float(np.mean(x > 1.0)))


summary = {}
L = ["# Coût d'une mauvaise croyance de mortalité dans la décision de rente du couple", "",
     f"Couples : les {N} couples mis de côté. Vérité : le modèle complet (M3) ajusté sur les couples d'estimation. "
     "Le ménage choisit la part de son patrimoine mise en rente, le taux de réversion, puis sa consommation de chaque année. "
     "Un ménage qui croit un autre modèle décide comme si ce modèle était vrai ; sa perte est la part des ressources "
     "initiales qu'il accepterait de céder pour décider avec le modèle complet (en %).", "",
     "Cas de base : aversion relative au risque 3, échelle d'équivalence 1,5, taux 3 %, chargement de la rente 10 %, "
     "rente tarifée sur l'âge et le sexe seulement, aucune rente préexistante, réversion au choix parmi 50 %, 66,7 %, 75 % et 100 %. "
     "Ces paramètres sont des hypothèses de scénario, pas des estimations.", "",
     ("Chaque loi de mortalité a un âge limite de 110 ans : personne n'est en vie à 110 ans." if AGE_LIMIT else
      "Sans âge limite : l'histoire d'un couple est coupée quand le plus jeune conjoint atteint 110 à 119 ans."), "",
     "Intervalles : bootstrap sur les couples. Ils ne contiennent pas l'incertitude du modèle de mortalité.", ""]

for name in ("base", "pre50"):
    if not os.path.exists(path_of(name)):
        continue
    z, par, beliefs = load(name)
    res = {"title": TITLES[name], "couples": int(z["index"].size), "seconds": float(z["seconds"]), "beliefs": {}, "choices": {}}
    part = "Part du patrimoine liquide mise en rente (moyenne)" if par.xi > 0 else "Part mise en rente (moyenne)"
    L += [f"## {TITLES[name]} ({z['index'].size} couples)", "", "### Décisions", "",
          "Les colonnes de réversion donnent la répartition parmi les ménages qui achètent une rente.", "",
          f"| Croyance | {part} | 99,9 % ou plus | Moins de la moitié | Aucune rente | Réversion 50 % | 66,7 % | 75 % | 100 % |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in ["TRUE"] + beliefs:
        a, p = z[f"alpha|{m}"], z[f"phi|{m}"]
        has = a > 0
        ch = dict(alpha_mean=float(a.mean()), share_full=float(np.mean(a >= 0.999)), share_below_half=float(np.mean(a < 0.5)),
                  share_none=float(np.mean(a == 0)),
                  phi_shares={f"{q:.3f}": float(np.mean(np.isclose(p[has], q))) for q in (0.5, 2 / 3, 0.75, 1.0)})
        res["choices"][m] = ch
        s_ = ch["phi_shares"]
        L.append(f"| {NAMES[m]} | {fr(100 * ch['alpha_mean'], 1)} % | {fr(100 * ch['share_full'], 1)} % | {fr(100 * ch['share_below_half'], 1)} % | "
                 f"{fr(100 * ch['share_none'], 1)} % | {fr(100 * s_['0.500'], 1)} % | {fr(100 * s_['0.667'], 1)} % | {fr(100 * s_['0.750'], 1)} % | {fr(100 * s_['1.000'], 1)} % |")
    L += ["", "### Pertes (en % des ressources)", "",
          "| Croyance | Moyenne [IC 95 %] | Médiane | 90ᵉ centile | 99ᵉ centile | Maximum | Couples perdant plus de 0,1 % | Plus de 1 % | Part de la perte due au choix de la rente | Même réversion que la vérité (quand les deux achètent une rente) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m in beliefs:
        st = stats(loss_of(z, par, m))
        st["action_part_mean"] = float(loss_of(z, par, m, "true_action").mean())
        buy = (z[f"alpha|{m}"] > 0) & (z["alpha|TRUE"] > 0)
        st["same_phi"] = float(np.mean(np.isclose(z[f"phi|{m}"][buy], z["phi|TRUE"][buy])))
        st["alpha_abs_diff_mean"] = float(np.abs(z[f"alpha|{m}"] - z["alpha|TRUE"]).mean())
        res["beliefs"][m] = st
        L.append(f"| {NAMES[m]} | {fr(st['mean'])} [{fr(st['ci'][0])} ; {fr(st['ci'][1])}] | {fr(st['median'])} | {fr(st['q90'])} | {fr(st['q99'], 2)} | "
                 f"{fr(st['max'], 2)} | {fr(100 * st['share_above_0_1'], 1)} % | {fr(100 * st['share_above_1'], 1)} % | "
                 f"{fr(100 * st['action_part_mean'] / st['mean'], 0)} % | {fr(100 * st['same_phi'], 1)} % |")
    L.append("")
    for ref in ("MARG", "M0"):
        if ref in beliefs and "POP" in beliefs:
            res[f"ratio_pop_over_{ref.lower()}"] = res["beliefs"]["POP"]["mean"] / res["beliefs"][ref]["mean"]
            L.append(f"Perte moyenne avec POP divisée par la perte moyenne avec {ref} : {fr(res[f'ratio_pop_over_{ref.lower()}'], 1)}.")
    res["smallest_loss"] = min(res["beliefs"][m]["min"] for m in beliefs)
    L += ["", f"Plus petite perte individuelle : {res['smallest_loss']:.1e} % (une perte négative signalerait une erreur numérique).", ""]
    summary[name] = res

# ---- sensitivity scenarios, on a subsample
rows = []
for name in ("freephi", "g2", "g5", "load0", "load20"):
    if not os.path.exists(path_of(name)) or not os.path.exists(path_of("base")):
        continue
    z, par, beliefs = load(name)
    zb, parb, _ = load("base")
    i = z["index"]
    res = {"title": TITLES[name], "couples": int(i.size), "seconds": float(z["seconds"]), "beliefs": {}, "base_same_couples": {}}
    for m in beliefs:
        res["beliefs"][m] = stats(loss_of(z, par, m))
        if f"true|{m}" in zb:
            res["base_same_couples"][m] = float((100 * wealth_equivalent_loss(zb[f"true|{m}"][i], zb["true|TRUE"][i], parb)).mean())
    res["smallest_loss"] = min(res["beliefs"][m]["min"] for m in beliefs)
    summary[name] = res
    rows.append((name, res))
if rows:
    n_sub = rows[0][1]["couples"]
    L += [f"## Sensibilité ({n_sub} couples tirés au hasard, les mêmes dans chaque scénario)", "",
          "Perte moyenne en % des ressources [IC 95 %]. Première ligne : le cas de base sur ces mêmes couples.", "",
          "| Scénario | MARG | M0 | POP | POP / M0 |", "|---|---|---|---|---|"]
    b = rows[0][1]["base_same_couples"]
    L.append(f"| Cas de base | {fr(b['MARG']) if 'MARG' in b else '—'} | {fr(b['M0'])} | {fr(b['POP'])} | {fr(b['POP'] / b['M0'], 1)} |")
    for name, res in rows:
        cell = lambda m: (f"{fr(res['beliefs'][m]['mean'])} [{fr(res['beliefs'][m]['ci'][0])} ; {fr(res['beliefs'][m]['ci'][1])}]"
                          if m in res["beliefs"] else "—")
        L.append(f"| {TITLES[name]} | {cell('MARG')} | {cell('M0')} | {cell('POP')} | "
                 f"{fr(res['beliefs']['POP']['mean'] / res['beliefs']['M0']['mean'], 1)} |")
    L += ["", "Plus petite perte individuelle dans ces scénarios : "
          + ", ".join(f"{TITLES[n]} {r['smallest_loss']:.1e} %" for n, r in rows) + ".", ""]

# ---- robustness to the horizon: limiting age of 110 in every law
pairs = [] if SUFFIX else [(a, b) for a, b in (("base", "base_limit300"), ("pre50", "pre50_limit300"), ("load20", "load20_limit"))
                           if os.path.exists(f"results_decision/{a}.npz") and os.path.exists(f"results_decision/{b}.npz")]
if pairs:
    L += ["## Sensibilité à l'horizon : âge limite de 110 ans", "",
          "Dans les calculs ci-dessus, l'histoire d'un couple est coupée quand le plus jeune conjoint atteint 110 à 119 ans, alors "
          "que le modèle complet laisse encore une probabilité d'être en vie à cette date (0,17 % en moyenne, 2,8 % au plus). "
          "Ici, chaque loi reçoit un âge limite de 110 ans : plus rien n'est coupé. Mêmes 300 couples que pour la sensibilité. "
          "Perte moyenne en % des ressources.", "",
          "| Scénario | Loi | MARG | M0 | POP | POP / M0 | Aucune rente (vérité) | Perte maximale M0 |", "|---|---|---|---|---|---|---|---|"]
    summary["age_limit"] = {}
    for a, b in pairs:
        zl = dict(np.load(f"results_decision/{b}.npz"))
        parl = Params(g=float(zl["params"][0]), load=float(zl["params"][4]), xi=float(zl["params"][5]), phis=tuple(zl["phis"]))
        bel = [m for m in ORDER if f"true|{m}" in zl]
        i = zl["index"]
        za, para, _ = load(a)
        full = za["index"].size == N                                   # scenario computed on all couples: restrict to the same 300
        pos = i if full else np.arange(i.size)
        assert full or np.array_equal(za["index"], i)
        row = {}
        for label, zz, pp, sel in (("coupée à l'horizon", za, para, pos), ("âge limite 110 ans", zl, parl, np.arange(i.size))):
            vals = {m: float((100 * wealth_equivalent_loss(zz[f"true|{m}"][sel], zz["true|TRUE"][sel], pp)).mean())
                    for m in ("MARG", "M0", "POP") if f"true|{m}" in zz}
            none = float(np.mean(zz["alpha|TRUE"][sel] == 0))
            mx = float((100 * wealth_equivalent_loss(zz["true|M0"][sel], zz["true|TRUE"][sel], pp)).max())
            row[label] = dict(vals, none=none, max_m0=mx)
            L.append(f"| {TITLES.get(a, a)} | {label} | {fr(vals['MARG']) if 'MARG' in vals else '—'} | {fr(vals['M0'])} | {fr(vals['POP'])} | "
                     f"{fr(vals['POP'] / vals['M0'], 1)} | {fr(100 * none, 1)} % | {fr(mx, 2)} |")
        row["smallest_loss_with_limit"] = min(float((100 * wealth_equivalent_loss(zl[f"true|{m}"], zl["true|TRUE"], parl)).min()) for m in bel)
        summary["age_limit"][a] = row
    L += ["", "Plus petite perte individuelle avec l'âge limite : "
          + ", ".join(f"{TITLES.get(a, a)} {summary['age_limit'][a]['smallest_loss_with_limit']:.1e} %" for a, _ in pairs) + ".", "",
          "Ces moyennes ne dépendent donc pas de la coupure. En revanche, pour un ménage qui n'achète aucune rente et n'a aucun revenu, "
          "la perte dépend fortement de ce que le modèle suppose aux âges extrêmes : voir `check_horizon.py` dans les contrôles. "
          "Ce cas n'existe pas dans le cas de base (aucun ménage sans rente) ni quand la moitié des ressources est déjà en rente ; "
          "il concerne quelques couples du scénario « chargement de 20 % », dont la moyenne est à lire avec prudence.", ""]

# ---- context for the base case: value of the annuity, and who loses
if "base" in summary:
    z, par, beliefs = load("base")
    both, only = law("TRUE", AGE_LIMIT)
    pb, po = law("POP", AGE_LIMIT)
    Tb = np.maximum(20, 10 * np.ceil(H / 10.0)).astype(int)
    _, _, no_income = make_solver(par)
    v0 = np.zeros(N)
    for T in sorted(set(Tb.tolist())):
        i = np.where(Tb == T)[0]
        b, o = both[i][:, :T + 1], only[i][:, :, :T, :T + 1]
        q, pB = conditional_probabilities(b, o)
        v0[i] = np.asarray(no_income(jnp.asarray(q), jnp.asarray(pB), jnp.asarray(b), jnp.asarray(o), jnp.asarray(b), jnp.asarray(o))[1])
    gain = 100 * wealth_equivalent_loss(v0, z["true|TRUE"], par)
    summary["base"]["annuity_access_gain"] = stats(gain)
    L += ["## Repères pour le cas de base", "",
          f"Valeur de l'accès à la rente (perte du ménage qui ne peut rien mettre en rente) : moyenne {fr(gain.mean(), 1)} %, "
          f"médiane {fr(np.median(gain), 1)} %, de {fr(gain.min(), 1)} % à {fr(gain.max(), 1)} %.", ""]
    # relative health: true value of a joint annuity with 2/3 to the survivor, against its population value
    rel = annuity_due(both, only, 2 / 3, par.r) / annuity_due(pb, po, 2 / 3, par.r)
    grp = np.digitize(rel, np.quantile(rel, [0.2, 0.4, 0.6, 0.8]))
    cols = [m for m in ("POP", "M0", "MARG") if m in beliefs]
    L += ["Perte moyenne selon la santé relative du couple. Santé relative : valeur vraie d'une rente réversible aux 2/3, divisée "
          "par sa valeur pour la population du même âge ; cinq groupes de même taille, du moins bon état de santé au meilleur.", "",
          "| Groupe | Santé relative moyenne | Part mise en rente (vérité) | " + " | ".join(f"Perte {m}" for m in cols) + " |",
          "|---|---|---|" + "---|" * len(cols)]
    rows = []
    for k in range(5):
        m_ = grp == k
        row = dict(ratio=float(rel[m_].mean()), alpha_true=float(z["alpha|TRUE"][m_].mean()),
                   **{f"loss_{c}": float(loss_of(z, par, c)[m_].mean()) for c in cols})
        rows.append(row)
        L.append(f"| {k + 1} | {fr(row['ratio'], 2)} | {fr(100 * row['alpha_true'], 1)} % | "
                 + " | ".join(fr(row[f"loss_{c}"]) for c in cols) + " |")
    summary["base"]["by_relative_health"] = rows
    age = np.minimum(dte.age_h.values, dte.age_w.values)
    L += ["", "Perte moyenne selon l'âge du plus jeune conjoint :", "",
          "| Âge | Couples | " + " | ".join(f"Perte {m}" for m in cols) + " | Valeur de l'accès à la rente |",
          "|---|---|" + "---|" * (len(cols) + 1)]
    rows = []
    for lo, hi in ((50, 59), (60, 69), (70, 79), (80, 120)):
        m_ = (age >= lo) & (age <= hi)
        row = dict(ages=f"{lo}-{hi}", n=int(m_.sum()), gain=float(gain[m_].mean()),
                   **{f"loss_{c}": float(loss_of(z, par, c)[m_].mean()) for c in cols})
        rows.append(row)
        L.append(f"| {lo} à {hi} ans | {row['n']} | ".replace("80 à 120 ans", "80 ans et plus")
                 + " | ".join(fr(row[f"loss_{c}"]) for c in cols) + f" | {fr(row['gain'], 1)} % |")
    summary["base"]["by_age"] = rows
    L.append("")

json.dump(summary, open(f"results_decision/decision_summary{SUFFIX}.json", "w"), indent=2)
open(f"results_decision/RESULTATS_DECISION{'_LIMITE' if SUFFIX else ''}.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
