"""
check_xai.py -- controls of the explanation step (results_xai/check_xai.txt).

  1. The hybrid machinery rebuilds the real couples exactly: inputs of the networks, optimum of
     dynamic programming, decisions and exact gap of the agents (against the files of step 4);
     the hybrid couples are the ones the analysis assumes (positions 0, 31 and 32 + c).
  2. Shapley weights: against the average over all orders of the players (5 players); enumeration
     restricted to the differing characteristics against the full enumeration (13 players),
     including a pair in which characteristics have been made equal.
  3. Same SHAP values as the shap library (exact explainer, independent background), on a few
     couples, with the agent's decision rewritten in numpy.
  4. Properties of the results: the SHAP values add up to the decision minus the average decision;
     the pair (j, i) is the opposite of the pair (i, j); bootstrap in matrix form against explicit
     resampling.
  5. Changed couples of the "all else equal" effects: size of the change in natural units.
  6. Trees: first split of the direct tree against an exhaustive search on the real losses; our routing
     of the couples through a tree against scikit-learn's.
"""
import os
from itertools import permutations

import numpy as np
import jax
import jax.numpy as jnp

from xai_setup import Setup, RAW, N_RAW, OWN_H, OWN_W, BINARY, coalitions, shapley_matrix, group_masks, hybrid
import xai_shap as X
from rl_core import build, MENU
from decision_dp import wealth_equivalent_loss

L = []


def say(msg):
    print(msg, flush=True)
    L.append(msg)


S = Setup()
Z = S.Z["test"]
table = X.annuity_table(S)
files = X.agent_files()
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ 1. real couples rebuilt
say("1. Couples réels reconstruits à partir de leurs 13 caractéristiques")
B = build(S.par)
for name, label in (("train", "d'estimation"), ("test", "mis de côté")):
    say(f"   entrées des réseaux, couples {label} : écart maximal {np.abs(S.features(S.Z[name]) - np.asarray(B[name]['feat'])).max():.1e}")
if os.path.exists("results_xai/hybrids.npz"):
    h = np.load("results_xai/hybrids.npz")
    ref = np.load("results_decision/base_limit.npz")
    q = np.load("results_rl/agents_per_couple.npz")
    row, pi, pj, Zu = h["row"], h["i"], h["j"], h["Z"]
    rows, couples = np.concatenate([row[:, 31], row[:, 0]]), np.concatenate([pi, pj])
    say(f"   position 31 = couple i : {bool(np.array_equal(Zu[row[:, 31]], Z[pi]))} ; position 0 = couple j : {bool(np.array_equal(Zu[row[:, 0]], Z[pj]))} ; "
        f"position 32 + c = couple i avec la caractéristique c de j : "
        f"{bool(all(np.array_equal(Zu[row[:, 32 + c]], hybrid(Z[pi], Z[pj], np.arange(N_RAW) != c)) for c in range(N_RAW)))}")
    gm = group_masks()
    say(f"   positions 0 à 31 = combinaisons des groupes : {bool(all(np.array_equal(Zu[row[:, c]], hybrid(Z[pi], Z[pj], gm[c])) for c in range(32)))}")
    say(f"   optimum de {len(np.unique(couples))} couples réels recalculé avec les couples hybrides, contre l'étape 4 : "
        f"part en rente {np.nanmax(np.abs(h['alpha'][rows] - ref['alpha|TRUE'][couples])):.1e}, "
        f"réversion {np.nanmax(np.abs(h['phi'][rows] - ref['phi|TRUE'][couples])):.1e}, "
        f"valeur (écart relatif) {np.abs(h['value'][rows] / ref['true|TRUE'][couples] - 1).max():.1e}")
    for path in files:
        name = os.path.basename(path)[:-4]
        if f"{name}|value" in h.files and f"{name}|gap" in q.files:
            gap = 100 * wealth_equivalent_loss(h[f"{name}|value"][rows], h["value"][rows], S.par)
            say(f"   {name} : part en rente {np.abs(h[f'{name}|alpha'][rows] - q[f'{name}|alpha'][couples]).max():.1e}, "
                f"écart à l'optimum (points) {np.abs(gap - q[f'{name}|gap'][couples]).max():.1e}")
else:
    say("   results_xai/hybrids.npz absent : contrôle de l'optimum non fait")

# ------------------------------------------------------------------ 2. Shapley weights
say("2. Poids de Shapley")
v = rng.normal(size=32)
phi = np.zeros(5)
for perm in permutations(range(5)):
    s = 0
    for k in perm:
        phi[k] += v[s | (1 << k)] - v[s]
        s |= 1 << k
say(f"   5 joueurs, matrice contre la moyenne sur les 120 ordres : écart maximal {np.abs(shapley_matrix(5) @ v - phi / 120).max():.1e}")
params = X.load_agent(files[0])
M, C = jnp.asarray(coalitions(N_RAW)), jnp.asarray(shapley_matrix(N_RAW))
full = jax.jit(lambda zi, zj: jax.vmap(lambda a, b: C @ X.decide(params, jnp.where(M, a, b), table))(zi, zj))     # 8 192 coalitions per pair
I, J = rng.choice(len(Z), 40, replace=False), rng.choice(len(Z), 40, replace=False)
got = X.pair_shapley(params, Z[I], Z[J], table)
say(f"   13 joueurs, {len(I)} paires : énumération des seules caractéristiques différentes contre énumération des 8 192 coalitions : "
    f"écart maximal {np.abs(got - np.asarray(full(jnp.asarray(Z[I]), jnp.asarray(Z[J])))).max():.1e}")
Zj2 = Z[J].copy()
Zj2[:, [4, 6, 12]] = Z[I][:, [4, 6, 12]]                                 # three characteristics made equal in the two couples
got2 = X.pair_shapley(params, Z[I], Zj2, table)
say(f"   mêmes paires, trois caractéristiques rendues égales ({', '.join(RAW[k] for k in (4, 6, 12))}) : écart maximal avec l'énumération complète "
    f"{np.abs(got2 - np.asarray(full(jnp.asarray(Z[I]), jnp.asarray(Zj2)))).max():.1e}")

# ------------------------------------------------------------------ 3. shap library
say("3. Bibliothèque shap (explainers.Exact, maskers.Independent), décision de l'agent réécrite en numpy")
name = os.path.basename(files[0])[:-4]
ann = [(np.asarray(w), np.asarray(b)) for w, b in params["ann"]]
T_np = np.asarray(table)


def decide_np(z):
    """Decision (alpha, phi) of the agent from the 13 characteristics, without JAX."""
    z = np.asarray(z, dtype=float)
    x = np.concatenate([(z[:, :2] - 65.0) / 10.0, z[:, OWN_H], z[:, OWN_W],
                        T_np[(z[:, 0] - X.A0).astype(int), (z[:, 1] - X.A0).astype(int)]], axis=1)
    for w, b in ann[:-1]:
        x = np.tanh(x @ w + b)
    out = x @ ann[-1][0] + ann[-1][1]
    k = out[:, len(MENU):].argmax(axis=1)
    return np.stack([1.0 / (1.0 + np.exp(-out[np.arange(len(out)), k])), MENU[k]], axis=1)


say(f"   décisions numpy contre JAX, {len(Z)} couples : écart maximal {np.abs(decide_np(Z) - X.decisions(params, Z, table)).max():.1e}")
try:
    import shap
    couples, ref_couples = rng.choice(len(Z), 3, replace=False), rng.choice(len(Z), 150, replace=False)
    ii, jj = np.meshgrid(couples, ref_couples, indexing="ij")
    ours = X.pair_shapley(params, Z[ii.ravel()], Z[jj.ravel()], table).reshape(3, 150, N_RAW, 2).mean(axis=1)
    expl = shap.explainers.Exact(decide_np, shap.maskers.Independent(Z[ref_couples], max_samples=len(ref_couples)))
    sv = expl(Z[couples], max_evals=2 ** N_RAW + 10, silent=True)
    say(f"   shap {shap.__version__}, agent {name}, {len(couples)} couples expliqués, {len(ref_couples)} couples de référence (la bibliothèque manque de "
        f"mémoire avec les {len(Z)}) : écart maximal avec notre calcul sur les mêmes couples {np.abs(np.asarray(sv.values) - ours).max():.1e}")
except ImportError:
    say("   bibliothèque shap non installée : contrôle non fait")

# ------------------------------------------------------------------ 4. properties of the stored values
say("4. Propriétés des valeurs")
for path in files:
    nm = os.path.basename(path)[:-4]
    if os.path.exists(f"results_xai/shap_{nm}.npz"):
        r = np.load(f"results_xai/shap_{nm}.npz")
        eff = np.abs(r["shap"].sum(axis=1) - (r["decision"] - r["decision"].mean(axis=0))).max()
        say(f"   {nm} : somme des valeurs SHAP = décision du couple - décision moyenne des couples, écart maximal {eff:.1e}")
ba = X.pair_shapley(params, Z[J], Z[I], table)
say(f"   paire (j, i) contre paire (i, j), calculées séparément : somme des deux, valeur maximale {np.abs(got + ba).max():.1e}")
sub = rng.choice(len(Z), 80, replace=False)
ii, jj = np.meshgrid(sub, sub, indexing="ij")
P = X.pair_shapley(params, Z[ii.ravel()], Z[jj.ravel()], table).reshape(80, 80, N_RAW, 2)
boot = X.importance_bootstrap(P.astype(np.float32), 20, 3, block=8)
r2, explicit = np.random.default_rng(3), []
for lo in range(0, 20, 8):
    for c in r2.multinomial(80, np.full(80, 1.0 / 80), size=min(8, 20 - lo)):
        pick = np.repeat(np.arange(80), c)
        explicit.append(np.abs(P[pick][:, pick].mean(axis=1)).mean(axis=0))      # SHAP values of the resampled couples against the resampled couples
say(f"   bootstrap sous forme matricielle contre rééchantillonnage explicite (80 couples, 20 tirages) : écart maximal {np.abs(boot - np.array(explicit)).max():.1e}")

# ------------------------------------------------------------------ 5. changed couples
say("5. Couples modifiés d'une variation donnée (section 4 du rapport)")
Zc, sign = S.changed_couples(Z)
nat = S.natural["test"]
for k, key in enumerate(RAW):
    d = np.abs(Zc[:, k, k] - Z[:, k])
    others = np.abs(np.delete(Zc[:, k, :], k, axis=1) - np.delete(Z, k, axis=1)).max()
    if key in BINARY:
        size = f"autre état pour tous les couples : {bool(np.allclose(d, 1.0))}"
    elif key == "wealth":
        c0, (mu, sd) = S.scaler["centre"][0, 5], S.scaler["wealth"]
        w1 = 1000 * np.sinh((Zc[:, k, k] + c0) * sd + mu)
        big = nat[:, k] > 50000
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sign[:, k] > 0, w1 / nat[:, k], nat[:, k] / w1)[big]
        size = f"patrimoine multiplié par {ratio.min():.3f} à {ratio.max():.3f} pour les {int(big.sum())} couples au-dessus de 50 000 $"
    else:
        unit = d * (1.0 if key.startswith("age") else S.scaler[key][1])
        size = f"variation de {unit.min():.3f} à {unit.max():.3f} en unités naturelles"
    say(f"   {key} : {size} ; augmentée pour {100 * np.mean(sign[:, k] > 0):.1f} % des couples ; autres caractéristiques inchangées : {bool(others == 0)}")

# ------------------------------------------------------------------ 6. direct tree
if os.path.exists("results_xai/grid_train.npz"):
    say("6. Arbres")
    import xai_tree as XT
    grid, opt = np.load("results_xai/grid_train.npz"), np.load("results_xai/optimum_train.npz")
    Xtr = XT.inputs(S.natural["train"])
    Lm = 100 * wealth_equivalent_loss(grid["value"], opt["value"][:, None], S.par)
    best = (np.inf, None)
    for f in range(Xtr.shape[1]):
        for thr in np.unique(Xtr[:, f])[:-1]:
            left = Xtr[:, f] <= thr
            if min(left.sum(), (~left).sum()) >= XT.MIN_LEAF:
                tot = Lm[left].sum(axis=0).min() + Lm[~left].sum(axis=0).min()
                if tot < best[0]:
                    best = (tot, (XT.FEATURES[f], float(thr)))
    g1 = XT.grow_direct(Xtr, Lm, grid["alpha"], grid["phi"], np.arange(len(Xtr)), 1)
    a_in, p_in = XT.actions(g1, Xtr)
    k_in = np.array([int(np.where(np.isclose(grid["alpha"], a) & np.isclose(grid["phi"], p))[0][0]) for a, p in zip(a_in, p_in)])
    say(f"   première coupure, recherche exhaustive : {best[1][0]} après la valeur {best[1][1]:g}, perte totale {best[0]:.6f} ; "
        f"arbre direct : {XT.FEATURES[g1['feature']]} au seuil {g1['threshold']:g}, perte totale {Lm[np.arange(len(Xtr)), k_in].sum():.6f}")
    part, model = XT.partition_from_teacher(Xtr, opt["alpha"], opt["phi"], 6, return_model=True)
    same = []
    for Xc in (Xtr, XT.inputs(S.natural["test"])):
        ours = np.empty(len(Xc), dtype=int)
        for n, (_, idx) in enumerate(XT.leaves(part, Xc)):
            ours[idx] = n
        theirs = model.apply(Xc)
        same.append(len(set(zip(ours.tolist(), theirs.tolist()))) == len(set(ours.tolist())) == len(set(theirs.tolist())))
    say(f"   arbre de profondeur 6 tiré de l'optimum : mêmes feuilles que scikit-learn pour les couples d'estimation : {same[0]}, "
        f"pour les couples mis de côté : {same[1]}")
    say(f"   plus petite perte d'une action de la grille par rapport à l'optimum, en % des ressources : {Lm.min():.1e} "
        "(une valeur négative mesure la précision de la recherche de la part en rente)")
open("results_xai/check_xai.txt", "w", encoding="utf-8").write("\n".join(L) + "\n")
