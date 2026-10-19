"""
xai_tree.py -- short decision trees for the annuity decision, and what following them costs.

All trees are learned on the training couples and applied to the held-out couples, which played no
part in them. Inputs: the 13 characteristics in natural units and the age difference between spouses.

  imitation : the partition is learned from the decisions of a "teacher" (an agent, or the exact
              optimum of the training couples); each leaf takes the teacher's mean annuitised share
              and its most frequent survivor share. It is the tree that explains the teacher.
  adjusted  : same partition; each leaf takes, among the 81 actions of a grid (xai_dp.py grid), the
              one with the smallest mean loss over the training couples of the leaf.
  direct    : no teacher: the splits are chosen one after the other (greedy search), each one with the
              best grid action on its two sides, to reduce the mean loss of the training couples
              (it needs the exact values of the grid actions; it is not the best tree of its size).

Reported for the held-out couples:
    fidelity : how well the imitation tree reproduces its teacher's decisions
    cost     : exact loss, in % of the initial resources, of making the tree's annuity choice (followed
               by the best consumption) instead of the optimal choice
Depth 0 is the same action for every couple.

Usage: python xai_tree.py
Output: results_xai/xai_tree.json, results_xai/trees_per_couple.npz
"""
import copy
import json
import os

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from xai_setup import Setup, RAW
from xai_shap import agent_files, load_agent, annuity_table, decisions
from decision_setup import law, held_out, horizon
from decision_dp import Params, value_at_actions, wealth_equivalent_loss
from rl_core import AGE_LIMIT, MENU

DEPTHS, MIN_LEAF, SHOWN = (0, 1, 2, 3, 4, 5, 6), 50, 3
FEATURES = RAW + ["age_gap"]


def inputs(nat):
    """The 13 characteristics in natural units and the age difference. The values are rounded to single
    precision, the precision at which scikit-learn compares a value with a threshold: our routing of a
    couple through a tree is then exactly scikit-learn's."""
    X = np.concatenate([nat, (nat[:, 0] - nat[:, 1])[:, None]], axis=1)
    return X.astype(np.float32).astype(np.float64)


# ----------------------------------------------------------------------------- trees as nested dicts
def partition_from_teacher(X, alpha, phi, depth, return_model=False):
    """CART on the standardised pair (alpha, phi) of the teacher; returns the splits only
    (and the scikit-learn tree when return_model is True)."""
    if depth == 0:
        return {"leaf": True}
    sd = lambda v: v.std() if v.std() > 0 else 1.0
    Y = np.stack([(alpha - alpha.mean()) / sd(alpha), (phi - phi.mean()) / sd(phi)], axis=1)
    model = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=MIN_LEAF, random_state=0).fit(X, Y)
    t = model.tree_

    def node(k):
        if t.children_left[k] == -1:
            return {"leaf": True}
        return {"leaf": False, "feature": int(t.feature[k]), "threshold": float(t.threshold[k]),
                "left": node(t.children_left[k]), "right": node(t.children_right[k])}

    return (node(0), model) if return_model else node(0)


def leaves(node, X, idx=None):
    """List of (leaf, indices of the rows of X that reach it)."""
    idx = np.arange(len(X)) if idx is None else idx
    if node["leaf"]:
        return [(node, idx)]
    go = X[idx, node["feature"]] <= node["threshold"]
    return leaves(node["left"], X, idx[go]) + leaves(node["right"], X, idx[~go])


def set_actions(root, X, rule):
    """Copy of the tree whose leaves carry the action given by rule(indices of the training couples)."""
    root = copy.deepcopy(root)
    for leaf, idx in leaves(root, X):
        a, p = rule(idx)
        leaf.update(alpha=float(a), phi=float(p), n_train=int(len(idx)))
    return root


def actions(root, X):
    a, p = np.empty(len(X)), np.empty(len(X))
    for leaf, idx in leaves(root, X):
        a[idx], p[idx] = leaf["alpha"], leaf["phi"]
    return a, p


def used(node):
    return set() if node["leaf"] else {FEATURES[node["feature"]]} | used(node["left"]) | used(node["right"])


def grow_direct(X, Lm, ga, gp, idx, depth):
    """Greedy tree that minimises the total loss of the training couples, leaves at their best grid action."""
    total = Lm[idx].sum(axis=0)
    k = int(total.argmin())
    node = {"leaf": True, "alpha": float(ga[k]), "phi": float(gp[k]), "n_train": int(len(idx))}
    if depth == 0 or len(idx) < 2 * MIN_LEAF:
        return node
    best_val, best = total[k] - 1e-12, None
    for f in range(X.shape[1]):
        order = idx[np.argsort(X[idx, f], kind="stable")]
        xs = X[order, f]
        cs = np.cumsum(Lm[order], axis=0)
        pos = np.where(xs[:-1] < xs[1:])[0]                                  # a split between two different values
        pos = pos[(pos + 1 >= MIN_LEAF) & (len(idx) - pos - 1 >= MIN_LEAF)]
        if not len(pos):
            continue
        tot = cs[pos].min(axis=1) + (cs[-1][None, :] - cs[pos]).min(axis=1)  # best action on each side
        j = int(tot.argmin())
        if tot[j] < best_val:
            best_val, best = tot[j], (f, 0.5 * (xs[pos[j]] + xs[pos[j] + 1]), order[:pos[j] + 1], order[pos[j] + 1:])
    if best is None:
        return node
    f, thr, left, right = best
    return {"leaf": False, "feature": int(f), "threshold": float(thr),
            "left": grow_direct(X, Lm, ga, gp, left, depth - 1), "right": grow_direct(X, Lm, ga, gp, right, depth - 1)}


def describe(node, X, loss):
    """Tree with feature names, and in each leaf the held-out couples and their mean exact loss."""
    out = copy.deepcopy(node)
    for leaf, idx in leaves(out, X):
        leaf.update(n_test=int(len(idx)), loss=float(loss[idx].mean()) if len(idx) else None)

    def rename(n):
        if not n["leaf"]:
            n["feature"] = FEATURES[n["feature"]]
            rename(n["left"]), rename(n["right"])

    rename(out)
    return out


if __name__ == "__main__":
    par = Params()
    S = Setup(par)
    table = annuity_table(S)
    Xtr, Xte = inputs(S.natural["train"]), inputs(S.natural["test"])
    ref = np.load("results_decision/base_limit.npz")
    best, a_opt, p_opt = ref["true|TRUE"], ref["alpha|TRUE"], ref["phi|TRUE"]
    opt_tr = np.load("results_xai/optimum_train.npz")
    grid = np.load("results_xai/grid_train.npz")
    ga, gp = grid["alpha"], grid["phi"]
    Lm = 100 * wealth_equivalent_loss(grid["value"], opt_tr["value"][:, None], par)        # (training couples, 81): loss of each grid action
    teachers = {"exact": (opt_tr["alpha"], opt_tr["phi"], a_opt, p_opt)}                   # decisions on the training and on the held-out couples
    for path in agent_files():
        name = os.path.basename(path)[:-4]
        if name.startswith("apg"):
            params = load_agent(path)
            dtr, dte = decisions(params, S.Z["train"], table), decisions(params, S.Z["test"], table)
            teachers[name] = (dtr[:, 0], dtr[:, 1], dte[:, 0], dte[:, 1])

    def mode(v):                                                                           # most frequent survivor share, as a value of the menu
        vals, counts = np.unique(np.round(v, 6), return_counts=True)
        return MENU[int(np.abs(MENU - vals[counts.argmax()]).argmin())]

    def best_grid(idx):
        k = int(Lm[idx].mean(axis=0).argmin())
        return ga[k], gp[k]

    trees = {}                                                                             # (kind, teacher, depth) -> tree with actions
    for name, (a_tr, p_tr, _, _) in teachers.items():
        for d in DEPTHS:
            part = partition_from_teacher(Xtr, a_tr, p_tr, d)
            trees[("imitation", name, d)] = set_actions(part, Xtr, lambda idx: (a_tr[idx].mean(), mode(p_tr[idx])))
            trees[("adjusted", name, d)] = set_actions(part, Xtr, best_grid)
    for d in DEPTHS:
        trees[("direct", "none", d)] = grow_direct(Xtr, Lm, ga, gp, np.arange(len(Xtr)), d)
    keys = list(trees)
    act = [actions(trees[k], Xte) for k in keys]
    # exact value of every tree's choice (best consumption afterwards), on the held-out couples
    laws = {b: law(b, AGE_LIMIT) for b in ("TRUE", "POP")}
    _, dte_frame, *_ = held_out()
    W = value_at_actions(par, laws, horizon(dte_frame), np.stack([a for a, _ in act], axis=1), np.stack([p for _, p in act], axis=1))
    loss = 100 * wealth_equivalent_loss(W, best[:, None], par)                             # (held-out couples, trees)

    idx_b = np.random.default_rng(7).integers(0, len(best), (2000, len(best)))
    stats = lambda x: dict(mean=float(x.mean()), ci=[float(np.percentile(x[idx_b].mean(axis=1), 2.5)), float(np.percentile(x[idx_b].mean(axis=1), 97.5))],
                           median=float(np.median(x)), q90=float(np.percentile(x, 90)), max=float(x.max()))
    out = {"features": FEATURES, "min_leaf": MIN_LEAF, "depths": list(DEPTHS), "shown_depth": SHOWN, "couples": int(len(best)),
           "train_couples": int(len(Xtr)), "grid_actions": int(len(ga)), "teachers": {}, "direct": {}}
    per_couple = {}
    for c, (kind, name, d) in enumerate(keys):
        tree = trees[(kind, name, d)]
        a_hat, p_hat = act[c]
        entry = dict(leaves=len(leaves(tree, Xte)), used=sorted(used(tree)), loss=stats(loss[:, c]), alpha_mean=float(100 * a_hat.mean()),
                     optimum_mae_alpha=float(100 * np.abs(a_opt - a_hat).mean()), optimum_same_phi=float(np.mean(np.isclose(p_opt, p_hat))))
        if kind != "imitation":                                                            # leaves on the grid: loss on the training couples too
            a_in, p_in = actions(tree, Xtr)
            k_in = np.array([int(np.where(np.isclose(ga, a) & np.isclose(gp, p))[0][0]) for a, p in zip(a_in, p_in)])
            entry["loss_train"] = float(Lm[np.arange(len(Xtr)), k_in].mean())
        if kind == "imitation":
            _, _, a_te, p_te = teachers[name]
            entry.update(fidelity_r2_alpha=float(1.0 - ((a_te - a_hat) ** 2).sum() / ((a_te - a_te.mean()) ** 2).sum()),
                         fidelity_mae_alpha=float(100 * np.abs(a_te - a_hat).mean()), fidelity_same_phi=float(np.mean(np.isclose(p_te, p_hat))))
        if d <= 4:
            entry["tree"] = describe(tree, Xte, loss[:, c])
        if kind == "direct":
            out["direct"][str(d)] = entry
        else:
            out["teachers"].setdefault(name, {}).setdefault(str(d), {})[kind] = entry
        per_couple[f"{kind}|{name}|{d}|alpha"], per_couple[f"{kind}|{name}|{d}|phi"], per_couple[f"{kind}|{name}|{d}|loss"] = a_hat, p_hat, loss[:, c]
    # the teachers' own choices and reference households, same measure (annuity choice followed by the best consumption)
    q = np.load("results_rl/agents_per_couple.npz")
    choice = lambda v: stats(100 * wealth_equivalent_loss(v, best, par))
    out["references"] = {"POP": choice(ref["true_action|POP"]), "M0": choice(ref["true_action|M0"]), "MARG": choice(ref["true_action|MARG"])}
    for name in teachers:
        if name != "exact" and f"{name}|gap_choice" in q.files:
            out["references"][name] = stats(q[f"{name}|gap_choice"])
    out["grid_best_loss_train"] = float(Lm.min(axis=1).mean())                            # what the grid itself loses (5 % steps)
    json.dump(out, open("results_xai/xai_tree.json", "w"), indent=1)
    np.savez("results_xai/trees_per_couple.npz", **per_couple)
    for name in teachers:
        for kind in ("imitation", "adjusted"):
            print(f"{name:10s} {kind:9s}", " ".join(f"d{d}: {out['teachers'][name][str(d)][kind]['loss']['mean']:.3f}" for d in DEPTHS))
    print(f"{'direct':20s}", " ".join(f"d{d}: {out['direct'][str(d)]['loss']['mean']:.3f}" for d in DEPTHS))
    print("references", {k: round(v["mean"], 3) for k, v in out["references"].items()})
