"""
xai_shap.py -- which characteristics of the couple drive the annuity decision, for the learned
policies and for the exact policy.

The decision explained is the pair (alpha, phi): share of the wealth converted into the annuity and
survivor share chosen. Four analyses, all without sampling of coalitions (see xai_setup):

A. Gradient agents, 13 characteristics, every held-out couple. Shapley values of the decision of
   couple i against every other held-out couple j, averaged over j: the usual SHAP values with an
   independent background, the background being the whole held-out sample (no sampling at all).
   Characteristics equal in i and j contribute nothing and are left out of the enumeration of the
   pair (it does not change the values of the others); the values of the pair (j, i) are the
   opposite of those of (i, j), so each pair is enumerated once.
B. Exact policy and agents, 5 groups of characteristics, the pairs of xai_dp.py. For each pair the
   32 hybrid couples are solved by dynamic programming, so the Shapley values of the exact policy
   are exact as well. Same pairs, same hybrid couples for the agents: like-for-like comparison.
C. Exact policy and agents, one characteristic exchanged: couple i receives one characteristic of
   couple j, everything else unchanged (same pairs). Weight of the characteristic: mean absolute
   change of the decision.
D. Exact policy and agents, effect of a stated change of one characteristic (xai_setup.CHANGE:
   +5 years, yes instead of no, ...), everything else unchanged, on the couples i of the pairs
   (xai_dp.py effects).

Usage: python xai_shap.py            (needs results_xai/hybrids.npz for B and C, results_xai/effects.npz for D)
Output: results_xai/shap_<agent>.npz (A; the values of all pairs are cached in results_xai/cache),
        results_xai/xai_shap.json
"""
import glob
import hashlib
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from xai_setup import (Setup, RAW, GROUPS, OWN_H, OWN_W, N_RAW, N_GROUPS, coalitions, shapley_matrix, group_masks)
from rl_core import init_policy, load_params, annuity_choice, MENU

ROWS = 1 << 17                      # hybrid couples evaluated per call
N_BOOT, SEED_BOOT = 2000, 7
A0 = 50                             # first age of the table of annuity values


# ----------------------------------------------------------------------------- agents as functions of the 13 characteristics
def agent_files():
    """The gradient agents (all seeds) and, for PPO, the first seed of each treatment of the noise."""
    return sorted(glob.glob("results_rl/apg_seed*.npz") + glob.glob("results_rl/ppoa_seed0.npz") + glob.glob("results_rl/ppo_seed0.npz"))


def load_agent(path):
    width = int(np.load(path)["width"])
    return load_params(path, init_policy(jax.random.PRNGKey(0), hidden=(width, width)))


def annuity_table(S: Setup):
    """log(aB / 10), log(aA / 5) for every pair of ages (the two inputs of the networks that depend on the ages only)."""
    amax = int(max(S.Z["train"][:, :2].max(), S.Z["test"][:, :2].max()))
    n = amax - A0 + 1
    grid = np.array([(h, w) for h in range(A0, amax + 1) for w in range(A0, amax + 1)])
    aB, aA = S.annuity_values(grid)
    return jnp.asarray(np.stack([np.log(aB / 10.0), np.log(aA / 5.0)], axis=1).reshape(n, n, 2))


def decide(params, z, table):
    """Decision (alpha, phi) of an agent for couples given by their characteristics z (n, 13)."""
    ih, iw = (z[:, 0] - A0).astype(jnp.int32), (z[:, 1] - A0).astype(jnp.int32)
    feat = jnp.concatenate([(z[:, :2] - 65.0) / 10.0, z[:, jnp.asarray(OWN_H)], z[:, jnp.asarray(OWN_W)], table[ih, iw]], axis=1)
    alpha, k = jax.vmap(lambda f: annuity_choice(params, f))(feat)
    return jnp.stack([alpha, jnp.asarray(MENU)[k]], axis=1)


_DECIDE = jax.jit(decide)


def decisions(params, Z, table, chunk=65536):
    return np.concatenate([np.asarray(_DECIDE(params, jnp.asarray(Z[lo:lo + chunk]), table)) for lo in range(0, len(Z), chunk)], axis=0)


# ----------------------------------------------------------------------------- A. exact enumeration for pairs of couples
_PAIR_FN = {}


def _pair_fn(m):
    """Shapley values of pairs whose couples differ in exactly m characteristics (positions idx)."""
    if m not in _PAIR_FN:
        Mm, Cm = jnp.asarray(coalitions(m)), jnp.asarray(shapley_matrix(m))

        def one(params, table, a, b, idx):
            mask = jnp.zeros((2 ** m, N_RAW), dtype=bool).at[:, idx].set(Mm)
            v = decide(params, jnp.where(mask, a, b), table)                 # (2^m, 2): every coalition of the m characteristics
            return jnp.zeros((N_RAW, 2)).at[idx].set(Cm @ v)

        _PAIR_FN[m] = jax.jit(jax.vmap(one, in_axes=(None, None, 0, 0, 0)))
    return _PAIR_FN[m]


def pair_shapley(params, Zi, Zj, table):
    """(P, 13, 2): Shapley values of decision(Zi[p]) - decision(Zj[p]) over the 13 characteristics."""
    diff = Zi != Zj
    m_all = diff.sum(axis=1)
    out = np.zeros((len(Zi), N_RAW, 2))
    for m in range(1, N_RAW + 1):
        rows = np.where(m_all == m)[0]
        if not len(rows):
            continue
        idx = np.argsort(~diff[rows], axis=1, kind="stable")[:, :m]         # the m differing characteristics
        f, nb = _pair_fn(m), max(1, ROWS >> m)
        for lo in range(0, len(rows), nb):
            sel = np.minimum(np.arange(lo, lo + nb), len(rows) - 1)          # pad the last batch
            res = np.asarray(f(params, table, jnp.asarray(Zi[rows[sel]]), jnp.asarray(Zj[rows[sel]]), jnp.asarray(idx[sel])))
            n = min(nb, len(rows) - lo)
            out[rows[lo:lo + n]] = res[:n]
    return out


def agent_shap(S, path, table, log):
    """Shapley values of every ordered pair (i explained, j reference) of held-out couples, and their
    average over j (the SHAP values). Returns (dict of arrays, pair values (N, N, 13, 2) float32)."""
    name = os.path.basename(path)[:-4]
    Z = S.Z["test"]
    N = len(Z)
    fp = hashlib.sha1(open(path, "rb").read()).hexdigest()[:10] + "_" + hashlib.sha1(np.ascontiguousarray(Z).tobytes()).hexdigest()[:6]
    out, cache = f"results_xai/shap_{name}.npz", f"results_xai/cache/pairs_{name}_{fp}.npy"      # agent weights and couples
    if os.path.exists(out) and os.path.exists(cache) and str(np.load(out)["fingerprint"]) == fp:
        return dict(np.load(out)), np.load(cache, mmap_mode="r")
    t0 = time.time()
    params = load_agent(path)
    iu, ju = np.triu_indices(N, k=1)
    upper = pair_shapley(params, Z[iu], Z[ju], table)
    pairs = np.zeros((N, N, N_RAW, 2), dtype=np.float32)
    pairs[iu, ju], pairs[ju, iu] = upper, -upper
    shap, abs_pair = np.zeros((N, N_RAW, 2)), np.zeros((N, N_RAW, 2))
    np.add.at(shap, iu, upper), np.add.at(shap, ju, -upper)
    np.add.at(abs_pair, iu, np.abs(upper)), np.add.at(abs_pair, ju, np.abs(upper))
    res = dict(shap=shap / N, abs_pair=abs_pair / N, decision=decisions(params, Z, table), fingerprint=np.array(fp),
               seconds=time.time() - t0)
    os.makedirs("results_xai/cache", exist_ok=True)
    np.save(cache, pairs)
    np.savez(out, **res)
    log(f"{name}: Shapley values of {len(iu)} pairs of couples in {time.time() - t0:.0f}s")
    return res, pairs


def importance_bootstrap(pairs, n_boot, seed, block=250):
    """Mean absolute SHAP value of each characteristic when the couples are resampled (as explained
    couples and as reference couples at the same time). Returns (n_boot, 13, 2)."""
    N = pairs.shape[0]
    rng = np.random.default_rng(seed)
    P = np.ascontiguousarray(np.transpose(np.asarray(pairs).reshape(N, N, -1), (0, 2, 1))).reshape(-1, N)     # (N * 26, N)
    out = []
    for lo in range(0, n_boot, block):
        c = rng.multinomial(N, np.full(N, 1.0 / N), size=min(block, n_boot - lo)).astype(np.float32)          # (b, N) counts
        sh = (P @ c.T).reshape(N, -1, c.shape[0]) / N                                                         # SHAP values of each resample
        out.append(np.einsum("ikb,bi->bk", np.abs(sh), c) / N)
    return np.concatenate(out, axis=0).reshape(n_boot, N_RAW, 2)


def group_weights_all_pairs(params, Z, table, chunk=4096):
    """Mean absolute Shapley value of each group over EVERY pair of held-out couples, for an agent:
    (5, 2), in points. It tells whether the pairs drawn for the exact policy are representative."""
    GM, C5 = jnp.asarray(group_masks()), jnp.asarray(shapley_matrix(N_GROUPS))

    @jax.jit
    def f(params, zi, zj):
        zh = jnp.where(GM[None], zi[:, None, :], zj[:, None, :])                       # (pairs, 32, 13)
        v = decide(params, zh.reshape(-1, N_RAW), table).reshape(zi.shape[0], GM.shape[0], 2)
        return jnp.abs(jnp.einsum("gc,pco->pgo", C5, v))

    iu, ju = np.triu_indices(len(Z), k=1)
    total = np.zeros((N_GROUPS, 2))
    for lo in range(0, len(iu), chunk):
        sel = np.minimum(np.arange(lo, lo + chunk), len(iu) - 1)                       # pad the last batch
        r = np.asarray(f(params, jnp.asarray(Z[iu[sel]]), jnp.asarray(Z[ju[sel]])))
        total += r[:min(chunk, len(iu) - lo)].sum(axis=0)
    return 100 * total / len(iu)


# ----------------------------------------------------------------------------- statistics
def boot_index(n):
    return np.random.default_rng(SEED_BOOT).integers(0, n, (N_BOOT, n))


def mean_ci(x, idx):
    m = x[idx].mean(axis=1)
    return dict(mean=float(x.mean()), ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))])


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def corr_ci(a, b, idx):
    c = np.array([corr(a[r], b[r]) for r in idx])
    return dict(value=corr(a, b), ci=[float(np.nanpercentile(c, 2.5)), float(np.nanpercentile(c, 97.5))])


def ranks(x):
    return np.argsort(np.argsort(-np.asarray(x)))


def pairs_analysis(h, S, table, files, names, summary, log=lambda msg: None):
    """Parts B and C: exact policy and agents on the pairs of xai_dp.py (h: content of hybrids.npz)."""
    Z, OUT = S.Z["test"], ("alpha", "phi")
    row, Zu, pi, pj = h["row"], h["Z"], h["i"], h["j"]
    P = len(pi)
    idx = boot_index(P)
    dec = {"exact": np.stack([h["alpha"], h["phi"]], axis=1)}
    for path, name in zip(files, names):
        dec[name] = decisions(load_agent(path), Zu, table)
        stored = np.stack([h[f"{name}|alpha"], h[f"{name}|phi"]], axis=1) if f"{name}|alpha" in h else None
        if stored is not None:
            summary.setdefault("check_hybrids", {})[name] = float(np.nanmax(np.abs(stored - dec[name])))  # same decisions as in xai_dp.py
    C5 = shapley_matrix(N_GROUPS)
    vals = {k: 100 * v[row] for k, v in dec.items()}                                   # (P, 45, 2), in points
    ok_phi = ~np.isnan(vals["exact"][:, :, 1]).any(axis=1)                             # pairs where every hybrid couple buys an annuity
    summary["pairs"] = int(P)
    summary["pairs_with_phi"] = int(ok_phi.sum())
    summary["hybrids_no_annuity"] = int(np.isnan(h["phi"]).sum())
    summary["hybrids_distinct"] = int(len(Zu))
    summary["pairs_distinct_couples"] = int(len(np.unique(np.concatenate([pi, pj]))))   # the pairs share some couples
    n_coal = 2.0 ** (Z[pi] != Z[pj]).sum(axis=1)                                        # hybrid couples a pair would need at the level of the 13 characteristics
    summary["pairs_coalitions_13"] = {"mean": float(n_coal.mean()), "max": float(n_coal.max()), "total": float(n_coal.sum())}
    B, Cc = {}, {}
    for o, out_name in enumerate(OUT):
        keep = np.arange(P) if o == 0 else np.where(ok_phi)[0]
        bidx = idx if o == 0 else boot_index(len(keep))
        sv = {k: np.einsum("gc,pc->pg", C5, v[keep][:, :32, o]) for k, v in vals.items()}       # (pairs, 5)
        e = {"n_pairs": int(len(keep)),
             "efficiency_error": float(max(np.abs(sv[k].sum(axis=1) - (vals[k][keep][:, 31, o] - vals[k][keep][:, 0, o])).max() for k in sv)),
             "difference_mean_abs": {k: mean_ci(np.abs(vals[k][keep][:, 31, o] - vals[k][keep][:, 0, o]), bidx) for k in sv}}
        for k in sv:
            imp = np.abs(sv[k]).mean(axis=0)
            e[k] = {"mean_abs": {g: mean_ci(np.abs(sv[k][:, gi]), bidx) for gi, (g, _) in enumerate(GROUPS)},
                    "share": {g: float(imp[gi] / imp.sum()) for gi, (g, _) in enumerate(GROUPS)}}
            if k != "exact":
                e[k]["corr"] = {g: corr_ci(sv[k][:, gi], sv["exact"][:, gi], bidx) for gi, (g, _) in enumerate(GROUPS)}
                e[k]["mean_abs_diff"] = {g: mean_ci(np.abs(sv[k][:, gi] - sv["exact"][:, gi]), bidx) for gi, (g, _) in enumerate(GROUPS)}
                e[k]["corr_all"] = corr_ci(sv[k].ravel(), sv["exact"].ravel(), np.repeat(bidx, N_GROUPS, axis=1) * N_GROUPS + np.tile(np.arange(N_GROUPS), bidx.shape[1])[None, :])
                e[k]["mean_abs_diff_all"] = float(np.abs(sv[k] - sv["exact"]).mean())
        B[out_name] = e
        # C. one characteristic at a time: couple i receives characteristic c of couple j
        ec = {"n_pairs": int(len(keep))}
        for c, key in enumerate(RAW):
            d = {k: v[keep][:, 32 + c, o] - v[keep][:, 31, o] for k, v in vals.items()}  # change of the decision of couple i
            moved = (Z[pj] != Z[pi])[keep, c]                                            # pairs where the characteristic differs
            item = {"share_different": float(moved.mean())}
            for k in d:
                item[k] = {"mean_abs": mean_ci(np.abs(d[k]), bidx),                      # over all pairs: zero where it does not differ
                           "share_nonzero": float(np.mean(d[k] != 0))}
                if k != "exact":
                    item[k]["corr"] = corr(d[k][moved], d["exact"][moved]) if moved.sum() > 2 else float("nan")
                    item[k]["mean_abs_diff"] = float(np.abs(d[k] - d["exact"]).mean())
            ec[key] = item
        Cc[out_name] = ec
    summary["B"], summary["C"] = B, Cc
    # the same weights for the gradient agents over every pair of held-out couples (are the drawn pairs representative?)
    summary["B_all_pairs"] = {}
    for path, name in zip(files, names):
        if name.startswith("apg"):
            w = group_weights_all_pairs(load_agent(path), Z, table)
            summary["B_all_pairs"][name] = {out_name: {g: float(w[gi, o]) for gi, (g, _) in enumerate(GROUPS)} for o, out_name in enumerate(OUT)}
    # exact gap of the agents on the hybrid couples (their exact value comes from xai_dp.py)
    g = S.par.g
    hyb = np.unique(row[:, :32])
    real = np.unique(np.concatenate([row[:, 0], row[:, 31]]))
    only_h = np.setdiff1d(hyb, real)
    summary["gap_hybrids"] = {"hybrid_couples": int(len(only_h)), "real_couples": int(len(real))}
    for name in names:
        if f"{name}|value" in h:
            gap = 100 * (1.0 - (h[f"{name}|value"] / h["value"]) ** (1.0 / (1.0 - g)))
            summary["gap_hybrids"][name] = {"hybrid": float(gap[only_h].mean()), "real": float(gap[real].mean()),
                                            "hybrid_median": float(np.median(gap[only_h])), "hybrid_q99": float(np.percentile(gap[only_h], 99))}
    log("B and C done")


def effects_analysis(e, S, table, files, names, summary, log=lambda msg: None):
    """Part D: effect on the decision of a stated change of one characteristic (xai_setup.CHANGE),
    everything else unchanged, for the couples of results_xai/effects.npz: exact policy and agents.
    For the agents, the same effect is also computed on every held-out couple."""
    Z, OUT = S.Z["test"], ("alpha", "phi")
    i, sign, row = e["i"], e["sign"], e["row"]
    ref = np.load("results_decision/base_limit.npz")
    base = {"exact": np.stack([ref["alpha|TRUE"][i], ref["phi|TRUE"][i]], axis=1)}
    changed = {"exact": np.stack([e["alpha"], e["phi"]], axis=1)[row]}                   # (couples, 13, 2)
    Zc_all, sign_all = S.changed_couples(Z)
    everyone = {}
    for path, name in zip(files, names):
        params = load_agent(path)
        base[name] = decisions(params, Z[i], table)
        changed[name] = decisions(params, e["Z"], table)[row]
        if f"{name}|alpha" in e:
            stored = np.stack([e[f"{name}|alpha"], e[f"{name}|phi"]], axis=1)
            summary.setdefault("check_effects", {})[name] = float(np.nanmax(np.abs(stored - decisions(params, e["Z"], table))))
        d_all = decisions(params, Zc_all.reshape(-1, N_RAW), table).reshape(len(Z), N_RAW, 2) - decisions(params, Z, table)[:, None, :]
        everyone[name] = 100 * (sign_all[:, :, None] * d_all).mean(axis=0)               # (13, 2)
    eff = {k: 100 * sign[:, :, None] * (changed[k] - base[k][:, None, :]) for k in base}  # effect of the stated increase, in points
    D = {"couples": int(len(i)), "share_raised": dict(zip(RAW, (sign > 0).mean(axis=0).tolist())),
         "wealth_not_positive": int((S.natural["test"][i, RAW.index("wealth")] <= 0).sum())}
    for o, out_name in enumerate(OUT):
        ok = ~np.isnan(eff["exact"][:, :, o]).any(axis=1)                                # the survivor share is defined before and after
        bidx = boot_index(int(ok.sum()))
        item = {"n_couples": int(ok.sum())}
        for c, key in enumerate(RAW):
            entry = {}
            for k in eff:
                v = eff[k][ok, c, o]
                entry[k] = mean_ci(v, bidx)
                entry[k]["share_nonzero"] = float(np.mean(v != 0))                       # couples whose decision changes
                if k != "exact":
                    entry[k]["corr"] = corr(v, eff["exact"][ok, c, o])                   # do the same couples react more?
                    entry[k]["mean_abs_diff"] = float(np.abs(v - eff["exact"][ok, c, o]).mean())
                    entry[k]["all_couples"] = float(everyone[k][c, o])
            item[key] = entry
        D[out_name] = item
    summary["D"] = D
    log("D done")


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    t_start = time.time()
    log = lambda msg: print(f"[xai_shap] {msg}  {time.time() - t_start:.0f}s", flush=True)
    S = Setup()
    Z = S.Z["test"]
    table = annuity_table(S)
    files = agent_files()
    names = [os.path.basename(p)[:-4] for p in files]
    OUT = ("alpha", "phi")
    summary = {"characteristics": RAW, "groups": [g for g, _ in GROUPS], "agents": names, "couples": int(len(Z)),
               "pairs_A": int(len(Z) * (len(Z) - 1) // 2), "units": "points de pourcentage", "n_boot": N_BOOT}

    # ---------------- A. SHAP values of the gradient agents, 13 characteristics
    iu, ju = np.triu_indices(len(Z), k=1)
    summary["A_coalitions_evaluated"] = float((2.0 ** (Z[iu] != Z[ju]).sum(axis=1)).sum())     # per agent, with the two shortcuts
    summary["A_coalitions_full"] = float(len(Z)) * len(Z) * 2 ** N_RAW                           # every ordered pair, 8 192 coalitions
    del iu, ju
    A = {}
    for path, name in zip(files, names):
        if not name.startswith("apg"):
            continue
        r, pairs = agent_shap(S, path, table, log)
        A[name] = r
        shap, dec = 100 * r["shap"], 100 * r["decision"]
        eff = np.abs(shap.sum(axis=1) - (dec - dec.mean(axis=0))).max()                 # the values add up to decision - average decision
        boot = 100 * importance_bootstrap(pairs, N_BOOT, SEED_BOOT)
        entry = {"efficiency_error": float(eff), "seconds": float(r["seconds"]), "decision_mean": dec.mean(axis=0).tolist(),
                 "decision_sd": dec.std(axis=0).tolist()}
        for o, out_name in enumerate(OUT):
            imp = np.abs(shap[:, :, o]).mean(axis=0)
            entry[out_name] = {
                "mean_abs": dict(zip(RAW, imp.tolist())), "share": dict(zip(RAW, (imp / imp.sum()).tolist())),
                "mean_abs_ci": {k: [float(np.percentile(boot[:, c, o], 2.5)), float(np.percentile(boot[:, c, o], 97.5))] for c, k in enumerate(RAW)},
                "mean_abs_pair": dict(zip(RAW, (100 * r["abs_pair"][:, :, o]).mean(axis=0).tolist())),
                "direction": {k: corr(Z[:, c], shap[:, c, o]) for c, k in enumerate(RAW)},  # sign of the link characteristic -> its value
                "group_sum_mean_abs": {g: float(np.abs(shap[:, members, o].sum(axis=1)).mean()) for g, members in GROUPS}}
        summary.setdefault("A", {})[name] = entry
        del pairs
    # agreement between agents (same kind and across kinds)
    summary["A_between_agents"] = {}
    explained = list(A)
    for a in range(len(explained)):
        for b in range(a + 1, len(explained)):
            e = {}
            for o, out_name in enumerate(OUT):
                x, y = A[explained[a]]["shap"][:, :, o], A[explained[b]]["shap"][:, :, o]
                ia, ib = np.abs(x).mean(axis=0), np.abs(y).mean(axis=0)
                e[out_name] = dict(corr_values=corr(x.ravel(), y.ravel()), spearman_importance=corr(ranks(ia).astype(float), ranks(ib).astype(float)))
            summary["A_between_agents"][f"{explained[a]}|{explained[b]}"] = e
    log("A done")

    # ---------------- B and C: exact policy and agents on the pairs of xai_dp.py
    if os.path.exists("results_xai/hybrids.npz"):
        pairs_analysis(np.load("results_xai/hybrids.npz"), S, table, files, names, summary, log)
    else:
        log("results_xai/hybrids.npz not found: parts B and C skipped")
    if os.path.exists("results_xai/effects.npz"):
        effects_analysis(np.load("results_xai/effects.npz"), S, table, files, names, summary, log)
    else:
        log("results_xai/effects.npz not found: part D skipped")
    json.dump(summary, open("results_xai/xai_shap.json", "w"), indent=1)
    log("results_xai/xai_shap.json written")
