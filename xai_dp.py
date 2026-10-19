"""
xai_dp.py -- exact optimum (dynamic programming, full model, limiting age 110) for couples other than
the held-out ones, and exact value of the agents on the same couples.

    python xai_dp.py train      the 3 550 training couples       -> results_xai/optimum_train.npz
    python xai_dp.py hybrids    hybrid couples built from pairs of held-out couples
                                                                 -> results_xai/hybrids.npz
    python xai_dp.py effects    the couples i of the pairs, with one characteristic changed by a stated
                                amount (xai_setup.CHANGE), 13 changed couples per couple
                                                                 -> results_xai/effects.npz
    python xai_dp.py grid       value of every action of a grid (annuitised share by steps of 5 %,
                                the four survivor shares, and no annuity), followed by the best
                                consumption, for the training couples -> results_xai/grid_train.npz

Hybrid couples. N_PAIRS pairs (i, j) of held-out couples are drawn. For each pair:
    - the 32 coalitions of the 5 groups of characteristics (xai_setup.GROUPS): the groups of the
      coalition come from i, the others from j (coalition 31 is couple i, coalition 0 is couple j);
    - 13 couples equal to i except for one characteristic, taken from j.
Identical hybrid couples are solved once. Each chunk of couples is cached (results_xai/cache), so an
interrupted run resumes; the agents' part is cached per agent file and recomputed when the weights change.
"""
import glob
import hashlib
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from xai_setup import Setup, group_masks, hybrid, N_RAW
from rl_core import init_policy, load_params, make_exact_evaluator, MENU
from decision_dp import Params, value_at_actions

N_PAIRS, SEED, BATCHES = 400, 2026, 7
CACHE = "results_xai/cache"


def save_npz_atomic(path, **arrays):
    """np.savez under a temporary name, then renamed: a reader never sees a half-written file."""
    tmp = f"{path[:-4]}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def design(Z, n_pairs=N_PAIRS, seed=SEED):
    """Pairs, masks (45, 13) and the hybrid couples; row[p, m] is the position of hybrid m of pair p
    among the distinct hybrid couples Zu."""
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    i = rng.choice(n, n_pairs, replace=False)
    j = (i + rng.integers(1, n, n_pairs)) % n                                   # another couple
    masks = np.concatenate([group_masks(), ~np.eye(N_RAW, dtype=bool)], axis=0)
    allZ = hybrid(Z[i][:, None, :], Z[j][:, None, :], masks[None, :, :]).reshape(-1, N_RAW)
    Zu, row = np.unique(allZ, axis=0, return_inverse=True)
    return i, j, masks, Zu, np.asarray(row).reshape(n_pairs, masks.shape[0])


def agent_files():
    """The gradient agents (all seeds) and, for PPO, the first seed of each treatment of the noise."""
    return sorted(glob.glob("results_rl/apg_seed*.npz") + glob.glob("results_rl/ppoa_seed0.npz") + glob.glob("results_rl/ppo_seed0.npz"))


def fingerprint(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:10]


def cache_key(S: Setup, Z):
    """Identifies the couples and, when it is not the base case, the decision problem. The cache does not
    know the mortality draws: empty results_xai/cache if the models of results_eval are refitted."""
    extra = b"" if S.par == Params() else repr(S.par).encode()
    return hashlib.sha1(np.ascontiguousarray(Z).tobytes() + extra).hexdigest()[:10]


def run(S: Setup, Z, tag, log):
    """Optimum and agents' values for the couples Z, chunk by chunk. Returns a dict of arrays (len(Z),)."""
    os.makedirs(CACHE, exist_ok=True)
    key = cache_key(S, Z)
    agents = {os.path.basename(p)[:-4]: (p, fingerprint(p)) for p in agent_files()}
    evaluate = make_exact_evaluator(S.par)
    out = {k: np.full(len(Z), np.nan) for k in ["alpha", "phi", "value"]}
    for a in agents:
        out.update({f"{a}|alpha": np.full(len(Z), np.nan), f"{a}|phi": np.full(len(Z), np.nan), f"{a}|value": np.full(len(Z), np.nan)})
    H = S.horizons(Z)
    Tb = np.maximum(20, 10 * np.ceil(H / 10.0)).astype(int)
    done = 0
    for T in sorted(set(Tb.tolist())):
        members = np.where(Tb == T)[0]
        nb = int(np.clip(4e8 // (2 * T * T * S.par.Gw * 8), 4, 64))             # batch size of decision_dp.solve_couples
        for c, lo in enumerate(range(0, len(members), BATCHES * nb)):
            idx = members[lo:lo + BATCHES * nb]
            f_dp = f"{CACHE}/{tag}_{key}_T{T}_{c:03d}_dp.npz"
            f_ag = {a: f"{CACHE}/{tag}_{key}_T{T}_{c:03d}_{a}_{fp}.npz" for a, (_, fp) in agents.items()}
            laws = None
            if not os.path.exists(f_dp):
                laws = S.laws(Z[idx])
                alpha, phi, val = S.optimum(Z[idx], laws=laws)
                save_npz_atomic(f_dp, alpha=alpha, phi=phi, value=val)
            z = np.load(f_dp)
            out["alpha"][idx], out["phi"][idx], out["value"][idx] = z["alpha"], z["phi"], z["value"]
            for a, (path, _) in agents.items():
                if not os.path.exists(f_ag[a]):
                    laws = S.laws(Z[idx]) if laws is None else laws
                    width = int(np.load(path)["width"])
                    params = load_params(path, init_policy(jax.random.PRNGKey(0), hidden=(width, width)))
                    aB, aA = S.annuity_values(Z[idx, :2])
                    rB, rS = S.rule_rates(Z[idx, :2])
                    grp = dict(feat=jnp.asarray(S.features(Z[idx])), aB=jnp.asarray(aB), aA=jnp.asarray(aA),
                               rateB=jnp.asarray(rB), rateS=jnp.asarray(rS))
                    al, k, v = evaluate(params, grp, *laws["TRUE"])
                    save_npz_atomic(f_ag[a], alpha=al, phi=MENU[k], value=v)
                z = np.load(f_ag[a])
                out[f"{a}|alpha"][idx], out[f"{a}|phi"][idx], out[f"{a}|value"][idx] = z["alpha"], z["phi"], z["value"]
            done += len(idx)
            log(f"T = {T}: {done} / {len(Z)} couples")
    return out


def action_grid():
    """81 actions: annuitised share 5 %, 10 %, ..., 100 % for each survivor share on offer, and no annuity."""
    a, p = np.meshgrid(np.arange(1, 21) / 20.0, MENU, indexing="ij")
    return np.append(a.ravel(), 0.0), np.append(p.ravel(), MENU[0])


def run_grid(S: Setup, Z, tag, log):
    """Exact value (best consumption afterwards) of every action of the grid for the couples Z: (len(Z), 81)."""
    os.makedirs(CACHE, exist_ok=True)
    key = cache_key(S, Z)
    alpha, phi = action_grid()
    out = np.full((len(Z), len(alpha)), np.nan)
    H = S.horizons(Z)
    Tb = np.maximum(20, 10 * np.ceil(H / 10.0)).astype(int)
    done = 0
    for T in sorted(set(Tb.tolist())):
        members = np.where(Tb == T)[0]
        nb = int(np.clip(4e8 // (2 * T * T * S.par.Gw * 8), 4, 64))
        for c, lo in enumerate(range(0, len(members), BATCHES * nb)):
            idx = members[lo:lo + BATCHES * nb]
            f = f"{CACHE}/{tag}_{key}_T{T}_{c:03d}_grid{len(alpha)}.npz"
            if not os.path.exists(f):
                save_npz_atomic(f, value=value_at_actions(S.par, S.laws(Z[idx]), H[idx], np.tile(alpha, (len(idx), 1)), np.tile(phi, (len(idx), 1))))
            out[idx] = np.load(f)["value"]
            done += len(idx)
            log(f"T = {T}: {done} / {len(Z)} couples")
    return alpha, phi, out


if __name__ == "__main__":
    S = Setup()
    for what in sys.argv[1:] or ["train", "hybrids"]:
        t0 = time.time()
        log = lambda msg: print(f"[{what}] {msg}  {time.time() - t0:.0f}s", flush=True)
        if what == "train":
            out = run(S, S.Z["train"], "train", log)
            # written under another name first: the agents read this file when their training starts
            np.savez("results_xai/optimum_train_new.npz", seconds=time.time() - t0, **out)
            os.replace("results_xai/optimum_train_new.npz", "results_xai/optimum_train.npz")
        elif what == "hybrids":
            i, j, masks, Zu, row = design(S.Z["test"])
            log(f"{N_PAIRS} pairs, {row.size} hybrid couples, {len(Zu)} distinct")
            out = run(S, Zu, "hybrids", log)
            np.savez("results_xai/hybrids.npz", i=i, j=j, masks=masks, Z=Zu, row=row, seconds=time.time() - t0, **out)
        elif what == "effects":
            i, *_ = design(S.Z["test"])
            Zc, sign = S.changed_couples(S.Z["test"][i])
            Zu, row = np.unique(Zc.reshape(-1, N_RAW), axis=0, return_inverse=True)
            log(f"{len(i)} couples, {Zc.shape[0] * Zc.shape[1]} changed couples, {len(Zu)} distinct")
            out = run(S, Zu, "effects", log)
            np.savez("results_xai/effects.npz", i=i, sign=sign, Z=Zu, row=np.asarray(row).reshape(len(i), N_RAW), seconds=time.time() - t0, **out)
        elif what == "grid":
            alpha, phi, val = run_grid(S, S.Z["train"], "train", log)
            np.savez("results_xai/grid_train.npz", alpha=alpha, phi=phi, value=val, seconds=time.time() - t0)
        else:
            raise SystemExit(f"unknown part: {what}")
        log("done")
