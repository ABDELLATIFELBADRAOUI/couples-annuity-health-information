"""
rule_baseline.py -- what the simple consumption rule achieves on its own, before any learning.

The agents' consumption network outputs a correction to a simple rule (rl_core, CARUP_RULE). Here the
correction is set to zero and the rule is followed along every history of each held-out couple, with:
    - the optimal annuity choice of the couple (dynamic programming, results_decision/base_limit.npz);
    - everything annuitised with a 2/3 survivor share.
The loss is the exact wealth-equivalent loss against the optimum, as for the agents.

Usage: python rule_baseline.py        -> results_rl/rule_baseline_<rule>.json
"""
import json

import numpy as np
import jax
import jax.numpy as jnp

from decision_setup import law
from decision_dp import Params, wealth_equivalent_loss
from rl_core import build, init_policy, tree_value, AGE_LIMIT, RULE

par = Params()
S = build(par)
test = S["test"]
both, only = law("TRUE", AGE_LIMIT)
ref = np.load("results_decision/base_limit.npz")
N = len(ref["true|TRUE"])
params = init_policy(jax.random.PRNGKey(0))
w, b = params["cons"][-1]
params["cons"][-1] = (jnp.zeros_like(w), jnp.zeros_like(b))                      # no correction: the rule alone


@jax.jit
def run(feat, aB, aA, rB, rS, bo, on, alpha, phi):
    return jax.vmap(lambda f, ab, aa, rb, rs, x, y, a, p: tree_value(params, par, f, ab, aa, rb, rs, x, y, a, p))(
        feat, aB, aA, rB, rS, bo, on, alpha, phi)


def exact_value(alpha, phi, chunk=64):
    out = []
    for lo in range(0, N, chunk):
        idx = np.minimum(np.arange(lo, lo + chunk), N - 1)
        v = run(test["feat"][idx], test["aB"][idx], test["aA"][idx], test["rateB"][idx], test["rateS"][idx],
                jnp.asarray(both[idx]), jnp.asarray(only[idx]), jnp.asarray(alpha[idx]), jnp.asarray(phi[idx]))
        out.append(np.asarray(v)[:min(chunk, N - lo)])
    return np.concatenate(out)


def stats(x, seed=7):
    idx = np.random.default_rng(seed).integers(0, x.size, (2000, x.size))
    m = x[idx].mean(axis=1)
    return dict(mean=float(x.mean()), ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], median=float(np.median(x)),
                q90=float(np.percentile(x, 90)), max=float(x.max()), min=float(x.min()))


assert (ref["alpha|TRUE"] > 0).all(), "a couple without annuity has no survivor share: handle it before using this script"
best = ref["true|TRUE"]
out = dict(rule=RULE, couples=int(N),
           optimal_annuity=stats(100 * wealth_equivalent_loss(exact_value(ref["alpha|TRUE"], ref["phi|TRUE"]), best, par)),
           all_in_two_thirds=stats(100 * wealth_equivalent_loss(exact_value(np.ones(N), np.full(N, 2.0 / 3.0)), best, par)))
json.dump(out, open(f"results_rl/rule_baseline_{RULE}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
