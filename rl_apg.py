"""
rl_apg.py -- an agent that learns by gradient through the simulator (differentiable simulation).

Each training step draws lives of training couples from the simulator. On every life the four
survivor shares on offer are tried with the same dates of death: the couple annuitises the share
alpha given by the network for that survivor share, then follows the consumption network. The
discounted utility of a life is differentiable in the network weights, because the budget equation
is known; the dates of death are random and are only sampled. The survivor share is learned from the
same lives: for each share the network estimates, by regression, how much better or worse it does
than the average of the four on the same life, and the share with the best estimate is chosen.

The agent uses simulated lives only: nothing from the exact solution, nothing from the exact laws.
During training, the exact gap to the optimum is printed for 300 training couples (rl_core.monitoring_set);
it is never used to choose the weights (the last weights are kept). The held-out couples are not
looked at: they are only used afterwards, by evaluate_agents.py.

Usage: python rl_apg.py <seed> [steps] [lives per step] [hidden width]
Output: results_rl/apg_seed<seed>.npz (weights) and .json (learning curve); the folder can be changed
        with the environment variable CARUP_RL_OUT.
"""
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from decision_dp import Params, wealth_equivalent_loss
from rl_core import (build, sample_life, life_return, init_policy, mlp, adam_init, adam_update, save_params,
                     make_exact_evaluator, monitoring_set, MENU, RULE)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
LIVES = int(sys.argv[3]) if len(sys.argv) > 3 else 256
WIDTH = int(sys.argv[4]) if len(sys.argv) > 4 else 64
LR0, LR1, CLIP, EVAL_EVERY, Q_WEIGHT = 1e-3, 1e-5, 1.0, 500, 1.0
OUT = os.environ.get("CARUP_RL_OUT", "results_rl")
os.makedirs(OUT, exist_ok=True)

par = Params()
S = build(par)
train, sim = S["train"], S["sim"]
n_train = train["feat"].shape[0]
menu = jnp.asarray(MENU)


def objective(params, key):
    k1, k2 = jax.random.split(key)
    n = jax.random.randint(k1, (LIVES,), 0, n_train)
    K = jax.vmap(lambda k, i: sample_life(k, i, sim, train))(jax.random.split(k2, LIVES), n)

    def one_life(i, Ki):
        out = mlp(params["ann"], train["feat"][i])
        alphas, q = jax.nn.sigmoid(out[:len(MENU)]), out[len(MENU):]
        R = jax.vmap(lambda a, phi: life_return(params, par, train, i, Ki, a, phi))(alphas, menu)     # the four shares, same life
        advantage = jax.lax.stop_gradient(R - jnp.mean(R))                # how each share did on this life, against the four
        return jnp.mean(R) - Q_WEIGHT * jnp.mean((q - advantage) ** 2), R[jnp.argmax(q)]

    obj, chosen = jax.vmap(one_life)(n, K)
    return -jnp.mean(obj) / 10.0, jnp.mean(chosen)


@jax.jit
def train_step(params, opt, key, lr):
    (loss, chosen), grads = jax.value_and_grad(objective, has_aux=True)(params, key)
    params, opt = adam_update(params, grads, opt, lr, max_norm=CLIP)
    return params, opt, loss, chosen


# exact evaluation on 300 training couples whose optimum is known (never on the held-out couples)
mon, mon_group, mon_both, mon_only, mon_best = monitoring_set(S)
evaluate = make_exact_evaluator(par)


def monitor(params):
    alpha, k, val = evaluate(params, mon_group, mon_both, mon_only)
    gap = 100 * wealth_equivalent_loss(val, mon_best["value"], par)
    return dict(gap_mean=float(gap.mean()), gap_median=float(np.median(gap)), gap_max=float(gap.max()),
                alpha_mean=float(alpha.mean()), same_share=float(np.mean(np.isclose(MENU[k], mon_best["phi"]))))


params = init_policy(jax.random.PRNGKey(seed), hidden=(WIDTH, WIDTH))
opt = adam_init(params)
key = jax.random.PRNGKey(1000 + seed)
curve, t0, run_obj = [], time.time(), []
for step in range(STEPS + 1):
    if step % EVAL_EVERY == 0 or step == STEPS:
        m = dict(step=step, lives=step * LIVES, seconds=time.time() - t0, **monitor(params))
        curve.append(m)
        print(f"step {step:5d} ({m['lives']:8d} lives, {m['seconds']:5.0f}s): gap to the optimum on {len(mon)} training couples: "
              f"mean {m['gap_mean']:.3f} %, median {m['gap_median']:.3f} %, max {m['gap_max']:.2f} %; alpha {m['alpha_mean']:.3f}; "
              f"same survivor share as the optimum {m['same_share']:.2f}"
              + (f"; training objective {np.mean(run_obj):.3f}" if run_obj else ""), flush=True)
        run_obj = []
    if step == STEPS:
        break
    key, sub = jax.random.split(key)
    lr = LR1 + 0.5 * (LR0 - LR1) * (1 + np.cos(np.pi * step / STEPS))
    params, opt, loss, chosen = train_step(params, opt, sub, lr)
    run_obj.append(float(chosen))
save_params(f"{OUT}/apg_seed{seed}.npz", params, width=WIDTH)
json.dump(dict(seed=seed, steps=STEPS, lives_per_step=LIVES, width=WIDTH, lr=[LR0, LR1], clip=CLIP, q_weight=Q_WEIGHT, rule=RULE,
               monitoring="300 training couples", seconds=time.time() - t0, curve=curve), open(f"{OUT}/apg_seed{seed}.json", "w"), indent=2)
print(f"saved {OUT}/apg_seed{seed}.npz in {time.time() - t0:.0f}s")
