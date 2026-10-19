"""
rl_ppo.py -- a model-free agent: proximal policy optimisation (PPO, Schulman et al. 2017).

The agent only sees observations and rewards from simulated lives; it does not use the budget
equation, the mortality model or the exact solution. Each iteration simulates whole lives of
training couples:
    annuity decision : survivor share drawn from a categorical law, alpha = sigmoid(Gaussian)
    every year alive : share of cash consumed = sigmoid(Gaussian)
Clipped surrogate objective, generalised advantage estimation, separate value networks, Adam.
The annuity decision and year 0 happen at the same date, so no discount is applied between them;
between years the discount is the one of the problem (par.beta).

Rewards used for learning: scaled utility of the year (see rl_core) divided by 5 and floored at -50,
to keep the first iterations stable. The reported results never use these rewards: the learned
policy is evaluated exactly, with the true utility.

Exploration noise (standard deviations of the two Gaussians, on the logit scale):
    learned  (default) : two parameters, started at exp(-0.5), learned with the policy (usual PPO);
    annealed           : fixed by a schedule, from exp(-0.5) at the first iteration down linearly (in
                         log) to exp(final) at the last one. The policy that is evaluated is the mean
                         action, without noise, so noise that is still large at the end of training
                         means that the evaluated policy is not the one that was optimised.

Usage: python rl_ppo.py <seed> [iterations] [lives per iteration] [hidden width] [evaluation every] [final log-std]
       (final log-std given: annealed noise, files ppoa_seed<seed>; omitted: learned noise, files ppo_seed<seed>)
Output: results_rl/ppo_seed<seed>.npz or ppoa_seed<seed>.npz (weights) and .json (learning curve); the
        folder can be changed with the environment variable CARUP_RL_OUT.
"""
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from decision_dp import Params, wealth_equivalent_loss
from rl_core import (build, sample_life, status_at, incomes_one, utility, dyn_features, base_logit, rate_at, init_policy, init_mlp, mlp,
                     adam_init, adam_update, save_params, make_exact_evaluator, monitoring_set, MENU, T_MAX, N_STATIC, N_DYN,
                     STATUS_H, STATUS_W, STATUS_BOTH, RULE)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
LIVES = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
WIDTH = int(sys.argv[4]) if len(sys.argv) > 4 else 64
LR, CLIP_EPS, EPOCHS, MINIBATCHES, LAMBDA, ENT, MAX_NORM = 3e-4, 0.2, 4, 8, 0.95, 1e-3, 0.5
R_DIV, R_FLOOR = 5.0, -50.0
EVAL_EVERY = int(sys.argv[5]) if len(sys.argv) > 5 else 25
LOGSTD0 = -0.5
FINAL = float(sys.argv[6]) if len(sys.argv) > 6 else None          # None: learned noise
KIND = "ppo" if FINAL is None else "ppoa"
OUT = os.environ.get("CARUP_RL_OUT", "results_rl")
os.makedirs(OUT, exist_ok=True)

par = Params()
S = build(par)
train, sim = S["train"], S["sim"]
n_train = train["feat"].shape[0]
menu = jnp.asarray(MENU)
P = len(MENU)
log_normal = lambda x, mu, logstd: -0.5 * ((x - mu) / jnp.exp(logstd)) ** 2 - logstd - 0.5 * jnp.log(2 * jnp.pi)


def init(key):
    k1, k2, k3 = jax.random.split(key, 3)
    p = init_policy(k1, hidden=(WIDTH, WIDTH))
    p["v_ann"] = init_mlp(k2, (N_STATIC, WIDTH, WIDTH, 1), last_scale=1.0)
    p["v_cons"] = init_mlp(k3, (N_STATIC + N_DYN + 2, WIDTH, WIDTH, 1), last_scale=1.0)
    p["logstd"] = jnp.full(2, LOGSTD0)                   # annuitised share, consumed share (logit scale); unused when annealed
    return p


def one_life(params, key, logstd):
    """One simulated life under the stochastic policy. Returns the two kinds of transitions."""
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    n = jax.random.randint(k0, (), 0, n_train)
    K = sample_life(k1, n, sim, train)
    feat = train["feat"][n]
    out = mlp(params["ann"], feat)
    logits = out[P:]
    k = jax.random.categorical(k2, logits)
    a_alpha = out[k] + jnp.exp(logstd[0]) * jax.random.normal(k3)
    logp0 = jax.nn.log_softmax(logits)[k] + log_normal(a_alpha, out[k], logstd[0])
    v0 = mlp(params["v_ann"], feat)[0]
    phi = menu[k]
    yB, yS, x0 = incomes_one(par, train["aB"][n], train["aA"][n], jax.nn.sigmoid(a_alpha), phi)
    eps = jax.random.normal(k4, (T_MAX,))
    R = 1.0 + par.r

    def step(cash, inp):
        t, e = inp
        status = status_at(K, t)
        alive = status < 3
        since = jnp.where(status == STATUS_H, t - 1 - K[1], jnp.where(status == STATUS_W, t - 1 - K[0], 0))
        rate = rate_at(train["rateB"][n], train["rateS"][n], t, status)
        dyn = dyn_features(t, status, since, cash, yB, yS, phi, rate)
        x = jnp.concatenate([feat, dyn])
        xv = jnp.concatenate([x, jnp.log(jnp.stack([yB, yS]))])          # the value depends on the level of the incomes, the policy does not
        base = base_logit(cash, jnp.where(status == STATUS_BOTH, yB, yS), rate)
        mu = base + mlp(params["cons"], x)[0]
        a = mu + jnp.exp(logstd[1]) * e
        c = jax.nn.sigmoid(a) * cash
        r = jnp.where(alive, train["scale"][n] * utility(par, jnp.where(alive, c, 1.0), status), 0.0)
        r = jnp.maximum(r / R_DIV, R_FLOOR)
        nxt = status_at(K, t + 1)
        cash_next = jnp.where(alive, R * (cash - c) + jnp.where(nxt == STATUS_BOTH, yB, yS), cash)
        return cash_next, dict(x=x, xv=xv, base=base, a=a, logp=log_normal(a, mu, logstd[1]),
                               v=mlp(params["v_cons"], xv)[0], r=r, alive=alive)

    _, tr = lax.scan(step, x0, (jnp.arange(T_MAX), eps))
    # generalised advantage estimation along the life
    alive_next = jnp.concatenate([tr["alive"][1:], jnp.zeros(1, dtype=bool)])
    v_next = jnp.concatenate([tr["v"][1:], jnp.zeros(1)])

    def back(adv_next, inp):
        r, v, vn, an, al = inp
        delta = r + par.beta * vn * an - v
        adv = jnp.where(al, delta + par.beta * LAMBDA * an * adv_next, 0.0)
        return adv, adv

    _, adv = lax.scan(back, 0.0, (tr["r"], tr["v"], v_next, alive_next, tr["alive"]), reverse=True)
    adv0 = (tr["v"][0] - v0) + LAMBDA * adv[0]            # no time passes between the annuity decision and year 0
    ann = dict(feat=feat, k=k, a=a_alpha, logp=logp0, adv=adv0, ret=adv0 + v0)
    cons = dict(x=tr["x"], xv=tr["xv"], base=tr["base"], a=tr["a"], logp=tr["logp"], adv=adv, ret=adv + tr["v"], w=1.0 * tr["alive"])
    return ann, cons, jnp.sum(par.beta ** jnp.arange(T_MAX) * tr["r"])


def loss_fn(params, ann, cons, sched):
    logstd = params["logstd"] if FINAL is None else sched              # learned, or fixed by the schedule
    def normalise(adv, w):
        mean = jnp.sum(w * adv) / jnp.sum(w)
        std = jnp.sqrt(jnp.sum(w * (adv - mean) ** 2) / jnp.sum(w))
        return (adv - mean) / (std + 1e-8)

    def surrogate(logp, logp_old, adv, w):
        ratio = jnp.exp(logp - logp_old)
        return -jnp.sum(w * jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv)) / jnp.sum(w)

    # consumption years
    mu = cons["base"] + jax.vmap(lambda x: mlp(params["cons"], x)[0])(cons["x"])
    v = jax.vmap(lambda x: mlp(params["v_cons"], x)[0])(cons["xv"])
    w = cons["w"]
    pg1 = surrogate(log_normal(cons["a"], mu, logstd[1]), cons["logp"], normalise(cons["adv"], w), w)
    vl1 = 0.5 * jnp.sum(w * (v - cons["ret"]) ** 2) / jnp.sum(w)
    # annuity decisions
    out = jax.vmap(lambda f: mlp(params["ann"], f))(ann["feat"])
    logsm = jax.nn.log_softmax(out[:, P:])
    mu0 = jnp.take_along_axis(out[:, :P], ann["k"][:, None], 1)[:, 0]
    logp0 = jnp.take_along_axis(logsm, ann["k"][:, None], 1)[:, 0] + log_normal(ann["a"], mu0, logstd[0])
    ones = jnp.ones_like(ann["adv"])
    pg0 = surrogate(logp0, ann["logp"], normalise(ann["adv"], ones), ones)
    v0 = jax.vmap(lambda f: mlp(params["v_ann"], f)[0])(ann["feat"])
    vl0 = 0.5 * jnp.mean((v0 - ann["ret"]) ** 2)
    entropy = -jnp.mean(jnp.sum(jnp.exp(logsm) * logsm, axis=1)) + jnp.sum(logstd)                 # up to constants
    return pg1 + pg0 + 0.5 * (vl1 + vl0) - ENT * entropy


@jax.jit
def iteration(params, opt, key, lr, sched):
    k1, k2 = jax.random.split(key)
    logstd = params["logstd"] if FINAL is None else sched
    ann, cons, ret = jax.vmap(lambda k: one_life(params, k, logstd))(jax.random.split(k1, LIVES))
    cons = {name: x.reshape((LIVES * T_MAX,) + x.shape[2:]) for name, x in cons.items()}
    n_c, n_a = LIVES * T_MAX // MINIBATCHES, LIVES // MINIBATCHES

    def epoch(carry, key_e):
        ka, kc = jax.random.split(key_e)
        pa = jax.random.permutation(ka, LIVES).reshape(MINIBATCHES, n_a)
        pc = jax.random.permutation(kc, LIVES * T_MAX).reshape(MINIBATCHES, n_c)

        def minibatch(carry, idx):
            params, opt = carry
            ia, ic = idx
            grads = jax.grad(loss_fn)(params, {k: v[ia] for k, v in ann.items()}, {k: v[ic] for k, v in cons.items()}, sched)
            return adam_update(params, grads, opt, lr, max_norm=MAX_NORM), None

        return lax.scan(minibatch, carry, (pa, pc))[0], None

    (params, opt), _ = lax.scan(epoch, (params, opt), jax.random.split(k2, EPOCHS))
    return params, opt, jnp.mean(ret)


# exact evaluation on 300 training couples whose optimum is known (never on the held-out couples)
mon, mon_group, mon_both, mon_only, mon_best = monitoring_set(S)
evaluate = make_exact_evaluator(par)


def monitor(params):
    alpha, k, val = evaluate(params, mon_group, mon_both, mon_only)
    gap = 100 * wealth_equivalent_loss(val, mon_best["value"], par)
    return dict(gap_mean=float(gap.mean()), gap_median=float(np.median(gap)), gap_max=float(gap.max()),
                alpha_mean=float(alpha.mean()), same_share=float(np.mean(np.isclose(MENU[k], mon_best["phi"]))))


def schedule(it):
    """Log standard deviations of the exploration noise at iteration `it` (annealed variant)."""
    return jnp.full(2, LOGSTD0 + ((FINAL if FINAL is not None else LOGSTD0) - LOGSTD0) * min(it, ITERS - 1) / max(ITERS - 1, 1))


params = init(jax.random.PRNGKey(seed))
opt = adam_init(params)
key = jax.random.PRNGKey(2000 + seed)
curve, t0, rets = [], time.time(), []
for it in range(ITERS + 1):
    if it % EVAL_EVERY == 0 or it == ITERS:
        m = dict(iteration=it, lives=it * LIVES, seconds=time.time() - t0, **monitor(params))
        m["logstd"] = [float(x) for x in (params["logstd"] if FINAL is None else schedule(it))]
        curve.append(m)
        print(f"iteration {it:4d} ({m['lives']:8d} lives, {m['seconds']:5.0f}s): gap to the optimum on {len(mon)} training couples: "
              f"mean {m['gap_mean']:.3f} %, median {m['gap_median']:.3f} %, max {m['gap_max']:.2f} %; alpha {m['alpha_mean']:.3f}; "
              f"same survivor share {m['same_share']:.2f}; std {np.exp(m['logstd'][0]):.2f}/{np.exp(m['logstd'][1]):.2f}"
              + (f"; mean return {np.mean(rets):.3f}" if rets else ""), flush=True)
        rets = []
    if it == ITERS:
        break
    key, sub = jax.random.split(key)
    params, opt, ret = iteration(params, opt, sub, LR * (1.0 - it / ITERS), schedule(it))
    rets.append(float(ret))
save_params(f"{OUT}/{KIND}_seed{seed}.npz", {"ann": params["ann"], "cons": params["cons"]}, width=WIDTH)
json.dump(dict(seed=seed, iterations=ITERS, lives_per_iteration=LIVES, width=WIDTH, lr=LR, clip=CLIP_EPS, epochs=EPOCHS,
               minibatches=MINIBATCHES, gae_lambda=LAMBDA, entropy=ENT, reward_divisor=R_DIV, reward_floor=R_FLOOR, rule=RULE,
               noise="learned" if FINAL is None else f"annealed from exp({LOGSTD0}) to exp({FINAL})", monitoring="300 training couples",
               seconds=time.time() - t0, curve=curve), open(f"{OUT}/{KIND}_seed{seed}.json", "w"), indent=2)
print(f"saved {OUT}/{KIND}_seed{seed}.npz in {time.time() - t0:.0f}s")
