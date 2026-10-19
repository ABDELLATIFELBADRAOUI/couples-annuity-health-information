"""check_rl.py -- the simulator used for learning against the exact computations used for evaluation.

1. Simulated lives (rl_core.sample_life) against the exact law of the full model with the limiting
   age: probabilities of the three states after 5, 10, 20 and 30 years, for three held-out couples.
2. A consumption network with random weights and a fixed annuity choice: average discounted utility
   over simulated lives (the quantity the agents learn from) against the exact value obtained by
   following the same network along every history (the quantity reported in the results).
3. The trained agent: its exact value against the average over lives simulated by an independent
   simulator (couple_env.CoupleEnv), which draws the dates of death from the exact laws and keeps its
   own accounts.
"""
import numpy as np
import jax
import jax.numpy as jnp

from decision_setup import law
from decision_dp import Params
from rl_core import build, sample_life, life_return, tree_value, init_policy, MENU, AGE_LIMIT

par = Params()
S = build(par)
both, only = law("TRUE", AGE_LIMIT)
test, sim = S["test"], S["sim"]
LIVES = 1_000_000
couples = [3, 250, 1100]
sample = jax.jit(jax.vmap(lambda k, n: sample_life(k, n, sim, test)))
worst = 0.0
print(f"1. simulated lives against the exact law ({LIVES:,} lives per couple)")
Ks = {}
for n in couples:
    K = np.concatenate([np.asarray(sample(jax.random.split(jax.random.PRNGKey(10 * n + j), LIVES // 4), jnp.full(LIVES // 4, n)))
                        for j in range(4)])
    Ks[n] = K
    zs = []
    for t in (5, 10, 20, 30):
        for p, freq in ((both[n, t], np.mean((K[:, 0] >= t) & (K[:, 1] >= t))),
                        (only[n, 0, :, t].sum(), np.mean((K[:, 0] >= t) & (K[:, 1] < t))),
                        (only[n, 1, :, t].sum(), np.mean((K[:, 1] >= t) & (K[:, 0] < t)))):
            if p * LIVES > 50:
                zs.append((freq - p) / np.sqrt(p * (1 - p) / LIVES))
    worst = max(worst, max(abs(z) for z in zs))
    print(f"   couple {n} (ages {int(test['age0'][n, 0])}/{int(test['age0'][n, 1])}): {len(zs)} probabilities, largest |z| {max(abs(z) for z in zs):.2f}; "
          f"alive at {AGE_LIMIT} or later: {int(np.sum(test['age0'][n][None, :] + K >= AGE_LIMIT))}")

print("2. learning objective against exact evaluation (random network, fixed annuity choice)")
params = init_policy(jax.random.PRNGKey(0))
params["cons"] = [(w + 0.3 * jax.random.normal(jax.random.PRNGKey(5 + i), w.shape) * (i == len(params["cons"]) - 1), b)
                  for i, (w, b) in enumerate(params["cons"])]              # make the last layer non-trivial
ret = jax.jit(jax.vmap(lambda K, n, a, p: life_return(params, par, test, n, K, a, p), in_axes=(0, None, None, None)))
exact = jax.jit(lambda n, a, p: tree_value(params, par, test["feat"][n], test["aB"][n], test["aA"][n], test["rateB"][n], test["rateS"][n],
                                           jnp.asarray(both)[n], jnp.asarray(only)[n], a, p))
for n, alpha, phi in zip(couples, (0.6, 0.9, 0.3), (MENU[1], MENU[0], MENU[3])):
    r = np.concatenate([np.asarray(ret(jnp.asarray(Ks[n][lo:lo + 125_000]), n, alpha, phi))
                        for lo in range(0, LIVES, 125_000)]) / float(test["scale"][n])        # back to the true scale
    v = float(exact(n, alpha, phi))
    se = r.std(ddof=1) / np.sqrt(r.size)
    z = (r.mean() - v) / se
    worst = max(worst, abs(z))
    print(f"   couple {n}, alpha {alpha:.2f}, phi {phi:.3f}: exact {v:.3f}, simulated {r.mean():.3f} (standard error {se:.3f}, z {z:+.2f})")

# 3. the trained agent, valued exactly and by an independent simulator (couple_env.CoupleEnv, built
#    on the exact laws, with its own accounting of cash, incomes and rewards)
import os
if os.path.exists("results_rl/apg_seed0.npz"):
    from couple_env import CoupleEnv
    from rl_core import load_params, annuity_choice, dyn_features, share_logit, rate_at, STATUS_BOTH, STATUS_H, STATUS_W
    print("3. trained agent (results_rl/apg_seed0.npz): exact value against an independent simulator")
    width = int(np.load("results_rl/apg_seed0.npz")["width"])
    agent = load_params("results_rl/apg_seed0.npz", init_policy(jax.random.PRNGKey(0), hidden=(width, width)))
    pop_both, pop_only = law("POP", AGE_LIMIT)
    E = 400_000

    @jax.jit
    def shares(feat, rB, rS, t, status, since, cash, yB, yS, phi):
        def f(st, si, ca):
            rate = rate_at(rB, rS, t, st)
            return jax.nn.sigmoid(share_logit(agent, feat, dyn_features(t, st, si, ca, yB, yS, phi, rate), ca, jnp.where(st == STATUS_BOTH, yB, yS), rate))
        return jax.vmap(f)(status, since, cash)

    for n in couples:
        alpha, k = annuity_choice(agent, test["feat"][n])
        alpha, phi = float(alpha), float(MENU[int(k)])
        v = float(jax.jit(lambda: tree_value(agent, par, test["feat"][n], test["aB"][n], test["aA"][n], test["rateB"][n], test["rateS"][n],
                                             jnp.asarray(both)[n], jnp.asarray(only)[n], alpha, phi))())
        env = CoupleEnv(par, both[n:n + 1], only[n:n + 1], pop_both[n:n + 1], pop_only[n:n + 1], seed=500 + n)
        env.reset(np.zeros(E, dtype=int))
        env.step(np.column_stack([np.full(E, alpha), np.full(E, phi)]))
        ret, disc = np.zeros(E), 1.0
        for t in range(env.T):
            b, h, w = env._status()
            alive = b | h | w
            status = np.where(b, STATUS_BOTH, np.where(h, STATUS_H, STATUS_W))
            since = np.where(h, t - 1 - env.K[:, 1], np.where(w, t - 1 - env.K[:, 0], 0))
            cash = np.where(alive, env.cash, 1.0)
            sh = np.asarray(shares(test["feat"][n], test["rateB"][n], test["rateS"][n], t, jnp.asarray(status), jnp.asarray(since),
                                   jnp.asarray(cash), env.yB[0], env.yS[0], phi))
            _, reward, _ = env.step(np.where(alive, sh, 1.0))
            ret += disc * reward
            disc *= par.beta
        se = ret.std(ddof=1) / np.sqrt(E)
        z = (ret.mean() - v) / se
        worst = max(worst, abs(z))
        print(f"   couple {n}: the agent chooses alpha {alpha:.3f}, phi {phi:.3f}; exact value {v:.3f}, independent simulator {ret.mean():.3f} "
              f"(standard error {se:.3f}, z {z:+.2f})")
print(f"largest |z|: {worst:.2f}")
