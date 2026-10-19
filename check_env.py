"""check_env.py -- the simulator (couple_env.CoupleEnv) against the exact computations.

For a few couples and actions, the dynamic-programming policy is followed in the simulator over many
simulated lives. The average discounted sum of rewards must equal the exact value of that policy
(computed along every history in decision_dp), up to the Monte Carlo error. The simulated dates of
death are also compared with the exact law.
"""
import numpy as np
import jax
import jax.numpy as jnp

from decision_setup import law, held_out, horizon
from decision_dp import Params, grids, conditional_probabilities, solve_widow, solve_both, paths, value
from couple_env import CoupleEnv

E = 400_000
laws = {b: law(b, 110) for b in ["POP", "TRUE"]}        # the laws of the reference results (limiting age of 110)
_, dte, *_ = held_out()
H = horizon(dte)
T = 40
pool = np.where((H > 30) & (H <= 40))[0]
J = jnp.asarray


def cons_np(xq, a, c):
    xp = a + c
    i = np.clip(np.searchsorted(xp, xq, side="right") - 1, 0, len(a) - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = c[i] + (xq - xp[i]) / (xp[i + 1] - xp[i]) * (c[i + 1] - c[i])
    return np.where(xq <= xp[0], xq, inside)


worst = 0.0
for seed, pos, alpha, phi, xi in [(sd,) + case for sd in (100, 700) for case in ((0, 0.50, 2 / 3, 0.0), (1, 1.00, 0.50, 0.0), (5, 0.85, 0.75, 0.0), (3, 0.60, 0.50, 0.5))]:
    par = Params(xi=xi)
    aw, ab = [np.asarray(x) for x in grids(par)]
    i = pool[pos:pos + 1]
    cut = lambda b: (laws[b][0][i][:, :T + 1], laws[b][1][i][:, :, :T, :T + 1])
    both, only = cut("TRUE")
    env = CoupleEnv(par, both, only, *cut("POP"), seed=seed + pos)
    env.reset(np.zeros(E, dtype=int))
    obs, _, _ = env.step(np.column_stack([np.full(E, alpha), np.full(E, phi)]))
    yB, yS, x0 = float(env.yB[0]), float(env.yS[0]), float(env.cash[0])

    # exact: dynamic-programming policy and its value along every history
    q, pB = [x[0] for x in conditional_probabilities(both, only)]
    cw = jax.vmap(lambda qs: solve_widow(qs, J(aw), par))(J(q))
    t_idx = jnp.arange(T - 1)
    entry = cw[:, t_idx + 1, t_idx, :]
    cB = solve_both(J(pB), entry[0], entry[1], yB, yS, J(ab), J(aw), par)
    cb, cs = paths(cB, cw, yB, yS, x0, J(ab), J(aw), par)
    exact = float(value(cb, cs, J(both[0]), J(only[0]), par))
    cB, cw = np.asarray(cB), np.asarray(cw)

    # simulated dates of death against the law
    K = env.K
    z_law = []
    for t in (5, 10, 20, 30):
        p = both[0, t]
        z_law.append((np.mean((K[:, 0] >= t) & (K[:, 1] >= t)) - p) / np.sqrt(p * (1 - p) / E))
        p = only[0, 0, :, t].sum()
        z_law.append((np.mean((K[:, 0] >= t) & (K[:, 1] < t)) - p) / np.sqrt(p * (1 - p) / E))
        p = only[0, 1, :, t].sum()
        z_law.append((np.mean((K[:, 1] >= t) & (K[:, 0] < t)) - p) / np.sqrt(p * (1 - p) / E))

    # the policy followed in the simulator
    ret, disc = np.zeros(E), 1.0
    for t in range(T):
        b, h, w = env._status()
        cash, c = env.cash, np.ones(E)
        c[b] = cons_np(cash[b], ab, cB[t])
        for s, mask in ((0, h), (1, w)):
            partner_death = env.K[:, 1 - s]
            for d in np.unique(partner_death[mask]):
                m = mask & (partner_death == d)
                c[m] = yS * cons_np(cash[m] / yS, aw, cw[s, t, d])
        alive = b | h | w
        share = np.where(alive, c / np.where(alive, cash, 1.0), 1.0)
        _, reward, done = env.step(share)
        ret += disc * reward
        disc *= par.beta
    se = ret.std(ddof=1) / np.sqrt(E)
    z = (ret.mean() - exact) / se
    worst = max(worst, abs(z), max(abs(x) for x in z_law))
    print(f"run {seed}, couple {int(i[0])} (ages {dte.age_h.values[i[0]]}/{dte.age_w.values[i[0]]}), alpha {alpha:.2f}, phi {phi:.2f}, xi {xi}: "
          f"exact value {exact:.4f}, simulated {ret.mean():.4f} (standard error {se:.4f}, z {z:+.2f}); "
          f"dates of death: largest |z| {max(abs(x) for x in z_law):.2f} over {len(z_law)} probabilities", flush=True)
print(f"{E:,} simulated lives per case, two independent runs per case; largest |z|: {worst:.2f}")
