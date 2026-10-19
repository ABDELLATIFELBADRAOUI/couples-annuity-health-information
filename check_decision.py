"""check_decision.py -- four checks of the solution of the couple's decision problem.

1. Without any income (alpha = 0), the problem has a closed form: its value must equal the value
   obtained by following the closed-form policy along every history.
2. Optimality. Since the only uncertainty is the two dates of death, a policy is a finite list of
   numbers (the share of cash saved at each node of the tree of histories). The expected utility is
   maximised directly over this list with a general-purpose optimiser (L-BFGS-B, exact gradient),
   without any Euler equation or grid, started (a) from the dynamic-programming policy and (b) from
   a naive policy. The gaps are expressed as a share of the initial resources.
3. Grid refinement: same couples with twice as many grid points.
4. Search over the annuitised share: golden-section search against a grid of step 0.005.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from scipy.optimize import minimize

from decision_setup import law, held_out, horizon
from decision_dp import (Params, grids, conditional_probabilities, annuity_due, incomes, solve_widow, solve_both,
                         paths, value, make_solver, solve_couples, wealth_equivalent_loss)

t_start = time.time()
laws = {b: law(b, 110) for b in ["POP", "M0", "TRUE"]}        # the laws of the reference results (limiting age of 110)
_, dte, *_ = held_out()
H = horizon(dte)
J = jnp.asarray


def tree_value(sigB, sigS, yB, yS, x0, both, only, par):
    """Expected utility when the couple saves the share sigB[t] of its cash in year t and the survivor
    the share sigS[s, d, t] (everything is consumed in the last year)."""
    R, T = 1.0 + par.r, sigB.shape[0]
    last = jnp.arange(T) == T - 1

    def step_b(x, inp):
        s, is_last = inp
        s = jnp.where(is_last, 0.0, s)
        return R * s * x + yB, ((1.0 - s) * x, R * s * x)

    _, (cb, w_after) = lax.scan(step_b, x0, (sigB, last))
    d_idx = jnp.arange(T)

    def step_s(x, inp):
        t, s_t = inp
        s_t = jnp.where(t == T - 1, 0.0, s_t)
        active = (d_idx < t)[None, :]
        return jnp.where(active, R * s_t * x + yS, x), jnp.where(active, (1.0 - s_t) * x, 1.0)

    _, cs = lax.scan(step_s, jnp.broadcast_to(w_after + yS, (2, T)), (jnp.arange(T), jnp.transpose(sigS, (2, 0, 1))))
    return value(cb, cs, both, only, par)


def shares_of_paths(cb, cs, yB, yS, x0, par):
    """Saving shares implied by the consumption paths of a policy."""
    R, T = 1.0 + par.r, cb.shape[0]
    sigB, w_after, x = np.zeros(T), np.zeros(T), x0
    for t in range(T):
        sigB[t] = 1.0 - cb[t] / x
        w_after[t] = R * (x - cb[t])
        x = w_after[t] + yB
    sigS = np.zeros((2, T, T))
    for s in range(2):
        for d in range(T - 1):
            x = w_after[d] + yS
            for t in range(d + 1, T):
                sigS[s, d, t] = 1.0 - cs[t, s, d] / x
                x = R * (x - cs[t, s, d]) + yS
    return np.clip(sigB, 0.0, 1.0), np.clip(sigS, 0.0, 1.0)


# ------------------------------------------------------------------ 1 and 2
print("1-2. closed form without income, and optimality on the tree of histories")
cases = [  # (couple position among the held-out couples with a horizon of 31-40 years, alpha, phi, xi, risk aversion, belief)
    (0, 0.50, 2 / 3, 0.0, 3.0, "TRUE"), (0, 1.00, 0.50, 0.0, 3.0, "TRUE"), (1, 0.85, 0.75, 0.0, 3.0, "TRUE"),
    (2, 0.30, 1.00, 0.0, 3.0, "M0"), (3, 0.60, 0.50, 0.5, 3.0, "TRUE"), (4, 0.95, 2 / 3, 0.0, 5.0, "POP"),
    (5, 0.70, 2 / 3, 0.0, 2.0, "TRUE"), (6, 0.00, 2 / 3, 0.5, 3.0, "TRUE")]
T = 40
pool = np.where((H > 30) & (H <= 40))[0]
worst_dp, worst_naive = 0.0, 0.0
for pos, alpha, phi, xi, g, belief in cases:
    par = Params(g=g, xi=xi)
    aw, ab = grids(par)
    i = pool[pos:pos + 1]
    cut = lambda b: (laws[b][0][i][:, :T + 1], laws[b][1][i][:, :, :T, :T + 1])
    pb, po = cut("POP")
    aB = annuity_due(pb, po, 0.0, par.r)
    aA = annuity_due(pb, po, 1.0, par.r) - aB
    yB, yS, x0 = [float(x[0, 0]) for x in incomes(par, aB, aA, np.array([[alpha]]), np.array([[phi]]))]
    both, only = [x[0] for x in cut(belief)]
    q, pB = [x[0] for x in conditional_probabilities(*cut(belief))]
    cw = jax.vmap(lambda qs: solve_widow(qs, aw, par))(J(q))
    t_idx = jnp.arange(T - 1)
    entry = cw[:, t_idx + 1, t_idx, :]
    cB = solve_both(J(pB), entry[0], entry[1], yB, yS, ab, aw, par)
    cb, cs = paths(cB, cw, yB, yS, x0, ab, aw, par)
    w_dp = float(value(cb, cs, J(both), J(only), par))
    sigB, sigS = shares_of_paths(np.asarray(cb), np.asarray(cs), yB, yS, x0, par)

    scale = abs(w_dp)
    f = jax.jit(jax.value_and_grad(lambda z: -tree_value(z[:T], z[T:].reshape(2, T, T), yB, yS, x0, J(both), J(only), par) / scale))

    def fun(z):
        v, gr = f(J(z))
        return float(v), np.asarray(gr)

    res = {}
    for name, z0 in (("dp", np.concatenate([sigB, sigS.reshape(-1)])), ("naive", np.full(T + 2 * T * T, 0.5))):
        r = minimize(fun, np.clip(z0, 0.0, 0.999999), jac=True, method="L-BFGS-B", bounds=[(0.0, 0.999999)] * z0.size,
                     options=dict(maxiter=4000, maxfun=6000, ftol=1e-16, gtol=1e-12, maxcor=30))
        res[name] = (-r.fun * scale, r.nit)
    gap_dp = float(wealth_equivalent_loss(w_dp, res["dp"][0], par))            # > 0: the optimiser improved on the DP policy
    gap_naive = float(wealth_equivalent_loss(res["naive"][0], max(w_dp, res["dp"][0]), par))   # > 0: naive start ended below
    worst_dp, worst_naive = max(worst_dp, abs(gap_dp)), max(worst_naive, abs(gap_naive))
    print(f"   couple {int(i[0])} (ages {dte.age_h.values[i[0]]}/{dte.age_w.values[i[0]]}), alpha {alpha:.2f}, phi {phi:.2f}, xi {xi}, g {g}, law {belief}: "
          f"improvement found from the DP policy {100 * gap_dp:.2e} % ({res['dp'][1]} it.), "
          f"naive start ends {100 * gap_naive:.2e} % below ({res['naive'][1]} it.)", flush=True)
print(f"   largest improvement found on the DP policy: {100 * worst_dp:.2e} % of the resources; "
      f"largest distance of the naive start: {100 * worst_naive:.2e} %")

# ------------------------------------------------------------------ 3. grid refinement
print("3. grid refinement (48 couples, 16 per horizon group)")
pick = np.concatenate([np.where((H > lo) & (H <= lo + 10))[0][:16] for lo in (30, 40, 50)])
sub = {b: (laws[b][0][pick], laws[b][1][pick]) for b in laws}
outs = {}
for label, par in (("base", Params()), ("double", Params(Gw=280, Gb=200))):
    outs[label] = solve_couples(par, sub, ["TRUE", "M0", "POP"], H[pick])
par = Params()
print(f"   closed form without income: largest relative gap {max(np.nanmax(outs['base']['closed_form_gap'][m]) for m in ('TRUE', 'M0', 'POP')):.1e}")
for m in ("TRUE", "M0", "POP"):
    dv = np.abs(wealth_equivalent_loss(outs["base"]["true"][m], outs["double"]["true"][m], par))
    da = np.abs(outs["base"]["alpha"][m] - outs["double"]["alpha"][m])
    same_phi = np.mean(np.isclose(outs["base"]["phi"][m], outs["double"]["phi"][m], equal_nan=True))
    print(f"   {m:4s}: value of the policy under the truth, largest gap {100 * dv.max():.2e} % of the resources; "
          f"alpha: largest gap {da.max():.4f}; same survivor share: {same_phi:.2f}")
for m in ("M0", "POP"):
    lb = 100 * wealth_equivalent_loss(outs["base"]["true"][m], outs["base"]["true"]["TRUE"], par)
    ld = 100 * wealth_equivalent_loss(outs["double"]["true"][m], outs["double"]["true"]["TRUE"], par)
    print(f"   {m:4s}: mean loss {lb.mean():.4f} % with the base grids, {ld.mean():.4f} % with the double grids; "
          f"largest change for one couple {np.abs(lb - ld).max():.2e} points; smallest loss {lb.min():.2e} %")

# ------------------------------------------------------------------ 4. search over alpha
print("4. golden-section search against a grid of step 0.005 (16 couples, 4 survivor shares)")
pick = np.where((H > 40) & (H <= 50))[0][:16]
T = 50
par = Params()
cut = lambda b: (laws[b][0][pick][:, :T + 1], laws[b][1][pick][:, :, :T, :T + 1])
pb, po = cut("POP")
aB = annuity_due(pb, po, 0.0, par.r)
aA = annuity_due(pb, po, 1.0, par.r) - aB
widow_stage, action_stage, no_income = make_solver(par)
bt, ot = cut("TRUE")
q, pB = conditional_probabilities(bt, ot)
cw = widow_stage(J(q))
grid = np.arange(1, 201) * 0.005
best = np.full((16, len(par.phis)), -np.inf)
for j, phi in enumerate(par.phis):
    for lo in range(0, 200, 50):
        a = np.tile(grid[lo:lo + 50], (16, 1))
        yB, yS, x0 = incomes(par, aB, aA, a, np.full_like(a, phi))
        wo, _ = action_stage(cw, J(pB), J(yB), J(yS), J(x0), J(bt), J(ot), J(bt), J(ot))
        best[:, j] = np.maximum(best[:, j], np.asarray(wo).max(1))
w0 = np.asarray(no_income(J(q), J(pB), J(bt), J(ot), J(bt), J(ot))[0])
grid_best = np.maximum(best.max(1), w0)
out = solve_couples(par, {b: (laws[b][0][pick], laws[b][1][pick]) for b in laws}, ["TRUE"], H[pick])
gap = wealth_equivalent_loss(out["own"]["TRUE"], grid_best, par)       # > 0: the grid found better
print(f"   value found by the search against the best grid value: gap between {100 * gap.min():.2e} % and {100 * gap.max():.2e} % of the resources "
      f"(negative: the search found better than the grid)")
print(f"total time {time.time() - t_start:.0f}s")
