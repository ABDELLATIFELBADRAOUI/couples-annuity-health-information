"""
decision_dp.py -- the couple's retirement decision problem and its solution by dynamic programming.

The problem (one couple, resources normalised to 1 at time 0)
-------------------------------------------------------------
At time 0 the couple chooses a share alpha of its liquid wealth to convert into a joint annuity that
pays y per year while both spouses are alive and phi * y to the survivor (first payment at time 0).
Then, every year, the household receives its income, chooses its consumption c and carries the rest
forward at the interest rate r; it cannot borrow and leaves no bequest.

    utility of a year   : 2 u(c / n) while both are alive (n: equivalence scale), u(c) for a survivor
                          u(c) = c^(1 - g) / (1 - g)      (g: relative risk aversion)
    annuity price       : expected discounted payments under the POPULATION law (age and sex only,
                          spouses independent), divided by (1 - load)
    pre-annuitised share: a share xi of the resources may already be a fair joint annuity whose
                          survivor receives 2/3 (scenario parameter; 0 in the base case)

The only uncertainty is the two dates of death, so a history is "both alive after t years" or
"spouse s alone after t years, partner dead since year d". The law of these histories is given by
decision_law.joint_law, with no Markov approximation: the survivor's policy may depend on d.

Solution
--------
The consumption problem is concave, so the Euler equations with the no-borrowing condition are
necessary and sufficient. They are solved backwards with the endogenous-grid method (Carroll 2006):
    survivor : normalised problem with income 1 (CRRA utility makes the problem homogeneous in the
               income), one policy per (sex, year of the partner's death, year);
    couple   : one policy per action (alpha, phi) and year.
A policy is stored as the consumption c[j] at each point a[j] of a grid of end-of-year savings; the
corresponding cash-on-hand is a[j] + c[j], and below the first point the household consumes all its
cash. The value of a policy is NOT read from a value function: it is computed exactly by following
the policy along every history (their number is finite) and weighting each year's utility by the
probability of the history. The same policy can therefore be valued under any law: the one it was
computed for, or the true one.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

BIG = 1e12                    # "consume everything": the first endogenous point is out of reach
PHI_PRE = 2.0 / 3.0           # survivor share of the pre-existing annuity


@dataclass(frozen=True)
class Params:
    g: float = 3.0            # relative risk aversion
    n: float = 1.5            # equivalence scale of a couple
    r: float = 0.03           # interest rate
    beta: float = 1.0 / 1.03  # discount factor
    load: float = 0.10        # annuity loading
    xi: float = 0.0           # share of the resources already annuitised
    phis: tuple = (0.5, 2.0 / 3.0, 0.75, 1.0)   # survivor shares on offer
    Gw: int = 140             # grid points, survivor
    Gb: int = 100             # grid points, couple


GOLD = (np.sqrt(5.0) - 1.0) / 2.0
N_GOLD = 13                   # golden-section steps on the annuitised share: final bracket 0.618^13 = 0.002


def grids(par: Params):
    aw = np.concatenate([[0.0], np.logspace(-3, 5, par.Gw - 1)])   # in units of the survivor's income
    ab = 1.6 * np.linspace(0.0, 1.0, par.Gb) ** 3                                    # in units of the initial resources
    return jnp.asarray(aw), jnp.asarray(ab)


# ----------------------------------------------------------------------------- policies on a grid
def cons(xq, a, c):
    """Consumption at cash-on-hand xq for a policy stored as c[j] at the savings a[j]: the policy
    passes through the points (a[j] + c[j], c[j]); below the first one the household consumes all its
    cash; beyond the last one the last segment is extended. One query per policy: lazy binary search."""
    G = a.shape[0]
    lo = jnp.zeros(jnp.shape(xq), jnp.int32)
    hi = jnp.full(jnp.shape(xq), G - 1, jnp.int32)
    for _ in range(int(np.ceil(np.log2(G)))):              # last point with a + c <= xq
        mid = (lo + hi + 1) // 2
        go = (a[mid] + c[mid]) <= xq
        lo, hi = jnp.where(go, mid, lo), jnp.where(go, hi, mid - 1)
    i = jnp.clip(lo, 0, G - 2)
    x0, x1 = a[i] + c[i], a[i + 1] + c[i + 1]
    inside = c[i] + (xq - x0) / (x1 - x0) * (c[i + 1] - c[i])
    return jnp.where(xq <= a[0] + c[0], xq, inside)


def cons_grid(xq, a, c):
    """Same function for a whole vector of queries on one policy."""
    xp = a + c
    i = jnp.clip(jnp.searchsorted(xp, xq, side="right") - 1, 0, a.shape[0] - 2)
    inside = c[i] + (xq - xp[i]) / (xp[i + 1] - xp[i]) * (c[i + 1] - c[i])
    return jnp.where(xq <= xp[0], xq, inside)


# ----------------------------------------------------------------------------- probabilities of the histories
def conditional_probabilities(both, only):
    """From the law of the histories to the one-year probabilities used by the Euler equations.
    q[n, s, d, t]  : spouse s, alone since year d, survives year t           (0 where t <= d)
    pB[n, 0, t]    : both survive year t, given both alive after t years
    pB[n, 1 + s, t]: only spouse s survives year t (the partner dies in year t)
    """
    T = both.shape[1] - 1
    num, den = only[..., 1:], only[..., :-1]
    q = np.where(den > 1e-290, num / np.maximum(den, 1e-300), 0.0)                 # (N, 2, T, T)
    b = np.maximum(both[:, :T], 1e-300)
    idx = np.arange(T)
    pB = np.stack([both[:, 1:] / b, only[:, 0, idx, idx + 1] / b, only[:, 1, idx, idx + 1] / b], axis=1)
    return q, pB


def annuity_due(both, only, phi, r):
    """Expected discounted payments of 1 while both are alive and phi to a survivor, first payment at 0."""
    T = both.shape[1] - 1
    v = (1.0 + r) ** -np.arange(T)
    alone = only.sum(axis=(1, 2))[:, :T]
    return ((both[:, :T] + phi * alone) * v[None, :]).sum(axis=1)


def incomes(par: Params, aB, aA, alpha, phi):
    """Incomes and initial cash of the actions (alpha, phi), arrays (N, A), for couples whose
    population annuity values are aB (1 per year while both are alive) and aA (1 per year to a
    survivor). Returns yB (both alive), yS (survivor), x0 (cash at time 0, first payment included)."""
    price = (aB[:, None] + phi * aA[:, None]) / (1.0 - par.load)
    y = alpha * (1.0 - par.xi) / price
    y_pre = (par.xi / (aB + PHI_PRE * aA))[:, None]
    yB = y + y_pre
    return yB, phi * y + PHI_PRE * y_pre, (1.0 - par.xi) * (1.0 - alpha) + yB


# ----------------------------------------------------------------------------- backward solution
def solve_widow(q, aw, par: Params):
    """Survivor with income 1. q[d, t]. Returns c[t, d, :] (rows with t <= d are not used)."""
    T, R = q.shape[0], 1.0 + par.r
    xn = R * aw + 1.0

    def step(c_next, t):
        cn = jax.vmap(lambda c: cons_grid(xn, aw, c))(c_next)                        # consumption next year
        fac = (par.beta * R * jnp.maximum(q[:, t], 1e-300)) ** (-1.0 / par.g)
        c_t = jnp.minimum(fac[:, None] * cn, BIG)
        return c_t, c_t

    last = jnp.full((T, aw.shape[0]), BIG)                                           # year T-1: consume everything
    _, cs = lax.scan(step, last, jnp.arange(T - 2, -1, -1))
    return jnp.concatenate([cs[::-1], last[None]], axis=0)


def solve_widow0(q, par: Params):
    """Survivor without income: closed form. Returns the share of cash consumed, m[t, d], and K[t, d]
    with value = K * u(cash)."""
    T, R = q.shape[0], 1.0 + par.r

    def step(K_next, t):
        gr = (par.beta * q[:, t] * R ** (1.0 - par.g) * K_next) ** (1.0 / par.g)
        K = (1.0 + gr) ** par.g
        return K, (1.0 / (1.0 + gr), K)

    one = jnp.ones(T)
    _, (m, K) = lax.scan(step, one, jnp.arange(T - 2, -1, -1))
    return jnp.concatenate([m[::-1], one[None]], axis=0), jnp.concatenate([K[::-1], one[None]], axis=0)


def solve_both(pB, cH_entry, cW_entry, yB, yS, ab, aw, par: Params):
    """Couple, one action with an income. pB (3, T); c*_entry[t]: normalised policy of the survivor
    in year t + 1 when the partner dies in year t, t = 0..T-2. Returns c[t, :]."""
    R, n, g = 1.0 + par.r, par.n, par.g
    xB, xS = R * ab + yB, R * ab + yS

    def step(c_next, inp):
        p, ch, cw = inp
        cB = cons_grid(xB, ab, c_next)
        cH = yS * cons_grid(xS / yS, aw, ch)
        cW = yS * cons_grid(xS / yS, aw, cw)
        mu = p[0] * (2.0 / n) * (cB / n) ** (-g) + p[1] * cH ** (-g) + p[2] * cW ** (-g)
        c_t = jnp.minimum(n * ((n / 2.0) * par.beta * R * mu) ** (-1.0 / g), BIG)
        return c_t, c_t

    last = jnp.full(ab.shape[0], BIG)
    T = pB.shape[1]
    _, cs = lax.scan(step, last, (pB[:, :T - 1].T, cH_entry, cW_entry), reverse=True)
    return jnp.concatenate([cs, last[None]], axis=0)


def solve_both0(pB, KH_entry, KW_entry, par: Params):
    """Couple without any income (alpha = 0 and no pre-existing annuity): closed form.
    K*_entry[t]: value coefficient of the survivor in year t + 1 when the partner dies in year t.
    Returns the share of cash consumed, m[t], and k[t] with value = k * u(cash)."""
    R, n, g = 1.0 + par.r, par.n, par.g
    theta0 = 2.0 * n ** (g - 1.0)                              # 2 u(c / n) = theta0 u(c)

    def step(k_next, inp):
        p, kh, kw = inp
        Q = par.beta * R ** (1.0 - g) * (p[0] * k_next + p[1] * kh + p[2] * kw)
        gr = (Q / theta0) ** (1.0 / g)
        k = theta0 * (1.0 + gr) ** g
        return k, (1.0 / (1.0 + gr), k)

    T = pB.shape[1]
    _, (m, k) = lax.scan(step, jnp.asarray(theta0), (pB[:, :T - 1].T, KH_entry, KW_entry), reverse=True)
    return jnp.concatenate([m, jnp.ones(1)]), jnp.concatenate([k, jnp.full(1, theta0)])


# ----------------------------------------------------------------------------- exact value along the histories
def paths(cB, cw, yB, yS, x0, ab, aw, par: Params):
    """Consumption along every history for one couple and one action with an income.
    cB (T, Gb); cw (2, T, T, Gw) indexed [s, t, d]. Returns cB_path (T,), cS_path (T, 2, T) = [t, s, d]."""
    R, T = 1.0 + par.r, cB.shape[0]

    def step_b(x, c_row):
        c = cons(x, ab, c_row)
        w = R * (x - c)
        return w + yB, (c, w)

    _, (cb, w_after) = lax.scan(step_b, x0, cB)
    d_idx = jnp.arange(T)

    def step_s(x, inp):                                   # x (2, T): cash of the survivor, by year of the partner's death
        t, c_t = inp                                      # c_t (2, T, Gw)
        c = yS * jax.vmap(jax.vmap(lambda xq, c_row: cons(xq, aw, c_row)))(x / yS, c_t)
        active = (d_idx < t)[None, :]
        return jnp.where(active, R * (x - c) + yS, x), jnp.where(active, c, 1.0)

    x_init = jnp.broadcast_to(w_after + yS, (2, T))
    _, cs = lax.scan(step_s, x_init, (jnp.arange(T), jnp.swapaxes(cw, 0, 1)))
    return cb, cs


def paths0(mB, m, par: Params):
    """Same without any income, cash 1 at time 0. mB (T,): share of cash consumed by the couple;
    m (2, T, T) indexed [s, t, d]: share consumed by the survivor."""
    R, T = 1.0 + par.r, mB.shape[0]

    def step_b(x, share):
        c = share * x
        w = R * (x - c)
        return w, (c, w)

    _, (cb, w_after) = lax.scan(step_b, jnp.asarray(1.0), mB)
    d_idx = jnp.arange(T)

    def step_s(x, inp):
        t, m_t = inp
        c = m_t * x
        active = (d_idx < t)[None, :]
        return jnp.where(active, R * (x - c), x), jnp.where(active, c, 1.0)

    _, cs = lax.scan(step_s, jnp.broadcast_to(w_after, (2, T)), (jnp.arange(T), jnp.swapaxes(m, 0, 1)))
    return cb, cs


def value(cb, cs, both, only, par: Params):
    """Expected discounted utility of the consumption paths under the law (both, only) of one couple."""
    T = cb.shape[0]
    disc = par.beta ** jnp.arange(T)
    u = lambda c: c ** (1.0 - par.g) / (1.0 - par.g)
    w_only = jnp.transpose(only[:, :, :T], (2, 0, 1))                               # [t, s, d]
    us = jnp.where(w_only > 0, u(cs), 0.0)                                           # histories of probability zero do not count
    ub = jnp.where(both[:T] > 0, 2.0 * u(cb / par.n), 0.0)
    return jnp.sum(disc * both[:T] * ub) + jnp.sum(disc[:, None, None] * w_only * us)


# ----------------------------------------------------------------------------- one batch of couples
@functools.lru_cache(maxsize=None)        # the compiled functions are reused when the same problem is solved again
def make_solver(par: Params):
    """Jitted functions for a batch of couples.
    widow_stage(q)                      -> survivor policies (Nb, 2, T, T, Gw), indexed [n, s, t, d]
    action_stage(cw, pB, yB, yS, x0, both_own, only_own, both_true, only_true)
                                        -> value of the policy of each action under the belief and under the truth, (Nb, A)
    no_income(q, pB, both_own, only_own, both_true, only_true)
                                        -> same for alpha = 0 without any income, and the closed-form value under the belief
    """
    aw, ab = grids(par)

    @jax.jit
    def widow_stage(q):
        return jax.vmap(jax.vmap(lambda qs: solve_widow(qs, aw, par)))(q)

    def one_couple(cw, pB, yB, yS, x0, both_own, only_own, both_true, only_true):
        T = pB.shape[1]
        t_idx = jnp.arange(T - 1)
        entry = cw[:, t_idx + 1, t_idx, :]                                           # (2, T-1, Gw)

        def one_action(yb, ys, x):
            cB = solve_both(pB, entry[0], entry[1], yb, ys, ab, aw, par)
            cb, cs = paths(cB, cw, yb, ys, x, ab, aw, par)
            return value(cb, cs, both_own, only_own, par), value(cb, cs, both_true, only_true, par)

        return jax.vmap(one_action)(yB, yS, x0)

    def one_couple0(q, pB, both_own, only_own, both_true, only_true):
        T = pB.shape[1]
        m, K = jax.vmap(lambda qs: solve_widow0(qs, par))(q)                         # (2, T, T) [s, t, d]
        t_idx = jnp.arange(T - 1)
        K_entry = K[:, t_idx + 1, t_idx]
        mB, kB = solve_both0(pB, K_entry[0], K_entry[1], par)
        cb, cs = paths0(mB, m, par)
        closed_form = kB[0] / (1.0 - par.g)                                          # k u(1): value under the belief
        return value(cb, cs, both_own, only_own, par), value(cb, cs, both_true, only_true, par), closed_form

    return widow_stage, jax.jit(jax.vmap(one_couple)), jax.jit(jax.vmap(one_couple0))


def solve_couples(par: Params, laws, beliefs, horizons, truth="TRUE", pricing="POP", log=None):
    """Best action and value of each belief's policy, for every couple.

    laws     : dict belief -> (both (N, Tmax + 1), only (N, 2, Tmax, Tmax + 1))
    horizons : (N,) years to follow each couple (rounded up to a multiple of 10 for batching)
    For each belief m: the household believes m, chooses the action that is best under m and follows
    the consumption policy that is optimal under m. Returned, per couple:
        alpha[m], phi[m]     the action chosen
        own[m]               its value under m
        true[m]              its value under the truth
        true_action[m]       value under the truth of the TRUE policy at the action chosen by m
    For each survivor share on offer, the annuitised share is found by golden-section search on
    [0, 1] (the value is concave in alpha for a given survivor share); alpha = 1 and alpha = 0 are
    evaluated as well.
    """
    order = [truth] + [b for b in beliefs if b != truth]
    N = laws[truth][0].shape[0]
    phis = np.array(par.phis)
    P = len(phis)
    out = {k: {m: np.full(N, np.nan) for m in order} for k in ("alpha", "phi", "own", "true", "true_action")}
    out["closed_form_gap"] = {m: np.full(N, np.nan) for m in order}
    widow_stage, action_stage, no_income = make_solver(par)
    a_min = 0.0 if par.xi > 0 else 1e-3
    Tb = np.maximum(20, 10 * np.ceil(horizons / 10.0)).astype(int)
    J = lambda x: jnp.asarray(x)
    for T in sorted(set(Tb.tolist())):
        members = np.where(Tb == T)[0]
        nb = int(np.clip(4e8 // (2 * T * T * par.Gw * 8), 4, 64))
        for lo_i in range(0, len(members), nb):
            real = members[lo_i:lo_i + nb]
            idx = np.concatenate([real, np.repeat(real[-1:], nb - len(real))])       # pad the last batch
            cut = lambda b: (laws[b][0][idx][:, :T + 1], laws[b][1][idx][:, :, :T, :T + 1])
            pb, po = cut(pricing)
            aB = annuity_due(pb, po, 0.0, par.r)
            aA = annuity_due(pb, po, 1.0, par.r) - aB
            bt, ot = cut(truth)
            chosen, cw_truth, pB_truth = {}, None, None
            for m in order:
                bo, oo = cut(m)
                q, pB = conditional_probabilities(bo, oo)
                cw = widow_stage(J(q))
                if m == truth:
                    cw_truth, pB_truth = cw, pB

                def evaluate(alpha, phi):
                    yB, yS, x0 = incomes(par, aB, aA, alpha, phi)
                    wo, wt = action_stage(cw, J(pB), J(yB), J(yS), J(x0), J(bo), J(oo), J(bt), J(ot))
                    return np.asarray(wo), np.asarray(wt)

                phi = np.tile(phis, (nb, 1))
                lo, hi = np.full((nb, P), a_min), np.ones((nb, P))
                c, d = hi - GOLD * (hi - lo), lo + GOLD * (hi - lo)
                wo, wt = evaluate(np.concatenate([c, d, hi], 1), np.tile(phis, (nb, 3)))
                fc, fd, tc, td = wo[:, :P], wo[:, P:2 * P], wt[:, :P], wt[:, P:2 * P]
                cand_a = np.stack([c, d, hi], 2)                                     # (nb, P, 3)
                k = wo.reshape(nb, 3, P).argmax(1)
                take = lambda x: np.take_along_axis(x, k[:, :, None], 2)[:, :, 0]
                best_a, best_o, best_t = take(cand_a), take(wo.reshape(nb, 3, P).transpose(0, 2, 1)), take(wt.reshape(nb, 3, P).transpose(0, 2, 1))
                for _ in range(N_GOLD):
                    left = fc > fd                                                   # the maximum is in [lo, d]
                    hi, lo = np.where(left, d, hi), np.where(left, lo, c)
                    new = np.where(left, hi - GOLD * (hi - lo), lo + GOLD * (hi - lo))
                    wo, wt = evaluate(new, phi)
                    c, d, fc, fd = (np.where(left, new, d), np.where(left, c, new),
                                    np.where(left, wo, fd), np.where(left, fc, wo))
                    better = wo > best_o
                    best_a, best_o, best_t = np.where(better, new, best_a), np.where(better, wo, best_o), np.where(better, wt, best_t)
                if par.xi > 0:                                  # alpha = 0 keeps the pre-existing income
                    wo, wt = evaluate(np.zeros((nb, 1)), np.full((nb, 1), phis[0]))
                    phi0 = np.full((nb, 1), phis[0])
                else:                                           # alpha = 0: no income at all, closed form
                    wo, wt, cf = [np.asarray(x)[:, None] for x in no_income(J(q), J(pB), J(bo), J(oo), J(bt), J(ot))]
                    phi0 = np.full((nb, 1), np.nan)
                    out["closed_form_gap"][m][real] = np.abs(wo[:, 0] / cf[:, 0] - 1.0)[:len(real)]
                A_all = np.concatenate([best_a, np.zeros((nb, 1))], 1)
                P_all = np.concatenate([phi, phi0], 1)
                O_all, W_all = np.concatenate([best_o, wo], 1), np.concatenate([best_t, wt], 1)
                k = O_all.argmax(1)
                r = np.arange(nb)
                chosen[m] = (A_all[r, k], P_all[r, k])
                for key, arr in (("alpha", A_all), ("phi", P_all), ("own", O_all), ("true", W_all)):
                    out[key][m][real] = arr[r, k][:len(real)]
            # the TRUE policy at the action chosen by each belief
            out["true_action"][truth][real] = out["true"][truth][real]
            others = [m for m in order if m != truth]
            if others:
                alpha = np.stack([chosen[m][0] for m in others], 1)
                phi = np.stack([chosen[m][1] for m in others], 1)
                zero = alpha == 0.0
                yB, yS, x0 = incomes(par, aB, aA, np.where(zero, 0.5, alpha), np.where(zero, phis[0], phi))
                _, wt = action_stage(cw_truth, J(pB_truth), J(yB), J(yS), J(x0), J(bt), J(ot), J(bt), J(ot))
                wt = np.asarray(wt)
                if zero.any():
                    if par.xi > 0:
                        yB, yS, x0 = incomes(par, aB, aA, np.zeros((nb, 1)), np.full((nb, 1), phis[0]))
                        _, w0 = action_stage(cw_truth, J(pB_truth), J(yB), J(yS), J(x0), J(bt), J(ot), J(bt), J(ot))
                    else:
                        q_t, _ = conditional_probabilities(bt, ot)
                        w0 = no_income(J(q_t), J(pB_truth), J(bt), J(ot), J(bt), J(ot))[1][:, None]
                    wt = np.where(zero, np.asarray(w0), wt)
                for j, m in enumerate(others):
                    out["true_action"][m][real] = wt[:len(real), j]
            if log:
                log(f"T = {T}: {min(lo_i + nb, len(members))} / {len(members)} couples")
    return out


def value_at_actions(par: Params, laws, horizons, alpha, phi, truth="TRUE", pricing="POP", belief=None):
    """Value under the truth of a consumption policy when the annuity choice is imposed.
    alpha, phi: arrays (N, A), A imposed actions per couple. Returns (N, A).
    belief None (default): the consumption is the best one under the truth. Otherwise the consumption
    is the one that is optimal under the law laws[belief]: the policy of a household that believes it."""
    alpha, phi = np.asarray(alpha, dtype=float), np.asarray(phi, dtype=float)
    N, A = alpha.shape
    out = np.full((N, A), np.nan)
    widow_stage, action_stage, no_income = make_solver(par)
    Tb = np.maximum(20, 10 * np.ceil(horizons / 10.0)).astype(int)
    J = lambda x: jnp.asarray(x)
    for T in sorted(set(Tb.tolist())):
        members = np.where(Tb == T)[0]
        nb = int(np.clip(4e8 // (2 * T * T * par.Gw * 8), 4, 64))
        for lo_i in range(0, len(members), nb):
            real = members[lo_i:lo_i + nb]
            idx = np.concatenate([real, np.repeat(real[-1:], nb - len(real))])
            cut = lambda b: (laws[b][0][idx][:, :T + 1], laws[b][1][idx][:, :, :T, :T + 1])
            pb, po = cut(pricing)
            aB = annuity_due(pb, po, 0.0, par.r)
            aA = annuity_due(pb, po, 1.0, par.r) - aB
            bt, ot = cut(truth)
            bo, oo = (bt, ot) if belief is None else cut(belief)
            q, pB = conditional_probabilities(bo, oo)
            zero = (alpha[idx] == 0.0) & (par.xi == 0)                 # no income at all: closed form
            yB, yS, x0 = incomes(par, aB, aA, np.where(zero, 0.5, alpha[idx]), np.where(zero, par.phis[0], phi[idx]))
            _, w = action_stage(widow_stage(J(q)), J(pB), J(yB), J(yS), J(x0), J(bo), J(oo), J(bt), J(ot))
            w = np.asarray(w)
            if zero.any():
                w = np.where(zero, np.asarray(no_income(J(q), J(pB), J(bo), J(oo), J(bt), J(ot))[1])[:, None], w)
            out[real] = w[:len(real)]
    return out


def wealth_equivalent_loss(value_policy, value_best, par: Params):
    """Share of the initial resources the household would give up to move from the policy to the best
    one: the value is homogeneous of degree 1 - g in the resources."""
    return 1.0 - (value_policy / value_best) ** (1.0 / (1.0 - par.g))
