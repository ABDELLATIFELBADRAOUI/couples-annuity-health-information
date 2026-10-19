"""
copula_baseline.py -- the classical model of dependent lives: Frank copula with Gompertz margins
(Frees, Carriere and Valdez 1996), with proportional-hazards covariates, as a benchmark.

Margins : force of mortality of spouse s at age x, mu_s(x | z) = exp(a_s + b_s (x - 70) + z . beta_s),
          z: the six own covariates of pipeline.py (the "+partner" variant adds the partner's BMI and smoking).
          S_s(x) = P(alive at age x | alive at 50).
Copula  : P(X > x, Y > y) = C_theta(S_h(x), S_w(y)), Frank family; theta > 0: positive dependence.
Data    : annual. A couple enters with both spouses alive at ages (a_h, a_w): the likelihood is
          conditional on that event (left truncation). A death in follow-up year k lies between ages
          a + k and a + k + 1; a spouse alive after n years is right-censored at age a + n. Every case is
          a difference of values of the joint survival function at the corners of a rectangle.
Estimation: maximum likelihood (L-BFGS, gradients by JAX), standard errors from the Hessian.
Also fitted: theta = 0 (independent Gompertz margins), to isolate what the copula adds.

Usage: python copula_baseline.py           (results folder: results_eval or CARUP_RESULTS_DIR; split: CARUP_SPLIT_SEED;
                                            CARUP_COPULA_RECOVERY=0 skips the recovery check on simulated data)
Output: <folder>/copula_predictions.npz (survival after 1..12 years of the held-out couples, read by
        evaluate.py) and <folder>/copula_fit.json (estimates and a recovery check on simulated data).
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from pipeline import load_couples, split_indices, fit_scaler, make_data, SPLIT_SEED

jax.config.update("jax_enable_x64", True)
RES = os.environ.get("CARUP_RESULTS_DIR", "results_eval")
HORIZON, ORIGIN = 12, 50.0


def frank(u, v, theta):
    """Frank copula C_theta(u, v); theta = 0 is independence."""
    safe = jnp.where(jnp.abs(theta) < 1e-8, 1.0, theta)
    c = -jnp.log1p(jnp.expm1(-safe * u) * jnp.expm1(-safe * v) / jnp.expm1(-safe)) / safe
    return jnp.where(jnp.abs(theta) < 1e-8, u * v, c)


def unpack(par, P, free_theta):
    a, b = par[0:2], jnp.exp(par[2:4])                               # b > 0
    beta = par[4:4 + 2 * P].reshape(2, P)
    theta = par[4 + 2 * P] if free_theta else 0.0
    return a, b, beta, theta


def margin(age, s, lin, a, b):
    """S_s(age) given alive at ORIGIN, for couples with linear predictor lin."""
    H = jnp.exp(a[s] + lin) * (jnp.exp(b[s] * (age - 70.0)) - jnp.exp(b[s] * (ORIGIN - 70.0))) / b[s]
    return jnp.exp(-H)


def joint(age_h, age_w, lin, a, b, theta):
    return frank(margin(age_h, 0, lin[:, 0], a, b), margin(age_w, 1, lin[:, 1], a, b), theta)


def loglik(par, X, age0, n_surv, died, free_theta):
    """X: (N, 2, P) covariates; age0, n_surv: (N, 2); died: (N, 2) bool."""
    a, b, beta, theta = unpack(par, X.shape[2], free_theta)
    lin = jnp.einsum("nsp,sp->ns", X, beta)
    lo = age0 + n_surv                                               # alive at this age
    hi = lo + 1.0 * died                                             # dead by this age when died (same as lo when censored)
    S = lambda x, y: joint(x, y, lin, a, b, theta)
    # rectangle (lo_h, hi_h] x (lo_w, hi_w]; a censored spouse contributes the half-line beyond lo
    p = (S(lo[:, 0], lo[:, 1]) - jnp.where(died[:, 0], S(hi[:, 0], lo[:, 1]), 0.0) - jnp.where(died[:, 1], S(lo[:, 0], hi[:, 1]), 0.0)
         + jnp.where(died[:, 0] & died[:, 1], S(hi[:, 0], hi[:, 1]), 0.0))
    return jnp.sum(jnp.log(jnp.maximum(p, 1e-300)) - jnp.log(S(age0[:, 0] * 1.0, age0[:, 1] * 1.0)))


def fit(X, data, free_theta, start=None):
    P = X.shape[2]
    n_par = 4 + 2 * P + int(free_theta)
    args = (jnp.asarray(X), jnp.asarray(data.age0, dtype=float), jnp.asarray(data.n_surv, dtype=float), jnp.asarray(data.died))
    f = jax.jit(lambda p: -loglik(p, *args, free_theta))
    g = jax.jit(jax.grad(lambda p: -loglik(p, *args, free_theta)))
    x0 = np.zeros(n_par)
    x0[0:2], x0[2:4] = np.log(0.02), np.log(0.09)
    if free_theta:
        x0[-1] = 2.0
    if start is not None:
        x0[:len(start)] = start
    res = minimize(lambda p: float(f(jnp.asarray(p))), x0, jac=lambda p: np.asarray(g(jnp.asarray(p))), method="L-BFGS-B",
                   options=dict(maxiter=2000, maxfun=5000, ftol=1e-13, gtol=1e-7))
    H = np.asarray(jax.hessian(lambda p: -loglik(p, *args, free_theta))(jnp.asarray(res.x)))
    se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0.0, None))
    return res.x, se, -float(res.fun), bool(res.success), float(np.abs(np.asarray(g(jnp.asarray(res.x)))).max())


def predict(par, X, age0, free_theta):
    """Survival after 1..HORIZON years given both alive at entry: husband, wife, both, at least one."""
    a, b, beta, theta = unpack(jnp.asarray(par), X.shape[2], free_theta)
    lin = jnp.einsum("nsp,sp->ns", jnp.asarray(X), beta)
    ah, aw = jnp.asarray(age0[:, 0], dtype=float), jnp.asarray(age0[:, 1], dtype=float)
    S0 = joint(ah, aw, lin, a, b, theta)
    t = jnp.arange(1, HORIZON + 1, dtype=float)
    h = jnp.stack([joint(ah + k, aw, lin, a, b, theta) / S0 for k in t], axis=1)
    w = jnp.stack([joint(ah, aw + k, lin, a, b, theta) / S0 for k in t], axis=1)
    both = jnp.stack([joint(ah + k, aw + k, lin, a, b, theta) / S0 for k in t], axis=1)
    return {"husband": np.asarray(h), "wife": np.asarray(w), "joint_life": np.asarray(both), "last_survivor": np.asarray(h + w - both)}


def kendall_tau(theta, n=200001):
    """Kendall's tau of the Frank copula: 1 - 4 (1 - D1(theta)) / theta, D1 the Debye function."""
    if abs(theta) < 1e-8:
        return 0.0
    x = np.linspace(1e-12, theta, n)
    y = x / np.expm1(x)
    d1 = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))) / theta
    return 1.0 - 4.0 * (1.0 - d1) / theta


def simulate(par, X, data, rng):
    """Annual histories from the fitted copula model, same ages and lengths of follow-up as `data`
    (conditional inverse of the Frank copula, then rejection of the couples not both alive at entry)."""
    a, b, beta, theta = [np.asarray(v) for v in unpack(jnp.asarray(par), X.shape[2], True)]
    lin = np.einsum("nsp,sp->ns", X, beta)
    N = X.shape[0]
    age = np.zeros((N, 2))
    todo = np.arange(N)
    while len(todo):
        u, p = rng.random(len(todo)), rng.random(len(todo))
        v = -np.log1p(p * np.expm1(-theta) / (np.exp(-theta * u) * (1.0 - p) + p)) / theta      # conditional law of V given U = u
        for s, q in ((0, u), (1, v)):
            H = -np.log(q)
            age[todo, s] = 70.0 + np.log(H * b[s] / np.exp(a[s] + lin[todo, s]) + np.exp(b[s] * (ORIGIN - 70.0))) / b[s]
        todo = todo[(age[todo] <= data.age0[todo]).any(axis=1)]
    years = np.where(data.died, 0, data.n_surv).max(axis=1)
    years = np.where(data.died.all(axis=1), data.K - 1, years)
    k = np.floor(age - data.age0).astype(int)
    died = k < years[:, None]
    return type(data)(age0=data.age0, x_own=data.x_own, x_cross=data.x_cross, n_surv=np.where(died, k, years[:, None]), died=died, K=data.K)


if __name__ == "__main__":
    d = load_couples("cohort_w10.csv")
    tr, te = split_indices(d)
    dtr, dte = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)
    sc = fit_scaler(dtr)
    data_tr, data_te = make_data(dtr, sc), make_data(dte, sc)
    design = lambda data, partner: np.concatenate([data.x_own, data.x_cross], axis=2) if partner else data.x_own
    pred, info = {}, {}
    for name, partner, free in (("Gompertz", False, False), ("Copula", False, True), ("Copula+partner", True, True)):
        Xtr, Xte = design(data_tr, partner), design(data_te, partner)
        start = info["Gompertz"]["estimate"] if name == "Copula" else None            # margins of the independent fit as a start
        est, se, ll, ok, gmax = fit(Xtr, data_tr, free, start)
        pred[name] = predict(est, Xte, data_te.age0, free)
        info[name] = dict(estimate=est.tolist(), se=se.tolist(), loglik=ll, converged=ok, max_gradient=gmax, n_parameters=len(est),
                          couples=int(Xtr.shape[0]))
        if free:
            info[name].update(theta=float(est[-1]), theta_se=float(se[-1]), kendall_tau=kendall_tau(float(est[-1])))
        print(f"{name}: log-likelihood {ll:.2f}, converged {ok}, largest gradient {gmax:.1e}"
              + (f", theta {est[-1]:.3f} (se {se[-1]:.3f}), Kendall tau {info[name]['kendall_tau']:.3f}" if free else ""), flush=True)
    info["Copula"]["lr_test_vs_independence"] = 2.0 * (info["Copula"]["loglik"] - info["Gompertz"]["loglik"])
    # recovery check: data simulated from the copula model (fitted margins; theta as fitted, then theta = 3), refitted
    est = np.array(info["Copula"]["estimate"])
    info["recovery"] = {}
    checks = (("theta_as_fitted", float(est[-1])), ("theta_3", 3.0)) if os.environ.get("CARUP_COPULA_RECOVERY", "1") == "1" else ()
    for label, theta_true in checks:
        truth = est.copy()
        truth[-1] = theta_true
        thetas, zs = [], []
        for r in range(12):
            sim = simulate(truth, data_tr.x_own, data_tr, np.random.default_rng(100 + r))
            est2, se2, _, _, _ = fit(data_tr.x_own, sim, True)                    # default starting values, not the truth
            thetas.append(float(est2[-1]))
            zs.append((est2 - truth) / se2)
        zs = np.array(zs)
        info["recovery"][label] = dict(replications=12, theta_true=theta_true, theta_mean=float(np.mean(thetas)), theta_sd=float(np.std(thetas, ddof=1)),
                                       largest_abs_mean_z=float(np.abs(zs.mean(axis=0)).max()), share_abs_z_above_1_96=float(np.mean(np.abs(zs) > 1.96)))
        print(f"recovery, theta {theta_true:.3f}: mean estimate {np.mean(thetas):.3f} (sd {np.std(thetas, ddof=1):.3f}) over 12 simulated data sets; "
              f"share of |z| above 1.96 over all parameters {np.mean(np.abs(zs) > 1.96):.3f}", flush=True)
    np.savez(f"{RES}/copula_predictions.npz", split_seed=SPLIT_SEED, **{f"{m}|{tg}": x for m, p in pred.items() for tg, x in p.items()})
    json.dump(info, open(f"{RES}/copula_fit.json", "w"), indent=1)
