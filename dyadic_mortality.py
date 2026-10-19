"""
dyadic_mortality.py -- Bayesian dyadic mortality layer for the CA-RUP rebuild.

Model (annual discrete time, attained-age time scale)
-----------------------------------------------------
Couple i, spouse s in {0: husband, 1: wife}, follow-up year t = 0..K-1,
attained age a = age0[i, s] + t.

    eta_ist = theta_s[a] + x_own[i, s] . beta_s + x_cross[i, s] . gamma_s + omega_s * W_ist
    P(death in year t | alive at start of t, Z_i, past) = 1 - exp(-Z_i * exp(eta_ist))

    theta_s[a] = log(-log(1 - p0_s[a]))      p0_s[a]: one-year death probability at age a
                                               for the reference profile (x = 0, Z = 1)
    p0_s[a] ~ Beta(c * pstar_s[a], c * (1 - pstar_s[a]))
              -> one urn per age and sex: c * pstar red balls, c * (1 - pstar) green balls
    Z_i ~ Gamma(shape = 1/kappa, rate = 1/kappa)   shared frailty, mean 1, variance kappa
    W_ist = 1 if the partner died in a year strictly before t (widowhood effect)

Given Z_i and the past, the two spouses' deaths within a year are independent.
The frailty is integrated out exactly with psi(u) = E[exp(-Z u)] = (1 + kappa u)^(-1/kappa):
    no death observed     : psi(A)
    one death (spouse s)  : psi(A) - psi(A + b_s)
    both deaths           : psi(A) - psi(A + b_h) - psi(A + b_w) + psi(A + b_h + b_w)
where A = sum of exp(eta) over the years survived by both spouses and b_s = exp(eta) in
spouse s's year of death. Right censoring enters only through the years at risk.

Without covariates, widowhood or frailty, the posterior of p0_s[a] is the reinforced urn:
Beta(c * pstar + deaths at age a, c * (1 - pstar) + survivals at age a).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

OWN_COVS = ("bmi_z", "smoker", "hypertension", "diabetes")
CROSS_COVS = ("partner_bmi_z", "partner_smoker")
CROSS_FROM_OWN = (0, 1)  # partner covariates taken from OWN_COVS columns 0 and 1


# ----------------------------------------------------------------------------- data
@dataclass
class CoupleData:
    age0: np.ndarray      # (N, 2) int, age at baseline
    x_own: np.ndarray     # (N, 2, P) own covariates
    x_cross: np.ndarray   # (N, 2, Q) partner covariates, already aligned to each spouse
    n_surv: np.ndarray    # (N, 2) int, number of years survived under observation
    died: np.ndarray      # (N, 2) bool, death observed in year n_surv
    K: int                # maximum number of follow-up years


def build_arrays(data: CoupleData, age_min: int, age_max: int) -> dict:
    """Person-year arrays of shape (N, 2, K) used by the likelihood."""
    N, K = data.age0.shape[0], data.K
    t = np.arange(K)[None, None, :]
    n_surv = data.n_surv[:, :, None]
    died = data.died[:, :, None]
    surv_mask = (t < n_surv).astype(float)
    event_mask = (died & (t == n_surv)).astype(float)
    partner_n = data.n_surv[:, ::-1][:, :, None]
    partner_died = data.died[:, ::-1][:, :, None]
    W = (partner_died & (partner_n < t)).astype(float)   # partner died strictly before year t
    age = data.age0[:, :, None] + t
    at_risk = (surv_mask + event_mask) > 0
    if (age[at_risk] < age_min).any() or (age[at_risk] > age_max).any():
        raise ValueError("age grid does not cover all ages at risk")
    age_idx = np.clip(age - age_min, 0, age_max - age_min)
    return dict(age_idx=age_idx.astype(np.int64), surv_mask=surv_mask,
                event_mask=event_mask, W=W,
                x_own=data.x_own.astype(float), x_cross=data.x_cross.astype(float))


# ------------------------------------------------------------------- numerics
def log1mexp(x):
    """log(1 - exp(-x)) for x > 0, stable over the whole range."""
    return jnp.where(x < jnp.log(2.0), jnp.log(-jnp.expm1(-x)), jnp.log1p(-jnp.exp(-x)))


def couple_loglik(lam, surv_mask, event_mask, kappa, use_frailty=True):
    """Exact marginal log-likelihood of each couple (frailty integrated out). lam: (N, 2, K)."""
    A = jnp.sum(surv_mask * lam, axis=(1, 2))                 # (N,)
    b = jnp.sum(event_mask * lam, axis=2)                     # (N, 2)
    dh = jnp.sum(event_mask[:, 0, :], axis=1) > 0
    dw = jnp.sum(event_mask[:, 1, :], axis=1) > 0
    # safe values so that unselected branches never produce inf/nan (and nan gradients)
    bh = jnp.where(dh, b[:, 0], 1.0)
    bw = jnp.where(dw, b[:, 1], 1.0)

    if use_frailty:
        def logpsi(u):
            return -jnp.log1p(kappa * u) / kappa
    else:
        def logpsi(u):
            return -u

    lp_A = logpsi(A)
    lp_Ah = logpsi(A + bh)
    lp_Aw = logpsi(A + bw)
    lp_Ahw = logpsi(A + bh + bw)

    ll_none = lp_A
    ll_h = lp_A + log1mexp(lp_A - lp_Ah)          # log[psi(A) - psi(A + bh)]
    ll_w = lp_A + log1mexp(lp_A - lp_Aw)
    log_D2 = lp_Aw + log1mexp(lp_Aw - lp_Ahw)     # log[psi(A + bw) - psi(A + bh + bw)]
    ll_hw = ll_h + log1mexp(ll_h - log_D2)        # log[D(A, bh) - D(A + bw, bh)]
    return jnp.where(dh & dw, ll_hw, jnp.where(dh, ll_h, jnp.where(dw, ll_w, ll_none)))


def linear_predictor(p0, beta, gamma, omega, arr):
    theta = jnp.log(-jnp.log1p(-p0))                              # (2, n_ages)
    sex = jnp.arange(2)[None, :, None]
    base = theta[sex, arr["age_idx"]]                              # (N, 2, K)
    own = jnp.einsum("nsp,sp->ns", arr["x_own"], beta)[:, :, None]
    cross = jnp.einsum("nsq,sq->ns", arr["x_cross"], gamma)[:, :, None]
    return base + own + cross + omega[None, :, None] * arr["W"]


# ------------------------------------------------------------------- Bayesian model
def model(arr, pstar, c, use_covs=True, use_widowhood=True, use_frailty=True,
          prior_scale=1.0, kappa_scale=1.0, use_cross=True, prior_family="normal"):
    """prior_family: law of the log hazard ratios, "normal" (default) or "student" (Student t with
    3 degrees of freedom, heavy tails), both with scale prior_scale."""
    n_ages = pstar.shape[1]
    with numpyro.plate("sex", 2, dim=-2):
        with numpyro.plate("age", n_ages, dim=-1):
            p0 = numpyro.sample("p0", dist.Beta(c * pstar, c * (1.0 - pstar)))
    P, Q = arr["x_own"].shape[-1], arr["x_cross"].shape[-1]
    beta, gamma = jnp.zeros((2, P)), jnp.zeros((2, Q))
    effect = dist.Normal(0.0, prior_scale) if prior_family == "normal" else dist.StudentT(3.0, 0.0, prior_scale)
    if use_covs:
        beta = numpyro.sample("beta", effect.expand([2, P]).to_event(2))
        if use_cross:
            gamma = numpyro.sample("gamma", effect.expand([2, Q]).to_event(2))
    if use_widowhood:
        omega = numpyro.sample("omega", effect.expand([2]).to_event(1))
    else:
        omega = jnp.zeros(2)
    kappa = numpyro.sample("kappa", dist.HalfNormal(kappa_scale)) if use_frailty else 0.0
    lam = jnp.exp(linear_predictor(p0, beta, gamma, omega, arr))
    ll = couple_loglik(lam, arr["surv_mask"], arr["event_mask"], kappa, use_frailty)
    numpyro.factor("loglik", jnp.sum(ll))


# ------------------------------------------------------------------- simulation
def gompertz_q(ages, q65, slope):
    """One-year death probability from a Gompertz force of mortality (simulation setting)."""
    return 1.0 - np.exp(-q65 * np.exp(slope * (ages - 65.0)))


def default_truth(age_min, age_max):
    """Simulation settings (not estimates from any dataset)."""
    ages = np.arange(age_min, age_max + 1)
    return dict(
        ages=ages,
        p0=np.stack([gompertz_q(ages, 0.015, 0.09), gompertz_q(ages, 0.010, 0.10)]),
        beta=np.array([[0.10, 0.50, 0.15, 0.35],
                       [0.08, 0.45, 0.12, 0.30]]),
        gamma=np.array([[0.05, 0.15],
                        [0.05, 0.20]]),
        omega=np.array([0.30, 0.20]),
        kappa=0.5,
    )


def simulate_couples(n, K, truth, seed=0, drop_rate=0.02, age_mean=66.0):
    """Simulate observed couple histories under the model above (couple-level censoring)."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    age_h = np.clip(np.round(rng.normal(age_mean, 9.0, n)), 50, 90).astype(int)
    age_w = np.clip(age_h - 3 + np.round(rng.normal(0.0, 4.0, n)), 45, 90).astype(int)
    age0 = np.stack([age_h, age_w], axis=1)
    bmi = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.2], [0.2, 1.0]], n)
    latent = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], n)
    smoker = (latent > norm.ppf(1 - 0.13)).astype(float)
    hyp = rng.binomial(1, [0.55, 0.50], (n, 2)).astype(float)
    diab = rng.binomial(1, [0.22, 0.17], (n, 2)).astype(float)
    x_own = np.stack([bmi, smoker, hyp, diab], axis=-1)            # (n, 2, 4)
    x_cross = x_own[:, ::-1, :][:, :, list(CROSS_FROM_OWN)]       # partner's bmi, smoker
    kappa = truth["kappa"]
    Z = rng.gamma(1.0 / kappa, kappa, n) if kappa > 0 else np.ones(n)
    C = np.minimum(rng.geometric(drop_rate, n), K)                # years observed per couple
    age_min = int(truth["ages"][0])
    theta = np.log(-np.log1p(-truth["p0"]))
    lin = (np.einsum("nsp,sp->ns", x_own, truth["beta"])
           + np.einsum("nsq,sq->ns", x_cross, truth["gamma"]))
    death_year = np.full((n, 2), -1)
    alive = np.ones((n, 2), dtype=bool)
    for t in range(K):
        observed = (t < C)[:, None]
        partner_dead = death_year[:, ::-1]
        W = ((partner_dead >= 0) & (partner_dead < t)).astype(float)
        eta = theta[np.arange(2)[None, :], age0 + t - age_min] + lin + truth["omega"][None, :] * W
        h = 1.0 - np.exp(-Z[:, None] * np.exp(eta))
        die = alive & observed & (rng.random((n, 2)) < h)
        death_year[die] = t
        alive &= ~die
    died = death_year >= 0
    n_surv = np.where(died, death_year, C[:, None])
    return CoupleData(age0=age0, x_own=x_own, x_cross=x_cross, n_surv=n_surv, died=died, K=K)


def urn_posterior(data: CoupleData, age_min, age_max, pstar, c):
    """Closed-form reinforced-urn posterior (no covariates, no widowhood, no frailty)."""
    arr = build_arrays(data, age_min, age_max)
    n_ages = age_max - age_min + 1
    deaths, surv = np.zeros((2, n_ages)), np.zeros((2, n_ages))
    for s in range(2):
        idx = arr["age_idx"][:, s, :]
        np.add.at(deaths[s], idx[arr["event_mask"][:, s, :] > 0], 1.0)
        np.add.at(surv[s], idx[arr["surv_mask"][:, s, :] > 0], 1.0)
    return c * pstar + deaths, c * (1.0 - pstar) + surv
