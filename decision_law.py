"""
decision_law.py -- exact law of a couple's survival history under a mortality model.

A couple's history is fully described by: both alive after t years, or spouse s alone after t years
with the partner dead since year d. The two arrays below hold the probability of every such history;
with them a decision problem can be solved and evaluated without any Markov approximation.

    both[n, t]        P(both alive after t years),                                 t = 0..T
    only[n, s, d, t]  P(spouse s alive after t years, partner died in year d),     d = 0..T-1, t = d+1..T
                      (s = 0 husband, 1 wife; zero when t <= d)

Given the frailty Z, every term is an exponential in Z (see pipeline.predict_states), so the frailty
integrates exactly through its Laplace transform psi(u) = (1 + kappa u)^(-1/kappa). The arrays are
averaged over the posterior draws: they are the posterior-predictive law.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _psi(u, kap):
    safe = jnp.maximum(kap, 1e-300)
    return jnp.where(kap > 1e-10, jnp.exp(-jnp.log1p(safe * u) / safe), jnp.exp(-u))


CERTAIN = 1e6      # hazard increment that makes death in the year certain


def _law_one_draw(theta, beta, gamma, omega, kappa, x_own, x_cross, idx, last_year, T):
    """theta (2, n_ages), beta (2, P), gamma (2, Q), omega (2,), kappa (); idx (N, 2, T) age index;
    last_year (N, 2, T) bool: years in which death is certain (limiting age)."""
    lin = jnp.einsum("nsp,sp->ns", x_own, beta) + jnp.einsum("nsq,sq->ns", x_cross, gamma)      # (N, 2)
    eta = jnp.stack([theta[s][idx[:, s, :]] for s in range(2)], axis=1) + lin[:, :, None]       # (N, 2, T)
    lam0 = jnp.where(last_year, CERTAIN, jnp.exp(eta))                                           # partner alive
    lam1 = jnp.where(last_year, CERTAIN, jnp.exp(eta + omega[None, :, None]))                    # widowed
    zeros = jnp.zeros_like(lam0[:, 0, :1])
    A = jnp.concatenate([zeros, jnp.cumsum(lam0[:, 0] + lam0[:, 1], axis=1)], axis=1)            # (N, T+1)
    both = _psi(A, kappa)
    lower = jnp.arange(T)[:, None] < jnp.arange(T + 1)[None, :]                                  # [d, t]: d < t
    only = []
    for s in range(2):
        o = 1 - s
        C1 = jnp.concatenate([zeros, jnp.cumsum(lam1[:, s], axis=1)], axis=1)                    # C1[t] = sum_{u<t} lam1[u]
        B = A[:, :T] + lam0[:, s] - C1[:, 1:]                                                    # B[d]
        a = jnp.where(lower[None], B[:, :, None] + C1[:, None, :], 0.0)                          # a[d, t]
        term = _psi(a, kappa) - _psi(a + lam0[:, o, :, None], kappa)
        only.append(jnp.where(lower[None], term, 0.0))
    return both, jnp.stack(only, axis=1)                                                         # (N, T+1), (N, 2, T, T+1)


@jax.jit
def _law_chunk(theta, beta, gamma, omega, kappa, x_own, x_cross, idx, last_year):
    T = idx.shape[2]
    b, o = jax.vmap(lambda th, be, ga, om, ka: _law_one_draw(th, be, ga, om, ka, x_own, x_cross, idx, last_year, T))(
        theta, beta, gamma, omega, kappa)
    return b.sum(axis=0), o.sum(axis=0)


def joint_law(draws, data, age_min, T, chunk=4, age_limit=None):
    """draws: dict with p0 (D, 2, n_ages), beta, gamma, omega, kappa (kappa = 0: no frailty).
    age_limit: limiting age; nobody is alive at that exact age (death is certain in the year of age
    age_limit - 1). None: no limiting age, ages above the last urn reuse the last urn."""
    D, n_ages = draws["kappa"].shape[0], draws["p0"].shape[2]
    theta = np.log(-np.log1p(-draws["p0"]))
    age = data.age0[:, :, None] + np.arange(T)[None, None, :]
    idx = np.clip(age - age_min, 0, n_ages - 1)
    last_year = jnp.asarray(age >= age_limit - 1 if age_limit is not None else np.zeros_like(age, dtype=bool))
    x_own, x_cross, idx = jnp.asarray(data.x_own), jnp.asarray(data.x_cross), jnp.asarray(idx)
    both = np.zeros((data.age0.shape[0], T + 1))
    only = np.zeros((data.age0.shape[0], 2, T, T + 1))
    for lo in range(0, D, chunk):
        sl = slice(lo, min(lo + chunk, D))
        b, o = _law_chunk(jnp.asarray(theta[sl]), jnp.asarray(draws["beta"][sl]), jnp.asarray(draws["gamma"][sl]),
                          jnp.asarray(draws["omega"][sl]), jnp.asarray(draws["kappa"][sl]), x_own, x_cross, idx, last_year)
        both += np.asarray(b)
        only += np.asarray(o)
    both, only = both / D, only / D
    if age_limit is not None:
        # exact zeros beyond the limiting age, so that every law has the same support
        can = (data.age0[:, :, None] + np.arange(T + 1)[None, None, :]) < age_limit      # can be alive after t years
        both = both * (can[:, 0] & can[:, 1])
        for s in range(2):
            only[:, s] *= can[:, s, None, :] & can[:, 1 - s, :T, None]                   # the partner was alive at the start of year d
    return both, only


def population_draws(a, b, n_cov, n_cross=2):
    """One 'draw' for the population belief: the urn posterior means, no covariates, no dependence.
    The posterior mean is the exact predictive as long as each urn is used once, because the urns of
    different ages and sexes are independent; above the last urn (age 104), where one urn is reused,
    it is an approximation."""
    return dict(p0=(a / (a + b))[None], beta=np.zeros((1, 2, n_cov)), gamma=np.zeros((1, 2, n_cross)),
                omega=np.zeros((1, 2)), kappa=np.zeros(1))
