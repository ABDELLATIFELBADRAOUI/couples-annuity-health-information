"""
pipeline.py -- data preparation, nested model variants and exact predictions for the couples model.

Covariates (own)    : BMI, smoker, hypertension, diabetes, years of education, household wealth
Covariates (partner): partner's BMI and smoking
Standardisation, centring and the wealth transform use the TRAINING couples only.

Nested variants
    M0  independence : own covariates only
    M1  + partner    : adds the partner's covariates (gamma)
    M2  + widowhood  : adds the widowhood effect (omega)
    M3  full         : adds the shared frailty (kappa)
    MD  dependence without health: age and sex only, with the widowhood effect and the shared frailty
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from dyadic_mortality import CoupleData

MIN_AGE, C0 = 50, 20.0
OWN = ["bmi_z", "smoker", "hypertension", "diabetes", "education_z", "wealth_z"]
CROSS = ["partner_bmi_z", "partner_smoker"]
VARIANTS = {
    "M0": dict(use_cross=False, use_widowhood=False, use_frailty=False),
    "M1": dict(use_cross=True, use_widowhood=False, use_frailty=False),
    "M2": dict(use_cross=True, use_widowhood=True, use_frailty=False),
    "M3": dict(use_cross=True, use_widowhood=True, use_frailty=True),
    # dependence without health: age and sex only, with the widowhood effect and the shared frailty
    "MD": dict(use_covs=False, use_cross=False, use_widowhood=True, use_frailty=True),
}
# the split seed can be changed through the environment, to repeat the evaluation on other splits
SPLIT_SEED, TEST_FRACTION = int(os.environ.get("CARUP_SPLIT_SEED", "20261005")), 0.30


def load_couples(cohort_csv):
    d = pd.read_csv(cohort_csv)
    keep = ((d.age_h >= MIN_AGE) & (d.age_w >= MIN_AGE) & d.educ_h.notna() & d.educ_w.notna() & d.wealth.notna())
    return d[keep].reset_index(drop=True)


def split_indices(d, seed=SPLIT_SEED, test_fraction=TEST_FRACTION):
    """Random split of couples, stratified on the joint outcome (none / husband / wife / both died)."""
    rng = np.random.default_rng(seed)
    strata = d.died_h.astype(int).values + 2 * d.died_w.astype(int).values
    test = np.zeros(len(d), dtype=bool)
    for k in np.unique(strata):
        idx = np.where(strata == k)[0]
        rng.shuffle(idx)
        test[idx[: int(round(test_fraction * len(idx)))]] = True
    return np.where(~test)[0], np.where(test)[0]


def fit_scaler(d):
    """Standardisation and centring constants, from the couples given (the training couples)."""
    sc = {}
    for s in "hw":
        sc[f"bmi_{s}"] = (float(d[f"bmi_{s}"].mean()), float(d[f"bmi_{s}"].std()))
        sc[f"educ_{s}"] = (float(d[f"educ_{s}"].mean()), float(d[f"educ_{s}"].std()))
    w = np.arcsinh(d.wealth.values / 1000.0)
    sc["wealth"] = (float(w.mean()), float(w.std()))
    own = _raw_own(d, sc)
    sc["centre"] = own.mean(axis=0)                                  # (2, P)
    return sc


def _raw_own(d, sc):
    wz = (np.arcsinh(d.wealth.values / 1000.0) - sc["wealth"][0]) / sc["wealth"][1]
    cols = []
    for s in "hw":
        cols.append(np.stack([(d[f"bmi_{s}"].values - sc[f"bmi_{s}"][0]) / sc[f"bmi_{s}"][1],
                              d[f"smoker_{s}"].values.astype(float), d[f"hyp_{s}"].values.astype(float),
                              d[f"diab_{s}"].values.astype(float),
                              (d[f"educ_{s}"].values - sc[f"educ_{s}"][0]) / sc[f"educ_{s}"][1], wz], axis=1))
    return np.stack(cols, axis=1)                                    # (N, 2, P)


def make_data(d, sc):
    own = _raw_own(d, sc) - sc["centre"][None]
    cross = own[:, ::-1, :][:, :, [0, 1]]
    return CoupleData(age0=d[["age_h", "age_w"]].values.astype(int), x_own=own, x_cross=cross,
                      n_surv=d[["n_surv_h", "n_surv_w"]].values.astype(int),
                      died=d[["died_h", "died_w"]].values.astype(bool),
                      K=int(d[["n_surv_h", "n_surv_w"]].values.max()) + 1)


def prior_arrays(prior_csv, age_max):
    ages = np.arange(MIN_AGE, age_max + 1)
    prior = pd.read_csv(prior_csv)
    pstar = np.stack([prior[prior.sex == sx].set_index("age").loc[ages, "q_gompertz"].values for sx in ["men", "women"]])
    return ages, pstar, np.maximum(C0, 1.0 / pstar)


def age_max_of(d):
    last = np.where(d[["died_h", "died_w"]].values, d[["n_surv_h", "n_surv_w"]].values,
                    d[["n_surv_h", "n_surv_w"]].values - 1)
    return int((d[["age_h", "age_w"]].values + last).max())


# ----------------------------------------------------------------------------- outcomes
def outcomes(d):
    """(time, event) in years for the four targets, in discrete time.

    A death in follow-up year k (0-based) has time k + 1. A spouse censored after n completed years
    is known alive at time n and cannot die before time n + 1: the censoring time is n + 0.5.
    joint_life    : first death. It is observed when the earliest death occurs no later than one
                    year after the censored spouse's last known-alive time.
    last_survivor : second death, observed only if both died; otherwise the couple is known to have
                    a survivor until the censoring time of a spouse censored alive.
    """
    n = d[["n_surv_h", "n_surv_w"]].values.astype(float)
    e = d[["died_h", "died_w"]].values.astype(bool)
    death = np.where(e, n + 1.0, np.inf)                 # death time, inf if not observed
    alive_until = np.where(e, np.inf, n)                 # last known-alive time of a censored spouse
    t = np.where(e, n + 1.0, n + 0.5)
    out = {"husband": (t[:, 0], e[:, 0]), "wife": (t[:, 1], e[:, 1])}
    first_death, first_cens = death.min(axis=1), alive_until.min(axis=1)
    ev = first_death <= first_cens + 1.0
    out["joint_life"] = (np.where(ev, first_death, first_cens + 0.5), ev)
    both = e.all(axis=1)
    one_cens = np.where(~e[:, 0] & ~e[:, 1], n.max(axis=1), np.where(~e[:, 0], n[:, 0], n[:, 1]))
    out["last_survivor"] = (np.where(both, death.max(axis=1), one_cens + 0.5), both)
    return out


# ----------------------------------------------------------------------------- predictions
def predict_states(draws, data, age_min, horizon, chunk=20):
    """Exact posterior-predictive survival curves of each couple after 1..horizon years.

    draws: dict with p0 (D, 2, n_ages), beta (D, 2, P), gamma (D, 2, Q), omega (D, 2), kappa (D,)
           (kappa = 0 means no frailty).
    Returns four arrays (N, horizon): husband, wife, joint_life (both alive), last_survivor
    (at least one alive), averaged over the draws.

    Given the frailty Z, with lam0 the yearly hazard increment while the partner is alive and lam1
    the one after the partner's death (widowhood effect from the year after that death):
        P(both alive after t years | Z)    = exp(-Z A[t]),  A[t] = sum_{u<t} (lam0_h[u] + lam0_w[u])
        P(husband only after t years | Z)  = sum_{d<t} exp(-Z a[d, t]) (1 - exp(-Z lam0_w[d]))
            a[d, t] = A[d] + lam0_h[d] + sum_{u=d+1}^{t-1} lam1_h[u]     (wife dies in year d)
    Every term is an exponential in Z, so the frailty integrates exactly through its Laplace
    transform psi(u) = (1 + kappa u)^(-1/kappa): no simulation and no quadrature.
    """
    D, N, T = draws["kappa"].shape[0], data.age0.shape[0], horizon
    n_ages = draws["p0"].shape[2]
    theta = np.log(-np.log1p(-draws["p0"]))                                      # (D, 2, n_ages)
    lin = (np.einsum("nsp,dsp->dns", data.x_own, draws["beta"])
           + np.einsum("nsq,dsq->dns", data.x_cross, draws["gamma"]))            # (D, N, 2)
    idx = np.clip(data.age0[:, :, None] + np.arange(T)[None, None, :] - age_min, 0, n_ages - 1)   # (N, 2, T)
    lower = np.arange(T)[:, None] < (np.arange(T)[None, :] + 1)                  # [d, t-1]: d < t
    res = {k: np.zeros((N, T)) for k in ["husband", "wife", "joint_life", "last_survivor"]}
    for lo in range(0, D, chunk):
        sl = slice(lo, min(lo + chunk, D))
        kap = draws["kappa"][sl][:, None, None]

        def psi(u):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(kap > 1e-10, np.exp(-np.log1p(np.maximum(kap, 1e-300) * u) / np.maximum(kap, 1e-300)),
                                np.exp(-u))

        lam0, lam1 = [], []
        for s in range(2):
            eta = np.stack([theta[sl, s, :][:, idx[:, s, t]] for t in range(T)], axis=2) + lin[sl, :, s][:, :, None]
            lam0.append(np.exp(eta))                                             # (d, N, T)
            lam1.append(np.exp(eta + draws["omega"][sl, s][:, None, None]))
        zeros = np.zeros_like(lam0[0][:, :, :1])
        A = np.concatenate([zeros, np.cumsum(lam0[0] + lam0[1], axis=2)], axis=2)        # A[t], t = 0..T
        both = psi(A[:, :, 1:]) if kap.ndim == 3 else None                               # after 1..T years
        only = []
        for s in range(2):
            o = 1 - s
            C1 = np.concatenate([zeros, np.cumsum(lam1[s], axis=2)], axis=2)             # C1[t] = sum_{u<t} lam1[u]
            B = A[:, :, :T] + lam0[s] - C1[:, :, 1:]                                     # B[d], d = 0..T-1
            a = B[:, :, :, None] + C1[:, :, None, 1:]                                    # a[d, t-1] for t = 1..T
            a = np.where(lower[None, None], a, 0.0)                                      # entries with d >= t are not used
            kap4 = kap[:, :, :, None]

            def psi4(u):
                return np.where(kap4 > 1e-10, np.exp(-np.log1p(np.maximum(kap4, 1e-300) * u) / np.maximum(kap4, 1e-300)),
                                np.exp(-u))

            term = psi4(a) - psi4(a + lam0[o][:, :, :, None])
            only.append((term * lower[None, None]).sum(axis=2))                          # (d, N, T)
        res["joint_life"] += both.sum(axis=0)
        res["husband"] += (both + only[0]).sum(axis=0)
        res["wife"] += (both + only[1]).sum(axis=0)
        res["last_survivor"] += (both + only[0] + only[1]).sum(axis=0)
    return {k: v / D for k, v in res.items()}


def load_draws(files, n_draws=200, seed=0):
    """Pool the chains of several npz files and take n_draws equally spaced draws."""
    parts = [np.load(f) for f in files]
    out = {}
    first = parts[0]
    n_total = sum(p["p0"].shape[0] * p["p0"].shape[1] for p in parts)
    take = np.linspace(0, n_total - 1, n_draws).round().astype(int)
    for k in ["p0", "beta", "gamma", "omega", "kappa"]:
        if k in first.files:
            x = np.concatenate([p[k].reshape((-1,) + p[k].shape[2:]) for p in parts], axis=0)
            out[k] = x[take]
    D = out["p0"].shape[0]
    out.setdefault("beta", np.zeros((D, 2, len(OWN))))                # variant without covariates
    out.setdefault("gamma", np.zeros((D, 2, len(CROSS))))
    out.setdefault("omega", np.zeros((D, 2)))
    out.setdefault("kappa", np.zeros(D))
    return out
