"""
decision_setup.py -- the laws of the couples' histories under each belief, for the held-out couples.

    TRUE : full model M3 (partner covariates, widowhood effect, shared frailty), posterior predictive
    M0   : independence model (own covariates only), posterior predictive
    M1   : M0 + partner covariates          (optional)
    M2   : M1 + widowhood effect            (optional)
    MARG : the two marginal laws of TRUE, combined as if the spouses were independent (same survival of
           each spouse as under TRUE, no dependence at all)
    POP  : age and sex only, spouses independent: reinforced urns of the training couples, in closed form
    MD   : age and sex only, with dependence (widowhood effect and shared frailty), posterior predictive
    STRONG : the full model with a stronger dependence imposed on every posterior draw (widowhood hazard
           ratio and frailty variance of STRONG below, the values of the simulation study); a stress case
    MARGS: the two marginal laws of STRONG, combined as if the spouses were independent

The models are those fitted on the training couples of the main split; the couples studied are the
held-out ones, whose covariates did not enter the estimation. The laws are cached in
results_decision/cache/. Without a limiting age, a decision problem solved over T years treats the
couple as extinct after T years although the law still gives a small probability of being alive.
"""
import glob
import os

import numpy as np

from dyadic_mortality import urn_posterior
from pipeline import (load_couples, split_indices, fit_scaler, make_data, prior_arrays, age_max_of, load_draws, OWN)
from decision_law import joint_law, population_draws

AGE_END, N_DRAWS = 110, 400
CACHE = "results_decision/cache"
SOURCES = {"TRUE": "M3", "M0": "M0", "M1": "M1", "M2": "M2", "MD": "MD", "STRONG": "M3"}
STRONG = dict(hazard_ratio=1.5, kappa=0.5)


def save_npz_atomic(path, **arrays):
    """np.savez under a temporary name, then renamed: a reader never sees a half-written file."""
    tmp = f"{path[:-4]}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def held_out():
    d = load_couples("cohort_w10.csv")
    tr, te = split_indices(d)
    dtr, dte = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)
    sc = fit_scaler(dtr)
    ages, pstar, c = prior_arrays("prior_table.csv", age_max_of(d))
    return dtr, dte, make_data(dtr, sc), make_data(dte, sc), ages, pstar, c


def horizon(dte):
    """Years until the younger spouse reaches AGE_END, for each couple."""
    return (AGE_END - np.minimum(dte.age_h.values, dte.age_w.values)).astype(int)


def law(belief, age_limit=None):
    """(both, only) for the held-out couples of the main split, over the longest horizon; cached.
    age_limit: limiting age imposed on the law (None: none, see decision_law.joint_law)."""
    if "CARUP_SPLIT_SEED" in os.environ:
        raise RuntimeError("the cached laws are those of the main split: unset CARUP_SPLIT_SEED")
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/law_{belief}{'' if age_limit is None else f'_limit{age_limit}'}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return z["both"], z["only"]
    if belief in ("MARG", "MARGS"):
        both, only = independent_law(*law("TRUE" if belief == "MARG" else "STRONG", age_limit))
        save_npz_atomic(path, both=both, only=only)
        return both, only
    dtr, dte, data_tr, data_te, ages, pstar, c = held_out()
    T = int(horizon(dte).max())
    if belief == "POP":
        a, b = urn_posterior(data_tr, int(ages[0]), int(ages[-1]), pstar, c)
        draws = population_draws(a, b, len(OWN))
    else:
        draws = load_draws(sorted(glob.glob(f"results_eval/{SOURCES[belief]}_train_*.npz")), n_draws=N_DRAWS)
        if belief == "STRONG":
            draws["omega"] = np.full_like(draws["omega"], np.log(STRONG["hazard_ratio"]))
            draws["kappa"] = np.full_like(draws["kappa"], STRONG["kappa"])
    both, only = joint_law(draws, data_te, int(ages[0]), T, age_limit=age_limit)
    save_npz_atomic(path, both=both, only=only)
    return both, only


def independent_law(both, only):
    """Law with the same marginal survival of each spouse, the two dates of death being independent."""
    T = both.shape[1] - 1
    surv = both[:, None, :] + only.sum(axis=2)                       # (N, 2, T+1): P(spouse s alive after t years)
    dens = surv[:, :, :-1] - surv[:, :, 1:]                          # (N, 2, T):   P(spouse s dies in year d)
    both_i = surv[:, 0] * surv[:, 1]
    lower = np.arange(T)[:, None] < np.arange(T + 1)[None, :]        # [d, t]: d < t
    only_i = np.stack([dens[:, 1 - s, :, None] * surv[:, s, None, :] * lower[None] for s in range(2)], axis=1)
    return both_i, only_i
