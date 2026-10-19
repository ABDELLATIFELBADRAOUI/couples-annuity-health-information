"""
fit_sens.py -- the full model (M3) on the training couples under other prior choices.

Usage: python fit_sens.py <setting> <rng_seed> <out.npz>          (2 chains, 1000 + 1000 iterations)

Settings (one change at a time from the reference fit):
    base         reference: strength c(a) = max(20, 1 / p*(a)), prior guess p* of prior_table.csv, Normal(0, 1)
                 on the log hazard ratios, HalfNormal(1) on the frailty variance (same model, data and seed
                 as results_eval/M3_train_a.npz: the two fits must coincide)
    c_quarter,   strength of every urn divided by 4 or multiplied by 4 (same prior guess). With c / 4 the
    c_times4     urns of the youngest ages hold less than one "death" ball.
    centre_low,  prior guess of the urns multiplied by 0.7 or by 1.3 (a baseline mortality 30 % too low
    centre_high  or too high), with the strength of the reference
    scale05,     Normal(0, 0.5) or Normal(0, 2) on the log hazard ratios
    scale2
    student      Student t (3 degrees of freedom, scale 1) on the log hazard ratios: heavy tails
    kappa05,     HalfNormal(0.5) or HalfNormal(2) on the frailty variance
    kappa2
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS

from dyadic_mortality import build_arrays, model
from pipeline import VARIANTS, load_couples, split_indices, fit_scaler, make_data, prior_arrays, age_max_of

WARMUP, DRAWS = 1000, 1000
SETTINGS = {
    "base": dict(), "c_quarter": dict(strength=0.25), "c_times4": dict(strength=4.0), "centre_low": dict(centre=0.7), "centre_high": dict(centre=1.3),
    "scale05": dict(prior_scale=0.5), "scale2": dict(prior_scale=2.0), "student": dict(prior_family="student"),
    "kappa05": dict(kappa_scale=0.5), "kappa2": dict(kappa_scale=2.0),
}

if __name__ == "__main__":
    name, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    s = dict(SETTINGS[name])
    d = load_couples("cohort_w10.csv")
    ages, pstar, c = prior_arrays("prior_table.csv", age_max_of(d))
    c = c * s.pop("strength", 1.0)                                    # strength of the reference, scaled as a whole
    pstar = np.clip(pstar * s.pop("centre", 1.0), 1e-6, 0.95)        # the centre changes, not the strength
    train_idx, _ = split_indices(d)
    d_fit = d.iloc[train_idx].reset_index(drop=True)
    data = make_data(d_fit, fit_scaler(d_fit))
    arr = {k: jnp.asarray(v) for k, v in build_arrays(data, int(ages[0]), int(ages[-1])).items()}
    print(f"{name}: couples {len(d_fit)}, settings {SETTINGS[name]}", flush=True)
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=WARMUP, num_samples=DRAWS, num_chains=2,
                chain_method="sequential", progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), arr, jnp.asarray(pstar), jnp.asarray(c), extra_fields=("diverging", "num_steps"),
             **VARIANTS["M3"], **s)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    np.savez(out, elapsed=time.time() - t0, ages=ages, pstar=pstar, c=c, n_couples=len(d_fit), **extra, **samples)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
