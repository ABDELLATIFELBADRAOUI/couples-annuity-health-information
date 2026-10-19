"""
fit_models.py -- fit one nested variant of the couples model.

Usage: python fit_models.py <M0|M1|M2|M3|MD> <train|full> <rng_seed> <n_chains> <out.npz>
The cohort file is cohort_w10.csv, or the file named by the environment variable CARUP_COHORT
(used for the analysis with a shorter follow-up).

train : the 70 % of couples kept for estimation (split defined in pipeline.py)
full  : all couples
"""
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS

from dyadic_mortality import build_arrays, model
from pipeline import VARIANTS, load_couples, split_indices, fit_scaler, make_data, prior_arrays, age_max_of

WARMUP, DRAWS = 1000, 1000

variant, subset, seed, n_chains, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
d = load_couples(os.environ.get("CARUP_COHORT", "cohort_w10.csv"))
ages, pstar, c = prior_arrays("prior_table.csv", age_max_of(d))
if subset == "train":
    train_idx, _ = split_indices(d)
    d_fit = d.iloc[train_idx].reset_index(drop=True)
else:
    d_fit = d
sc = fit_scaler(d_fit)
data = make_data(d_fit, sc)
arr = {k: jnp.asarray(v) for k, v in build_arrays(data, int(ages[0]), int(ages[-1])).items()}
print(f"{variant} on {subset}: couples {len(d_fit)}, deaths husbands {int(data.died[:, 0].sum())} "
      f"wives {int(data.died[:, 1].sum())}", flush=True)

mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=WARMUP, num_samples=DRAWS,
            num_chains=n_chains, chain_method="sequential", progress_bar=False)
t0 = time.time()
mcmc.run(jax.random.PRNGKey(seed), arr, jnp.asarray(pstar), jnp.asarray(c),
         extra_fields=("diverging", "num_steps"), **VARIANTS[variant])
samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
elapsed = time.time() - t0
np.savez(out, elapsed=elapsed, ages=ages, pstar=pstar, c=c, n_couples=len(d_fit), **extra, **samples)
print(f"done in {elapsed:.0f}s", flush=True)
