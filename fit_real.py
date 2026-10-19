"""
fit_real.py -- fit the dyadic mortality model on the real couples cohort.

Usage: python fit_real.py cohort_w10.csv prior_table.csv <rng_seed> <n_chains> <out.npz>

Cohort   : couples whose two spouses are aged 50 or more at Wave 10 (the HRS target population).
Prior    : p*(a) = Gompertz curve fitted on HRS 1998-2010 (build_prior.py), one urn per age and sex,
           with strength c(a) = max(C0, 1 / p*(a)), so that every urn starts with at least one
           death ball: Beta(c p*, c (1 - p*)).
Model    : see dyadic_mortality.py (own effects beta, partner effects gamma, widowhood omega,
           shared gamma frailty of variance kappa).
"""
import sys
import time

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS

from dyadic_mortality import CoupleData, build_arrays, model

MIN_AGE, C0 = 50, 20.0
WARMUP, DRAWS = 1000, 1000


def load(cohort_csv, prior_csv):
    d = pd.read_csv(cohort_csv)
    d = d[(d.age_h >= MIN_AGE) & (d.age_w >= MIN_AGE)].reset_index(drop=True)
    stats = {}
    for s in "hw":
        stats[s] = (float(d[f"bmi_{s}"].mean()), float(d[f"bmi_{s}"].std()))
        d[f"bmiz_{s}"] = (d[f"bmi_{s}"] - stats[s][0]) / stats[s][1]
    own = np.stack([np.stack([d[f"bmiz_{s}"], d[f"smoker_{s}"], d[f"hyp_{s}"], d[f"diab_{s}"]], axis=1)
                    for s in "hw"], axis=1).astype(float)                       # (N, 2, 4)
    # Covariates are centred within sex, so that the baseline p0(a) is the hazard of an average
    # husband or wife: this is the quantity the prior guess p*(a) (a population rate) describes.
    stats["cov_means"] = own.mean(axis=0)                                       # (2, 4)
    own = own - stats["cov_means"][None]
    cross = own[:, ::-1, :][:, :, [0, 1]]                                       # partner's bmi_z, smoker
    data = CoupleData(age0=d[["age_h", "age_w"]].values.astype(int), x_own=own, x_cross=cross,
                      n_surv=d[["n_surv_h", "n_surv_w"]].values.astype(int),
                      died=d[["died_h", "died_w"]].values.astype(bool),
                      K=int(d[["n_surv_h", "n_surv_w"]].values.max()) + 1)
    died_year = np.where(data.died, data.n_surv, data.n_surv - 1)               # last year at risk
    age_max = int((data.age0 + died_year).max())
    ages = np.arange(MIN_AGE, age_max + 1)
    prior = pd.read_csv(prior_csv)
    pstar = np.stack([prior[prior.sex == sx].set_index("age").loc[ages, "q_gompertz"].values
                      for sx in ["men", "women"]])
    c = np.maximum(C0, 1.0 / pstar)
    return data, ages, pstar, c, stats


if __name__ == "__main__":
    cohort_csv, prior_csv, seed, n_chains, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    data, ages, pstar, c, stats = load(cohort_csv, prior_csv)
    arr = {k: jnp.asarray(v) for k, v in build_arrays(data, int(ages[0]), int(ages[-1])).items()}
    print(f"couples {data.age0.shape[0]}, K {data.K}, ages {ages[0]}-{ages[-1]}, "
          f"deaths husbands {int(data.died[:, 0].sum())} wives {int(data.died[:, 1].sum())}", flush=True)
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=WARMUP, num_samples=DRAWS,
                num_chains=n_chains, chain_method="sequential", progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), arr, jnp.asarray(pstar), jnp.asarray(c),
             extra_fields=("diverging", "num_steps"))
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    elapsed = time.time() - t0
    np.savez(out, elapsed=elapsed, ages=ages, pstar=pstar, c=c, n_couples=data.age0.shape[0],
             bmi_mean_sd=np.array([stats["h"], stats["w"]]), cov_means=stats["cov_means"], **extra, **samples)
    print(f"done in {elapsed:.0f}s", flush=True)
