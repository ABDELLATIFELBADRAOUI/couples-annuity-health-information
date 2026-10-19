"""Without covariates, widowhood or frailty, the NUTS posterior of p0 must equal the reinforced-urn Beta posterior."""
import time
import numpy as np
import jax, jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
from dyadic_mortality import default_truth, simulate_couples, build_arrays, model, urn_posterior

age_min, age_max = 45, 101
truth = default_truth(age_min, age_max)
data = simulate_couples(3000, 12, truth, seed=11)
pstar = np.clip(truth["p0"] * 1.3, 1e-4, 0.9)        # deliberately misspecified prior guess
c = 20.0
arr = {k: jnp.asarray(v) for k, v in build_arrays(data, age_min, age_max).items()}
mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=1000, num_chains=2, chain_method="sequential", progress_bar=False)
t0 = time.time()
mcmc.run(jax.random.PRNGKey(0), arr, jnp.asarray(pstar), c, use_covs=False, use_widowhood=False, use_frailty=False)
p = np.asarray(mcmc.get_samples()["p0"])               # (S, 2, n_ages); waits for JAX to finish
print(f"NUTS time: {time.time()-t0:.1f}s")
a, b = urn_posterior(data, age_min, age_max, pstar, c)
mean_exact = a / (a + b)
sd_exact = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
z = np.abs(p.mean(0) - mean_exact) / sd_exact
ratio = p.std(0) / sd_exact
# Urns whose posterior holds less than one death ball (a < 1: no death observed at that age and a
# weak prior) have a very long tail on the logit scale; NUTS mixes slowly there, so they are
# reported separately.
for label, m in [("urns with a >= 1", a >= 1), ("urns with a <  1", a < 1)]:
    if m.any():
        print(f"[{label}: {int(m.sum()):3d} of {m.size}] max |posterior mean - urn mean| / urn sd: {z[m].max():.3f}"
              f"   posterior sd / urn sd: {ratio[m].min():.3f} - {ratio[m].max():.3f}")
for s, name in enumerate(["men", "women"]):
    for age in [60, 70, 80, 90]:
        j = age - age_min
        print(f"{name:5s} age {age}: NUTS {p[:, s, j].mean():.5f} ± {p[:, s, j].std():.5f} | urn Beta {mean_exact[s, j]:.5f} ± {sd_exact[s, j]:.5f}")
