"""Independent check of the closed-form couple likelihood against Monte Carlo integration over the frailty."""
import numpy as np
import jax.numpy as jnp
from dyadic_mortality import (default_truth, simulate_couples, build_arrays,
                              linear_predictor, couple_loglik)

age_min, age_max = 45, 101
truth = default_truth(age_min, age_max)
data = simulate_couples(4000, 12, truth, seed=7)
arr = build_arrays(data, age_min, age_max)
jarr = {k: jnp.asarray(v) for k, v in arr.items()}
lam = np.asarray(jnp.exp(linear_predictor(jnp.asarray(truth["p0"]), jnp.asarray(truth["beta"]),
                                          jnp.asarray(truth["gamma"]), jnp.asarray(truth["omega"]), jarr)))
kappa = truth["kappa"]
closed = np.asarray(couple_loglik(jnp.asarray(lam), jarr["surv_mask"], jarr["event_mask"], kappa, True))

A = (arr["surv_mask"] * lam).sum(axis=(1, 2))
b = (arr["event_mask"] * lam).sum(axis=2)
dead = arr["event_mask"].sum(axis=2) > 0
rng = np.random.default_rng(123)
Z = rng.gamma(1.0 / kappa, kappa, 2_000_000)

cases = {"no death": ~dead[:, 0] & ~dead[:, 1], "husband only": dead[:, 0] & ~dead[:, 1],
         "wife only": ~dead[:, 0] & dead[:, 1], "both": dead[:, 0] & dead[:, 1]}
worst = 0.0
for name, mask in cases.items():
    idx = np.where(mask)[0][:4]
    for i in idx:
        L = np.exp(-Z * A[i])
        if dead[i, 0]:
            L = L * (1 - np.exp(-Z * b[i, 0]))
        if dead[i, 1]:
            L = L * (1 - np.exp(-Z * b[i, 1]))
        mc, se = L.mean(), L.std() / np.sqrt(Z.size)
        rel = abs(np.exp(closed[i]) - mc) / mc
        worst = max(worst, rel / (se / mc))
        print(f"{name:13s} couple {i:5d}: closed={np.exp(closed[i]):.6e}  MC={mc:.6e} ± {se:.1e}  rel.diff={rel:.2e}")
print(f"largest |closed - MC| in Monte Carlo standard errors: {worst:.2f}")

# no-frailty path must equal the Bernoulli product
closed0 = np.asarray(couple_loglik(jnp.asarray(lam), jarr["surv_mask"], jarr["event_mask"], 0.0, False))
direct0 = (-(arr["surv_mask"] * lam).sum(axis=(1, 2))
           + (arr["event_mask"] * np.log(-np.expm1(-np.where(arr["event_mask"] > 0, lam, 1.0)))).sum(axis=(1, 2)))
print("no-frailty path, max abs diff vs Bernoulli product:", np.abs(closed0 - direct0).max())
