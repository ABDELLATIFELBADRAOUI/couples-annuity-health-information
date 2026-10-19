"""
sim_study.py -- replicated simulation study of the couples model.

For one scenario and one replication: lives of the training couples are simulated from the model with
known parameters, the full model M3 is fitted to the simulated data with the same priors as on the real
data, and the posterior is compared with the truth.

Design. The real ages and covariates are kept. A couple is followed as long as in the real data: until
the last year in which a spouse was observed alive when a spouse is censored, and until the
administrative end of follow-up (11 years) when both spouses died, since the follow-up they would have
had is not known.

Scenarios (truth). The baseline mortality and the effects of the own covariates are always the
posterior means of the fit on the real training couples (stored once in results_sim/truth.npz). The
effects of the partner's covariates are kept as well, except in "indep". The scenarios differ by the
widowhood effect and the shared frailty:
    base      widowhood effect and frailty variance at their posterior means
    indep     no dependence at all: no partner effect, no widowhood effect, no frailty
    frailty   shared frailty (variance 0.5), no widowhood effect
    widow     widowhood effect (hazard ratio 1.5 for both sexes), no frailty
    both      frailty variance 0.5 and widowhood hazard ratio 1.5
    small     base, with 1,000 couples (the same 1,000 in every replication)
    short     base, with at most 6 years of follow-up (more censoring)
    base_ref, indep_ref   the data sets of "base" and "indep" refitted with the protocol of the real data
              (4 chains, 1,000 + 1,000 iterations), to check the shorter protocol used elsewhere

Protocol of the other scenarios: 2 chains, 500 warm-up iterations and 500 draws each.

Usage: python sim_study.py <scenario> <first replication> <last replication>
Output: results_sim/<scenario>_<replication>.npz (posterior summaries, truth, diagnostics)
"""
import glob
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import MCMC, NUTS

from dyadic_mortality import CoupleData, build_arrays, model
from pipeline import VARIANTS, load_couples, split_indices, fit_scaler, make_data, prior_arrays, age_max_of, predict_states

SHORT, REFERENCE = (2, 500, 500), (4, 1000, 1000)                      # chains, warm-up iterations, draws per chain
SCENARIOS = {
    "base": dict(), "indep": dict(gamma=0.0, omega=0.0, kappa=0.0), "frailty": dict(omega=0.0, kappa=0.5),
    "widow": dict(omega=float(np.log(1.5)), kappa=0.0), "both": dict(omega=float(np.log(1.5)), kappa=0.5),
    "small": dict(n=1000), "short": dict(years=6),
    "base_ref": dict(data="base", protocol=REFERENCE), "indep_ref": dict(gamma=0.0, omega=0.0, kappa=0.0, data="indep", protocol=REFERENCE),
}
AGES_SHOWN = np.array([55, 65, 75, 85])
TRUTH_FILE = "results_sim/truth.npz"


def save_npz_atomic(path, **arrays):
    """np.savez under a temporary name, then renamed: a reader never sees a half-written file."""
    tmp = f"{path[:-4]}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def truth_from_fit():
    """Posterior means of the full model fitted on the real training couples; computed once and stored."""
    if os.path.exists(TRUTH_FILE):
        z = np.load(TRUTH_FILE)
        return dict(p0=z["p0"], beta=z["beta"], gamma=z["gamma"], omega=z["omega"], kappa=float(z["kappa"]))
    parts = [np.load(f) for f in sorted(glob.glob("results_eval/M3_train_*.npz"))]
    assert len(parts) == 2, "the two chain files of the full model are expected"
    mean = lambda k: np.concatenate([p[k].reshape((-1,) + p[k].shape[2:]) for p in parts], axis=0).mean(axis=0)
    truth = dict(p0=mean("p0"), beta=mean("beta"), gamma=mean("gamma"), omega=mean("omega"), kappa=float(mean("kappa")))
    save_npz_atomic(TRUTH_FILE, **truth)
    return truth


def simulate(data: CoupleData, truth, age_min, years, rng):
    """Lives of the couples of `data` under the model with parameters `truth`; couple n is followed years[n] years."""
    N = data.age0.shape[0]
    theta = np.log(-np.log1p(-truth["p0"]))
    lin = np.einsum("nsp,sp->ns", data.x_own, truth["beta"]) + np.einsum("nsq,sq->ns", data.x_cross, truth["gamma"])
    Z = rng.gamma(1.0 / truth["kappa"], truth["kappa"], N) if truth["kappa"] > 0 else np.ones(N)
    death = np.full((N, 2), -1)
    alive = np.ones((N, 2), dtype=bool)
    for t in range(int(years.max())):
        seen = (t < years)[:, None]
        widowed = ((death[:, ::-1] >= 0) & (death[:, ::-1] < t)).astype(float)
        age_idx = np.minimum(data.age0 + t - age_min, theta.shape[1] - 1)       # couples no longer followed are not at risk
        eta = theta[np.arange(2)[None, :], age_idx] + lin + truth["omega"][None, :] * widowed
        die = alive & seen & (rng.random((N, 2)) < 1.0 - np.exp(-Z[:, None] * np.exp(eta)))
        death[die] = t
        alive &= ~die
    died = death >= 0
    return CoupleData(age0=data.age0, x_own=data.x_own, x_cross=data.x_cross, n_surv=np.where(died, death, years[:, None]),
                      died=died, K=int(years.max()))


def follow_up(real: CoupleData, age_max, limit=None):
    """Years of follow-up of each couple in the simulated data (see the header)."""
    censored = ~real.died.all(axis=1)
    years = np.where(real.died, 0, real.n_surv).max(axis=1)                 # last year in which a censored spouse was seen alive
    years[~censored] = real.K - 1                                           # both died: administrative end of follow-up
    years = np.minimum(years, age_max - real.age0.max(axis=1) + 1)          # never beyond the last age of the urns
    return np.minimum(years, real.K if limit is None else limit).astype(int)


def summary(x):
    """x: (chains, draws, ...) -> mean, sd, quantiles, split R-hat and effective sample size of each component."""
    flat = x.reshape((-1,) + x.shape[2:])
    q = np.percentile(flat, [2.5, 50.0, 97.5], axis=0)
    return dict(mean=flat.mean(axis=0), sd=flat.std(axis=0, ddof=1), q025=q[0], q50=q[1], q975=q[2], rhat=np.asarray(split_gelman_rubin(x)),
                ess=np.asarray(effective_sample_size(x)))


if __name__ == "__main__":
    scenario, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs("results_sim", exist_ok=True)
    d = load_couples("cohort_w10.csv")
    ages, pstar, c = prior_arrays("prior_table.csv", age_max_of(d))
    age_min, age_max = int(ages[0]), int(ages[-1])
    train_idx, _ = split_indices(d)
    d_fit = d.iloc[train_idx].reset_index(drop=True)
    real = make_data(d_fit, fit_scaler(d_fit))
    setting = dict(SCENARIOS[scenario])
    chains, warmup, draws = setting.pop("protocol", SHORT)
    data_of = setting.pop("data", scenario)                                 # scenario whose simulated data sets are used
    truth = truth_from_fit()
    for k in ("gamma", "omega"):
        if k in setting:
            truth[k] = np.full_like(truth[k], setting[k])
    truth["kappa"] = float(setting.get("kappa", truth["kappa"]))
    years = follow_up(real, age_max, setting.get("years"))
    n_all = real.age0.shape[0]
    keep = np.sort(np.random.default_rng(2026).choice(n_all, setting["n"], replace=False)) if "n" in setting else np.arange(n_all)
    design = CoupleData(age0=real.age0[keep], x_own=real.x_own[keep], x_cross=real.x_cross[keep], n_surv=real.n_surv[keep],
                        died=real.died[keep], K=real.K)
    sub = np.sort(np.random.default_rng(5).choice(len(keep), 300, replace=False))        # couples whose predictions are compared
    dsub = CoupleData(age0=design.age0[sub], x_own=design.x_own[sub], x_cross=design.x_cross[sub], n_surv=design.n_surv[sub],
                      died=design.died[sub], K=design.K)
    code = list(SCENARIOS).index(data_of) + 1
    for rep in range(first, last + 1):
        out = f"results_sim/{scenario}_{rep:03d}.npz"
        if os.path.exists(out):
            continue
        t0 = time.time()
        data = simulate(design, truth, age_min, years[keep], np.random.default_rng(100000 * code + rep))
        arr = {k: jnp.asarray(v) for k, v in build_arrays(data, age_min, age_max).items()}
        mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warmup, num_samples=draws, num_chains=chains,
                    chain_method="sequential", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(rep + 7919 * code), arr, jnp.asarray(pstar), jnp.asarray(c), extra_fields=("diverging",), **VARIANTS["M3"])
        s = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
        res = dict(scenario=np.array(scenario), rep=rep, n_couples=len(keep), deaths=data.died.sum(axis=0), both_died=int(data.died.all(axis=1).sum()),
                   person_years=int((data.n_surv + data.died).sum()), divergences=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
                   chains=chains, warmup=warmup, draws=draws)
        for k in ("beta", "gamma", "omega", "kappa"):
            res.update({f"{k}_{name}": v for name, v in summary(s[k]).items()})
            res[f"{k}_true"] = np.asarray(truth[k])
        shown = AGES_SHOWN - age_min
        res.update({f"p0_{name}": v for name, v in summary(s["p0"][:, :, :, shown]).items()})
        res["p0_true"] = truth["p0"][:, shown]
        res["p0_rhat_max"] = float(np.max(np.asarray(split_gelman_rubin(s["p0"]))))                   # over every urn
        pooled = {k: v.reshape((-1,) + v.shape[2:]) for k, v in s.items()}
        res["corr_omega_kappa"] = np.array([np.corrcoef(pooled["omega"][:, j], pooled["kappa"])[0, 1] for j in range(2)])
        # accuracy of the predicted 10-year probabilities on 300 couples, against the probabilities under the truth
        take = np.linspace(0, pooled["kappa"].shape[0] - 1, 100).round().astype(int)
        est = predict_states({k: pooled[k][take] for k in ("p0", "beta", "gamma", "omega", "kappa")}, dsub, age_min, 10)
        tru = predict_states({k: np.asarray(truth[k])[None] for k in ("p0", "beta", "gamma", "omega")} | {"kappa": np.array([truth["kappa"]])},
                             dsub, age_min, 10)
        for tg in ("husband", "wife", "joint_life", "last_survivor"):
            diff = est[tg][:, 9] - tru[tg][:, 9]
            res[f"pred_bias_{tg}"], res[f"pred_mae_{tg}"] = float(diff.mean()), float(np.abs(diff).mean())
        res["seconds"] = time.time() - t0
        save_npz_atomic(out, **res)
        print(f"{scenario} {rep}: deaths {res['deaths']}, both {res['both_died']}, divergences {res['divergences']}, max R-hat "
              f"{max(max(float(np.max(res[k + '_rhat'])) for k in ('beta', 'gamma', 'omega', 'kappa')), res['p0_rhat_max']):.3f}, "
              f"kappa {float(res['kappa_mean']):.3f} (true {truth['kappa']:.3f}), {res['seconds']:.0f}s", flush=True)
