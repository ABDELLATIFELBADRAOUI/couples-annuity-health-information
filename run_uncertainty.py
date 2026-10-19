"""
run_uncertainty.py -- how much the losses of the decision analysis move with the parameters of the
mortality model.

In the main analysis the truth is the posterior predictive law of the full model. Here the truth is,
in turn, each of N_TRUTH posterior draws of its parameters. For each draw:
    - the optimum is recomputed (a household that knows these parameters);
    - the households of the main analysis keep their decisions and their consumption policies, which do
      not know the draw: POP (age and sex only), MD (dependence without health), M0 (health without
      dependence), MARG (the marginal laws of the main analysis, spouses independent) and PP (the
      posterior predictive law of the full model, the "truth" of the main analysis). The policies are
      valued under the draw.
The spread over the draws of the mean loss is the uncertainty that comes from the estimation of the
mortality model. The loss of PP is the cost of not knowing the parameters exactly.

Couples: the 300 held-out couples of the sensitivity analyses. Base case, limiting age of 110.
Usage: python run_uncertainty.py                  -> results_decision/uncertainty.npz
"""
import glob
import time

import numpy as np

from decision_setup import law, held_out, horizon, N_DRAWS
from decision_law import joint_law
from decision_dp import Params, solve_couples, value_at_actions, wealth_equivalent_loss
from dyadic_mortality import CoupleData
from pipeline import load_draws
from run_decision import SUBSAMPLE

N_TRUTH, AGE_LIMIT = 30, 110
FIXED = {"PP": "TRUE", "MARG": "MARG", "M0": "M0", "MD": "MD", "POP": "POP"}        # name here -> belief of results_decision/base_limit.npz

if __name__ == "__main__":
    par = Params()
    dtr, dte, data_tr, data_te, ages, pstar, c = held_out()
    H = horizon(dte)
    index = np.sort(np.random.default_rng(11).choice(len(H), SUBSAMPLE, replace=False))
    sub = CoupleData(age0=data_te.age0[index], x_own=data_te.x_own[index], x_cross=data_te.x_cross[index],
                     n_surv=data_te.n_surv[index], died=data_te.died[index], K=data_te.K)
    T = int(H.max())
    draws = load_draws(sorted(glob.glob("results_eval/M3_train_*.npz")), n_draws=N_DRAWS)
    picked = np.linspace(0, N_DRAWS - 1, N_TRUTH).round().astype(int)
    belief_law, action = {}, {}
    z = np.load("results_decision/base_limit.npz")
    assert np.array_equal(z["index"], np.arange(len(H))), "base_limit is expected for all the held-out couples, in their order"
    for name, source in FIXED.items():
        b, o = law(source, AGE_LIMIT)
        belief_law[name] = (b[index], o[index])
        action[name] = (z[f"alpha|{source}"][index], z[f"phi|{source}"][index])
    pop = belief_law["POP"]
    names = list(FIXED)
    loss = {m: np.full((N_TRUTH, SUBSAMPLE), np.nan) for m in names}
    best = np.full((N_TRUTH, SUBSAMPLE), np.nan)
    t0 = time.time()
    for j, dr in enumerate(picked):
        one = {k: v[dr:dr + 1] for k, v in draws.items()}
        truth = joint_law(one, sub, int(ages[0]), T, age_limit=AGE_LIMIT)
        out = solve_couples(par, {"TRUE": truth, "POP": pop}, ["TRUE"], H[index])
        best[j] = out["true"]["TRUE"]
        for name in FIXED:
            a, p = action[name]
            v = value_at_actions(par, {"TRUE": truth, "POP": pop, "B": belief_law[name]}, H[index], a[:, None], np.nan_to_num(p, nan=par.phis[0])[:, None],
                                 belief="B")[:, 0]
            loss[name][j] = 100 * wealth_equivalent_loss(v, best[j], par)
        print(f"draw {j + 1} / {N_TRUTH}: " + ", ".join(f"{m} {loss[m][j].mean():.3f}" for m in names) + f"  {time.time() - t0:.0f}s", flush=True)
    np.savez("results_decision/uncertainty.npz", index=index, draws=picked, best=best, seconds=time.time() - t0,
             **{f"loss|{m}": v for m, v in loss.items()})
