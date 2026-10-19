"""check_horizon.py -- how the treatment of extreme ages changes the result for a household without income.

The couple with the largest loss of the scenario "loading of 20 %" chooses no annuity at all. Its
problem is solved again with three horizons without limiting age (60, 70 and 80 years) and with a
limiting age of 110. With CRRA utility and no income, the plan for the last years of life weighs a lot,
although these years have a very small probability.
"""
import glob

import numpy as np

from dyadic_mortality import CoupleData, urn_posterior
from pipeline import load_draws, OWN
from decision_setup import held_out, horizon, N_DRAWS
from decision_law import joint_law, population_draws
from decision_dp import Params, solve_couples, wealth_equivalent_loss

dtr, dte, data_tr, data_te, ages, pstar, c = held_out()
H = horizon(dte)
z = np.load("results_decision/load20.npz")
par = Params(load=0.20)
loss = 100 * wealth_equivalent_loss(z["true|M0"], z["true|TRUE"], par)
k = int(np.argmax(loss))
i = int(z["index"][k])
print(f"couple {i} (ages {dte.age_h.values[i]}/{dte.age_w.values[i]}), loading 20 %: loss with M0 in the scenario file {loss[k]:.3f} %")
pick = np.array([i])
sub = CoupleData(age0=data_te.age0[pick], x_own=data_te.x_own[pick], x_cross=data_te.x_cross[pick],
                 n_surv=data_te.n_surv[pick], died=data_te.died[pick], K=data_te.K)
a, b = urn_posterior(data_tr, int(ages[0]), int(ages[-1]), pstar, c)
draws = {"POP": population_draws(a, b, len(OWN)),
         "TRUE": load_draws(sorted(glob.glob("results_eval/M3_train_*.npz")), n_draws=N_DRAWS),
         "M0": load_draws(sorted(glob.glob("results_eval/M0_train_*.npz")), n_draws=N_DRAWS)}
for T, limit in ((60, None), (70, None), (80, None), (60, 110)):
    laws = {m: joint_law(d, sub, int(ages[0]), T, age_limit=limit) for m, d in draws.items()}
    out = solve_couples(par, laws, ["TRUE", "M0", "POP"], np.array([T]))
    alive = laws["TRUE"][0][0, T] + laws["TRUE"][1][0, :, :, T].sum()
    l0 = 100 * wealth_equivalent_loss(out["true"]["M0"][0], out["true"]["TRUE"][0], par)
    lp = 100 * wealth_equivalent_loss(out["true"]["POP"][0], out["true"]["TRUE"][0], par)
    print(f"horizon {T} years, limiting age {limit}: probability of being alive at the end under the truth {alive:.1e}; "
          f"share annuitised TRUE {out['alpha']['TRUE'][0]:.3f}, M0 {out['alpha']['M0'][0]:.3f}, POP {out['alpha']['POP'][0]:.3f}; "
          f"loss M0 {l0:.3f} %, POP {lp:.3f} %", flush=True)
