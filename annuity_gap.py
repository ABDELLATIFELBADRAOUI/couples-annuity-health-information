"""annuity_gap.py -- what the independence assumption changes in annuity values, on the held-out couples.

Annuity of 1 per year paid at the end of each year, discounted at 3 % (illustrative rate):
    joint life    : paid while both spouses are alive
    last survivor : paid while at least one spouse is alive
Survival is projected to age 110 with the attained-age baselines of the models fitted on the training
couples (ages above the last urn use the last urn).
Two comparisons:
    A. full model M3 against the independence model M0 (two separately fitted models);
    B. within M3: its joint survival against the product of its own marginal survivals.
"""
import glob
import json

import numpy as np

from pipeline import (load_couples, split_indices, fit_scaler, make_data, prior_arrays, age_max_of,
                      predict_states, load_draws)

RATE, N_DRAWS, AGE_END = 0.03, 200, 110
d = load_couples("cohort_w10.csv")
tr, te = split_indices(d)
sc = fit_scaler(d.iloc[tr])
dte = d.iloc[te].reset_index(drop=True)
data_te = make_data(dte, sc)
ages, _, _ = prior_arrays("prior_table.csv", age_max_of(d))
H = int(AGE_END - min(dte.age_h.min(), dte.age_w.min()))
v = (1.0 + RATE) ** -np.arange(1, H + 1)

# per posterior draw: portfolio annuity values (sum over the held-out couples)
def portfolio_values(model_name):
    draws = load_draws(sorted(glob.glob(f"results_eval/{model_name}_train_*.npz")), n_draws=N_DRAWS)
    rows, couple_mean = [], None
    for i in range(N_DRAWS):
        one = {k: x[i:i + 1] for k, x in draws.items()}
        P = predict_states(one, data_te, int(ages[0]), H, chunk=1)
        a = {k: (P[k] * v[None, :]).sum(axis=1) for k in P}                     # per couple
        indep = P["husband"] * P["wife"]
        a["joint_life_indep"] = (indep * v[None, :]).sum(axis=1)
        a["last_survivor_indep"] = ((P["husband"] + P["wife"] - indep) * v[None, :]).sum(axis=1)
        rows.append({k: x.sum() for k, x in a.items()})
        couple_mean = a if couple_mean is None else {k: couple_mean[k] + a[k] for k in a}
    return ({k: np.array([r[k] for r in rows]) for k in rows[0]}, {k: x / N_DRAWS for k, x in couple_mean.items()})


port, couple = {}, {}
for m in ["M0", "M3"]:
    port[m], couple[m] = portfolio_values(m)

out = {"setting": dict(couples=len(dte), rate=RATE, projection_to_age=AGE_END, posterior_draws=N_DRAWS)}


def describe(label, num_draws, den_draws, num_couples, den_couples):
    g = num_draws / den_draws - 1.0                       # portfolio gap, one value per posterior draw
    r = num_couples / den_couples - 1.0                   # gap of each couple (posterior-mean values)
    res = dict(portfolio_mean=float(g.mean()), portfolio_q025=float(np.percentile(g, 2.5)),
               portfolio_q975=float(np.percentile(g, 97.5)), couples_q05=float(np.percentile(r, 5)),
               couples_median=float(np.median(r)), couples_q95=float(np.percentile(r, 95)))
    print(f"{label:42s} portfolio {100 * res['portfolio_mean']:+.2f} % [{100 * res['portfolio_q025']:+.2f} ; "
          f"{100 * res['portfolio_q975']:+.2f}]   couples: median {100 * res['couples_median']:+.2f} %, "
          f"5 % to 95 %: {100 * res['couples_q05']:+.2f} % to {100 * res['couples_q95']:+.2f} %")
    return res


print(f"{len(dte)} held-out couples, rate {RATE:.0%}, projection to age {AGE_END}, {N_DRAWS} posterior draws")
print("mean annuity values under M0: " + ", ".join(
    f"{k} {couple['M0'][k].mean():.2f}" for k in ["husband", "wife", "joint_life", "last_survivor"]))
print("A. full model M3 against the independence model M0")
out["A_M3_vs_M0"] = {k: describe("   " + k, port["M3"][k], port["M0"][k], couple["M3"][k], couple["M0"][k])
                     for k in ["husband", "wife", "joint_life", "last_survivor"]}
print("B. within M3: joint survival against the product of its own marginals")
out["B_within_M3"] = {k: describe("   " + k, port["M3"][k], port["M3"][k + "_indep"], couple["M3"][k], couple["M3"][k + "_indep"])
                      for k in ["joint_life", "last_survivor"]}
json.dump(out, open("results_eval/annuity_gap.json", "w"), indent=2)
