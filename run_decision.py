"""
run_decision.py -- cost of a wrong mortality belief in the couple's retirement decision.

For every held-out couple, the truth is the full model (TRUE). A household that believes another
model chooses its annuity (share of wealth, survivor share) and its consumption as if that model were
true; its expected utility is then computed under the truth and compared with the best achievable.
The loss is expressed as a share of the initial resources (wealth-equivalent loss).

    M0  ignores the partner (two unrelated individuals)        -> cost of ignoring dependence
    MARG same survival of each spouse as the truth, spouses independent -> dependence alone
    M1  M0 + partner covariates;  M2  M1 + widowhood effect    -> which part of the dependence matters
    POP age and sex only                                         -> cost of ignoring the health state
    MD  age and sex only, with dependence                        -> what dependence is worth without the health state

Usage: python run_decision.py <scenario> [<scenario> ...]      (scenarios: see SCENARIOS below)
Output: results_decision/<scenario>.npz
The base case and the scenario with a pre-existing annuity use all held-out couples; the sensitivity
scenarios use the same 300 couples drawn at random among them. In base_truthM0_limit the truth is the
independence model M0 instead of the full model; in strong_limit it is the full model with a stronger
dependence imposed (stress case), and the belief MARGS ignores that dependence.
"""
import sys
import time

import numpy as np

from decision_setup import law, held_out, horizon
from decision_dp import Params, solve_couples

FREE_PHI = tuple(np.round(np.arange(0.40, 1.0001, 0.05), 2))
SUBSAMPLE = 300                         # couples drawn at random for the sensitivity scenarios (same couples in each)
SCENARIOS = {
    # name: (parameters that differ from the base case, beliefs compared with the truth, all couples?, limiting age)
    # scenarios without limiting age (histories cut at the horizon), kept for the comparison of the two conventions
    "base":      (dict(), ["M2", "M1", "M0", "POP"], True, None),
    "base_marg": (dict(), ["MARG"], True, None),                                     # base case, one more belief
    "pre50":     (dict(xi=0.5), ["MARG", "M0", "POP"], True, None),                  # half of the resources already annuitised
    "freephi":   (dict(phis=FREE_PHI), ["MARG", "M0", "POP"], False, None),          # survivor share free between 40 % and 100 %
    "g2":        (dict(g=2.0), ["M0", "POP"], False, None),                          # risk aversion 2
    "g5":        (dict(g=5.0), ["M0", "POP"], False, None),                          # risk aversion 5
    "load0":     (dict(load=0.0), ["M0", "POP"], False, None),                       # annuity without loading
    "load20":    (dict(load=0.20), ["M0", "POP"], False, None),                      # loading of 20 %
    # reference results: limiting age of 110 in every law (nobody alive at 110); all couples for the two main cases
    "base_limit":    (dict(), ["MARG", "M0", "M1", "M2", "MD", "POP"], True, 110),
    "pre50_limit":   (dict(xi=0.5), ["MARG", "M0", "MD", "POP"], True, 110),
    # sensitivity scenarios: the same 300 couples drawn at random
    "freephi_limit": (dict(phis=FREE_PHI), ["MARG", "M0", "MD", "POP"], False, 110),
    "g2_limit":      (dict(g=2.0), ["MARG", "M0", "MD", "POP"], False, 110),
    "g5_limit":      (dict(g=5.0), ["MARG", "M0", "MD", "POP"], False, 110),
    "load0_limit":   (dict(load=0.0), ["MARG", "M0", "MD", "POP"], False, 110),
    "load20_limit":  (dict(load=0.20), ["MARG", "M0", "MD", "POP"], False, 110),
    # the same sensitivity scenarios on all the held-out couples (optional: about five times longer)
    "g2_all":      (dict(g=2.0), ["MARG", "M0", "MD", "POP"], True, 110),
    "g5_all":      (dict(g=5.0), ["MARG", "M0", "MD", "POP"], True, 110),
    "load0_all":   (dict(load=0.0), ["MARG", "M0", "MD", "POP"], True, 110),
    "load20_all":  (dict(load=0.20), ["MARG", "M0", "MD", "POP"], True, 110),
    "freephi_all": (dict(phis=FREE_PHI), ["MARG", "M0", "MD", "POP"], True, 110),
    # another truth: the independence model M0 (health, no dependence at all); belief: age and sex only
    "base_truthM0_limit": (dict(), ["POP"], True, 110),
    # stress case: the truth is the full model with a stronger dependence (decision_setup.STRONG); beliefs: the same
    # survival of each spouse without the dependence (MARGS), and the two beliefs without health information (MD, POP).
    # Same 300 couples as the sensitivity scenarios.
    "strong_limit": (dict(), ["MARGS", "MD", "POP"], False, 110),
}
TRUTH = {"base_truthM0_limit": "M0", "strong_limit": "STRONG"}     # scenarios whose truth is not the full model

if __name__ == "__main__":
    names = sys.argv[1:] or ["base"]
    _, dte, *_ = held_out()
    H = horizon(dte)
    for name in names:
        changes, beliefs, everyone, age_limit = SCENARIOS[name]
        truth = TRUTH.get(name, "TRUE")
        assert truth == "TRUE" or "MARG" not in beliefs, "MARG is built from the full model: not available with another truth"
        assert truth == "STRONG" or "MARGS" not in beliefs, "MARGS is built from the stress case: not available with another truth"
        par = Params(**changes)
        index = np.arange(len(H)) if everyone else np.sort(np.random.default_rng(11).choice(len(H), SUBSAMPLE, replace=False))
        laws = {}
        for b in dict.fromkeys(["TRUE", "POP"] + beliefs):
            both, only = law(truth if b == "TRUE" else b, age_limit)
            laws[b] = (both[index], only[index])
        t0 = time.time()
        out = solve_couples(par, laws, ["TRUE"] + beliefs, H[index],
                            log=lambda msg: print(f"[{name}] {msg}  {time.time() - t0:.0f}s", flush=True))
        flat = {f"{key}|{m}": v for key, by_belief in out.items() for m, v in by_belief.items()}
        np.savez(f"results_decision/{name}.npz", index=index, horizon=H[index], seconds=time.time() - t0,
                 params=np.array([par.g, par.n, par.r, par.beta, par.load, par.xi]), phis=np.array(par.phis),
                 age_limit=np.array(np.nan if age_limit is None else float(age_limit)),
                 truth=np.array("M3" if truth == "TRUE" else truth), **flat)        # the keys "...|TRUE" refer to this model
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
