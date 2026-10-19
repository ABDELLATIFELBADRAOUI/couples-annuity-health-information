"""
xai_setup.py -- common ground of the explanation of the annuity decision.

A couple is described by 13 characteristics (RAW): the two ages, four health characteristics of each
spouse (BMI, smoking, hypertension, diabetes), the education of each spouse and the household wealth.
A "hybrid" couple takes some characteristics from a couple i and the others from a couple j.
Everything a policy needs is rebuilt from the 13 characteristics:
    - the inputs of the agents' networks (rl_core: ages, covariates, and the two population annuity
      values, which depend on the two ages only);
    - the laws of the histories under the full model and under the population basis, hence the exact
      optimum of dynamic programming for the hybrid couple.
So the same question can be put to a learned policy and to the exact one: what does the decision
become when these characteristics are those of another couple?

Shapley values. For a pair (i, j) and a set of players (the 13 characteristics, or 5 groups of them),
v(S) is the decision of the hybrid couple that has the players of S from i and the others from j.
The Shapley value of a player is its average marginal contribution over all orders of the players;
the values of all players add up to decision(i) - decision(j). They are computed here by enumerating
every S: no sampling. Averaged over many couples j, they are the usual SHAP values with an
independent ("interventional") background.
"""
import glob
from itertools import combinations
from math import factorial

import numpy as np

from dyadic_mortality import CoupleData, urn_posterior
from pipeline import load_draws, fit_scaler, OWN
from decision_setup import held_out, N_DRAWS
from decision_law import joint_law, population_draws
from decision_dp import Params, annuity_due, solve_couples
from rl_core import T_MAX, AGE_LIMIT, spend_rates

RAW = ["age_h", "age_w", "bmi_h", "bmi_w", "smoker_h", "smoker_w", "hyp_h", "hyp_w", "diab_h", "diab_w",
       "educ_h", "educ_w", "wealth"]
LABEL = {"age_h": "Âge du mari", "age_w": "Âge de la femme", "bmi_h": "IMC du mari", "bmi_w": "IMC de la femme",
         "smoker_h": "Mari fumeur", "smoker_w": "Femme fumeuse", "hyp_h": "Hypertension du mari",
         "hyp_w": "Hypertension de la femme", "diab_h": "Diabète du mari", "diab_w": "Diabète de la femme",
         "educ_h": "Années d'études du mari", "educ_w": "Années d'études de la femme", "wealth": "Patrimoine du ménage"}
OWN_H, OWN_W = [2, 4, 6, 8, 10, 12], [3, 5, 7, 9, 11, 12]          # positions of each spouse's covariates (pipeline.OWN order)
GROUPS = [("Âge du mari", [0]), ("Âge de la femme", [1]), ("Santé du mari", [2, 4, 6, 8]),
          ("Santé de la femme", [3, 5, 7, 9]), ("Études et patrimoine", [10, 11, 12])]
N_RAW, N_GROUPS = len(RAW), len(GROUPS)
BINARY = ["smoker_h", "smoker_w", "hyp_h", "hyp_w", "diab_h", "diab_w"]
# stated change of each characteristic for the "all else equal" effects, in natural units. Wealth enters the model
# as arcsinh(wealth / 1000): the change is +ln(10) on that scale, i.e. wealth multiplied by 10 when it is large and positive.
CHANGE = {"age_h": 5.0, "age_w": 5.0, "bmi_h": 5.0, "bmi_w": 5.0, "smoker_h": 1.0, "smoker_w": 1.0, "hyp_h": 1.0, "hyp_w": 1.0,
          "diab_h": 1.0, "diab_w": 1.0, "educ_h": 4.0, "educ_w": 4.0, "wealth": float(np.log(10.0))}
CHANGE_LABEL = {"age_h": "+5 ans", "age_w": "+5 ans", "bmi_h": "+5 kg/m²", "bmi_w": "+5 kg/m²", "smoker_h": "non → oui",
                "smoker_w": "non → oui", "hyp_h": "non → oui", "hyp_w": "non → oui", "diab_h": "non → oui", "diab_w": "non → oui",
                "educ_h": "+4 ans", "educ_w": "+4 ans", "wealth": "× 10"}


# ----------------------------------------------------------------------------- characteristics <-> model inputs
def characteristics(data: CoupleData):
    """(N, 13) array: ages in years, covariates on the scale of the model (standardised and centred
    with the training couples). The household wealth enters both spouses' covariates: one characteristic."""
    assert np.array_equal(data.x_own[:, 0, 5], data.x_own[:, 1, 5]), "wealth must be the same for both spouses"
    Z = np.empty((data.age0.shape[0], N_RAW))
    Z[:, :2] = data.age0
    Z[:, OWN_H], Z[:, OWN_W] = data.x_own[:, 0, :], data.x_own[:, 1, :]
    return Z


def couple_data(Z):
    """The model's inputs for couples given by their characteristics (partner covariates rebuilt)."""
    own = np.stack([Z[:, OWN_H], Z[:, OWN_W]], axis=1)
    none = np.zeros((Z.shape[0], 2), dtype=int)
    return CoupleData(age0=np.rint(Z[:, :2]).astype(int), x_own=own, x_cross=own[:, ::-1, :][:, :, [0, 1]],
                      n_surv=none, died=none.astype(bool), K=1)


def hybrid(Zi, Zj, mask):
    """Characteristics of i where mask is True, of j elsewhere."""
    return np.where(mask, Zi, Zj)


# ----------------------------------------------------------------------------- coalitions and Shapley weights
def coalitions(n):
    """(2^n, n) bool: every subset of n players; row k is the binary writing of k."""
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)


def shapley_matrix(n):
    """(n, 2^n) matrix C with shapley = C @ v, v being the value of every coalition (order of coalitions(n))."""
    S = coalitions(n)
    size = S.sum(axis=1)
    w = np.array([factorial(s) * factorial(n - 1 - s) / factorial(n) for s in range(n)])     # weight of a coalition of size s without the player
    C = np.empty((n, 2 ** n))
    for k in range(n):
        C[k] = np.where(S[:, k], w[np.clip(size - 1, 0, n - 1)], -w[np.clip(size, 0, n - 1)])
    return C


def group_masks():
    """(32, 13) bool: the 2^5 coalitions of the groups, as masks over the 13 characteristics."""
    S = coalitions(N_GROUPS)
    M = np.zeros((S.shape[0], N_RAW), dtype=bool)
    for g, (_, members) in enumerate(GROUPS):
        M[:, members] = S[:, [g]]
    return M


# ----------------------------------------------------------------------------- the couples and the models
class Setup:
    """Couples of the main split, population pricing basis and posterior draws of the full model."""

    def __init__(self, par: Params = Params()):
        self.par = par
        self.dtr, self.dte, self.data_tr, self.data_te, ages, pstar, c = held_out()
        self.age_min = int(ages[0])
        a, b = urn_posterior(self.data_tr, self.age_min, int(ages[-1]), pstar, c)
        self.pop = population_draws(a, b, len(OWN))
        self.Z = {"train": characteristics(self.data_tr), "test": characteristics(self.data_te)}
        self.natural = {"train": self.dtr[RAW].to_numpy(dtype=float), "test": self.dte[RAW].to_numpy(dtype=float)}
        self.scaler = fit_scaler(self.dtr)
        self._draws, self._annuity, self._rates = None, {}, {}

    @property
    def draws(self):
        if self._draws is None:
            self._draws = load_draws(sorted(glob.glob("results_eval/M3_train_*.npz")), n_draws=N_DRAWS)
        return self._draws

    def annuity_values(self, age0, chunk=256):
        """Population values of 1 per year while both are alive (aB) and of 1 per year to a survivor (aA):
        functions of the two ages only. Computed once per pair of ages."""
        age0 = np.rint(np.asarray(age0)).astype(int)
        todo = sorted({(int(h), int(w)) for h, w in age0} - set(self._annuity))
        for lo in range(0, len(todo), chunk):
            pairs = np.array(todo[lo:lo + chunk])
            Zp = np.zeros((len(pairs), N_RAW))
            Zp[:, :2] = pairs
            both, only = joint_law(self.pop, couple_data(Zp), self.age_min, T_MAX, age_limit=AGE_LIMIT)
            aB = annuity_due(both, only, 0.0, self.par.r)
            aA = annuity_due(both, only, 1.0, self.par.r) - aB
            rB, rS = spend_rates(both, only, self.par)
            self._annuity.update({tuple(p): (x, y) for p, x, y in zip(map(tuple, pairs.tolist()), aB, aA)})
            self._rates.update({tuple(p): (x, y) for p, x, y in zip(map(tuple, pairs.tolist()), rB, rS)})
        vals = np.array([self._annuity[(int(h), int(w))] for h, w in age0])
        return vals[:, 0], vals[:, 1]

    def rule_rates(self, age0):
        """Rates of the simple consumption rule corrected by the agents (rl_core.spend_rates): they depend
        on the two ages only. Returns rateB (N, T) and rateS (N, 2, T)."""
        age0 = np.rint(np.asarray(age0)).astype(int)
        self.annuity_values(age0)
        return (np.stack([self._rates[(int(h), int(w))][0] for h, w in age0]),
                np.stack([self._rates[(int(h), int(w))][1] for h, w in age0]))

    def features(self, Z):
        """Inputs of the agents' networks (same construction as rl_core.build)."""
        aB, aA = self.annuity_values(Z[:, :2])
        return np.concatenate([(Z[:, :2] - 65.0) / 10.0, Z[:, OWN_H], Z[:, OWN_W],
                               np.log(aB / 10.0)[:, None], np.log(aA / 5.0)[:, None]], axis=1)

    def laws(self, Z):
        """Laws of the histories of the couples Z: full model (posterior predictive) and population basis."""
        data = couple_data(Z)
        return {"TRUE": joint_law(self.draws, data, self.age_min, T_MAX, age_limit=AGE_LIMIT),
                "POP": joint_law(self.pop, data, self.age_min, T_MAX, age_limit=AGE_LIMIT)}

    def changed_couples(self, Z):
        """For each couple and each characteristic, the couple in which that characteristic alone is
        changed by the stated amount (CHANGE). Binary characteristic: the other state. Others: the
        value is raised, or lowered when raising it would leave the range observed in the data.
        Returns Zc (N, 13, 13), Zc[n, k] being couple n with characteristic k changed, and sign (N, 13):
        +1 when the changed couple has the higher value, -1 when it has the lower one, so that
        sign * (decision(changed) - decision(couple)) is always the effect of the stated increase."""
        allZ = np.concatenate([self.Z["train"], self.Z["test"]], axis=0)
        low, high = allZ.min(axis=0), allZ.max(axis=0)
        Zc, sign = np.repeat(Z[:, None, :], N_RAW, axis=1), np.ones((Z.shape[0], N_RAW))
        for k, name in enumerate(RAW):
            if name in BINARY:
                assert np.isclose(high[k] - low[k], 1.0) and np.isclose(allZ[:, k], low[k]).sum() + np.isclose(allZ[:, k], high[k]).sum() == len(allZ)
                step, up = 1.0, np.isclose(Z[:, k], low[k])                    # "no" becomes "yes", "yes" becomes "no"
            else:
                step = CHANGE[name] / (1.0 if name.startswith("age") else self.scaler[name][1])    # ages in years, the others standardised
                up = Z[:, k] + step <= high[k] + 1e-9
            Zc[:, k, k] = Z[:, k] + np.where(up, step, -step)
            sign[:, k] = np.where(up, 1.0, -1.0)
        return Zc, sign

    @staticmethod
    def horizons(Z):
        return (AGE_LIMIT - Z[:, :2].min(axis=1)).astype(int)

    def optimum(self, Z, laws=None, log=None):
        """Exact optimum of the couples Z under the full model: alpha, phi, value."""
        laws = self.laws(Z) if laws is None else laws
        out = solve_couples(self.par, laws, ["TRUE"], self.horizons(Z), log=log)
        return out["alpha"]["TRUE"], out["phi"]["TRUE"], out["true"]["TRUE"]
