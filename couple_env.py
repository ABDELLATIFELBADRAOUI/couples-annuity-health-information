"""
couple_env.py -- the couple's decision problem as a simulator, for reinforcement learning.

One episode is the life of one couple. The two years of death are drawn from the exact law of the
couple's history (decision_law), so the simulator and the dynamic-programming solution of
decision_dp describe the same problem. Many episodes advance together (NumPy arrays).

    reset(couples)  -> observation
    step(action)    -> observation, reward, done
        first call after reset : action[:, 0] = alpha (share of the liquid wealth annuitised),
                                 action[:, 1] = phi (survivor share); no time passes, reward 0
        then, once per year    : action = share of the cash consumed, in (0, 1]
    reward : utility of the year, 2 u(c / n) for a couple and u(c) for a survivor.
             The quantity to maximise is the sum of the rewards discounted at par.beta.

The horizon is the one of the laws given at construction: to match a dynamic-programming solution,
give the laws cut at the same number of years (as check_env.py does).

Observation (one row per episode): year, both alive, husband alone, wife alone, years since the
partner's death, cash-on-hand, income while both are alive, income of a survivor, then the features
of the couple given at construction (ages, covariates).
"""
import numpy as np

from decision_dp import Params, annuity_due, incomes


def sample_rows(weights, rng):
    """One index per row, with probability proportional to the weights of the row."""
    cum = np.cumsum(weights, axis=1)
    u = rng.random(weights.shape[0]) * cum[:, -1]
    return np.minimum((cum <= u[:, None]).sum(axis=1), weights.shape[1] - 1)


class CoupleEnv:
    def __init__(self, par: Params, both, only, pop_both, pop_only, features=None, seed=0):
        self.par, self.both, self.only = par, both, only
        self.T = both.shape[1] - 1
        self.aB = annuity_due(pop_both, pop_only, 0.0, par.r)
        self.aA = annuity_due(pop_both, pop_only, 1.0, par.r) - self.aB
        self.features = np.zeros((both.shape[0], 0)) if features is None else np.asarray(features, dtype=float)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ dates of death
    def sample_deaths(self, couples):
        """Year of death of each spouse (T: alive at the end of the horizon), from the exact law."""
        both, only, T, rng = self.both[couples], self.only, self.T, self.rng
        E = len(couples)
        first = sample_rows(np.concatenate([both[:, :-1] - both[:, 1:], both[:, -1:]], axis=1), rng)   # T: no death
        K = np.full((E, 2), T)
        has = first < T
        d = np.minimum(first, T - 1)
        p_h, p_w = only[couples, 0, d, d + 1], only[couples, 1, d, d + 1]
        total = both[np.arange(E), d] - both[np.arange(E), d + 1]
        u = rng.random(E) * total
        survivor = np.where(u < p_h, 0, np.where(u < p_h + p_w, 1, 2))                 # 2: both die in year d
        K[has & (survivor != 0), 0] = d[has & (survivor != 0)]
        K[has & (survivor != 1), 1] = d[has & (survivor != 1)]
        for s in (0, 1):
            m = has & (survivor == s)
            if m.any():
                S = only[couples[m], s, d[m], :]                                        # alive after t years, t > d
                w = np.concatenate([S[:, :-1] - S[:, 1:], S[:, -1:]], axis=1)
                w[np.arange(T + 1)[None, :] <= d[m][:, None]] = 0.0                     # the survivor outlives year d
                K[m, s] = sample_rows(w, rng)
        return K

    # ------------------------------------------------------------------ episode
    def reset(self, couples):
        self.couples = np.asarray(couples)
        self.K = self.sample_deaths(self.couples)
        E = len(self.couples)
        self.t = 0
        self.need_annuity = True
        self.cash = np.full(E, np.nan)
        self.yB, self.yS = np.zeros(E), np.zeros(E)
        return self._obs()

    def _status(self):
        alive = self.K >= self.t                                                        # alive at the start of year t
        return alive[:, 0] & alive[:, 1], alive[:, 0] & ~alive[:, 1], ~alive[:, 0] & alive[:, 1]

    def _obs(self):
        b, h, w = self._status()
        since = np.where(h, self.t - 1 - self.K[:, 1], np.where(w, self.t - 1 - self.K[:, 0], 0))
        cash = np.where(self.need_annuity, 1.0, self.cash)
        core = np.stack([np.full(len(cash), float(self.t)), b, h, w, since, cash, self.yB, self.yS], axis=1).astype(float)
        return np.concatenate([core, self.features[self.couples]], axis=1)

    def step(self, action):
        par, action = self.par, np.asarray(action, dtype=float)
        E = len(self.couples)
        if self.need_annuity:
            alpha, phi = np.clip(action[:, 0], 0.0, 1.0), np.clip(action[:, 1], 0.0, 1.0)
            yB, yS, x0 = incomes(par, self.aB[self.couples], self.aA[self.couples], alpha[:, None], phi[:, None])
            self.yB, self.yS, self.cash = yB[:, 0], yS[:, 0], x0[:, 0]
            self.need_annuity = False
            return self._obs(), np.zeros(E), np.zeros(E, dtype=bool)
        b, h, w = self._status()
        alive = b | h | w
        c = np.clip(action, 1e-9, 1.0) * self.cash
        if self.t == self.T - 1:
            c = self.cash                                                               # last year: nothing is carried forward
        u = lambda x: x ** (1.0 - par.g) / (1.0 - par.g)
        reward = np.where(b, 2.0 * u(np.where(b, c, 1.0) / par.n), np.where(h | w, u(np.where(h | w, c, 1.0)), 0.0))
        self.t += 1
        b2, h2, w2 = self._status()
        self.cash = np.where(alive, (1.0 + par.r) * (self.cash - c), 0.0) + np.where(b2, self.yB, np.where(h2 | w2, self.yS, 0.0))
        done = ~(b2 | h2 | w2) | (self.t >= self.T)
        return self._obs(), reward, done
