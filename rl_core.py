"""
rl_core.py -- common ground of the learning agents.

    simulator   : lives of couples drawn from the full mortality model (same posterior draws as the
                  exact laws, shared frailty, widowhood effect, limiting age of 110)
    observation : what an agent sees (features of the couple, year, status, cash, incomes)
    networks    : one network for the annuity decision, one for the yearly consumption
    evaluation  : exact value of a learned policy, obtained by following it along every history of a
                  couple (decision_dp.value) -- no simulation noise in the reported gap to the optimum

Decision problem: the one of decision_dp (base case). At time 0 the agent chooses a survivor share
among MENU and the share alpha of the wealth converted into the annuity; then, every year, the share
of its cash that it consumes.

Rewards are the yearly utilities multiplied by a positive constant specific to each couple
(scale = (g - 1) c_ref^(g - 1), with c_ref the income of a couple that annuitises everything with a
2/3 survivor share). A positive constant per couple does not change the best policy of that couple;
it only puts couples of different ages on a comparable scale for learning.

Simple consumption rule corrected by the consumption network (environment variable CARUP_RULE):
    "af"     (default) consume the income of the year plus the wealth carried from the previous year
             divided by an annuity factor: the value, at the household's current ages, of 1 per year
             for life under the population table (age and sex only) and the interest rate of the
             problem. For a couple the factor is that of a joint annuity whose survivor receives the
             share (n / 2)^(1 / g) / n, the ratio of consumptions that equalises marginal utility
             between a couple and a survivor (0.606 in the base case). The table is public: it is the
             pricing basis of the annuity. Before any learning, with the optimal annuity choice, this
             rule loses 1.0 % of the resources on the monitoring couples, against 2.6 % for "flat10".
    "flat10" consume the income plus 10 % of the carried wealth (the agents of step 4).
In both rules the share of the cash consumed is capped at sigmoid(LOGIT_CAP) = 98.2 %: without the cap the
logit of the rule is infinite when the household carries no wealth, and the correction of the network could
not be learned (its gradient would vanish). A household without liquid wealth therefore consumes 98.2 % of
its income under the rule alone.
"""
import glob
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from dyadic_mortality import urn_posterior
from pipeline import load_draws, OWN
from decision_setup import held_out, law, N_DRAWS
from decision_law import joint_law, population_draws
from decision_dp import Params, annuity_due, value, PHI_PRE

jax.config.update("jax_enable_x64", True)

MENU = np.array(Params().phis)            # survivor shares on offer
T_MAX, AGE_LIMIT = 60, 110
RULE = os.environ.get("CARUP_RULE", "af")
assert RULE in ("af", "flat10")
N_STATIC, N_DYN = 16, (8 if RULE == "af" else 7)
M0, LOGIT_CAP = 0.10, 4.0                 # flat rate of the rule "flat10"; cap of the logit of the rule (see base_logit)
STATUS_BOTH, STATUS_H, STATUS_W = 0, 1, 2


# ----------------------------------------------------------------------------- data
def build(par: Params = Params()):
    """Arrays for the training couples (simulated lives) and the held-out couples (exact evaluation)."""
    dtr, dte, data_tr, data_te, ages, pstar, c = held_out()
    age_min = int(ages[0])
    draws = load_draws(sorted(glob.glob("results_eval/M3_train_*.npz")), n_draws=N_DRAWS)   # the truth
    a, b = urn_posterior(data_tr, age_min, int(ages[-1]), pstar, c)
    pop = population_draws(a, b, len(OWN))                                                   # pricing basis
    out = {}
    for name, data in (("train", data_tr), ("test", data_te)):
        both, only = joint_law(pop, data, age_min, T_MAX, age_limit=AGE_LIMIT)
        aB = annuity_due(both, only, 0.0, par.r)
        aA = annuity_due(both, only, 1.0, par.r) - aB
        c_ref = (1.0 - par.load) / (aB + (2.0 / 3.0) * aA)
        feat = np.concatenate([(data.age0 - 65.0) / 10.0, data.x_own[:, 0, :], data.x_own[:, 1, :],
                               np.log(aB / 10.0)[:, None], np.log(aA / 5.0)[:, None]], axis=1)
        lin = (np.einsum("nsp,dsp->dns", data.x_own, draws["beta"])
               + np.einsum("nsq,dsq->dns", data.x_cross, draws["gamma"]))                    # (D, N, 2)
        rateB, rateS = spend_rates(both, only, par)
        out[name] = dict(age0=jnp.asarray(data.age0), feat=jnp.asarray(feat), aB=jnp.asarray(aB), aA=jnp.asarray(aA),
                         scale=jnp.asarray((par.g - 1.0) * c_ref ** (par.g - 1.0)), lin=jnp.asarray(lin),
                         rateB=jnp.asarray(rateB), rateS=jnp.asarray(rateS))
    out["sim"] = dict(theta=jnp.asarray(np.log(-np.log1p(-draws["p0"]))), omega=jnp.asarray(draws["omega"]),
                      kappa=jnp.asarray(draws["kappa"]), age_min=age_min)
    out["par"] = par
    return out


def monitoring_set(S, n=300, seed=11):
    """Training couples followed during learning: n couples drawn at random, their exact laws under the
    full model and the value of their optimum (results_xai/optimum_train.npz, from `xai_dp.py train`).
    The held-out couples are never looked at during learning."""
    dtr, dte, data_tr, data_te, ages, pstar, c = held_out()
    idx = np.sort(np.random.default_rng(seed).choice(data_tr.age0.shape[0], n, replace=False))
    sub = type(data_tr)(age0=data_tr.age0[idx], x_own=data_tr.x_own[idx], x_cross=data_tr.x_cross[idx],
                        n_surv=data_tr.n_surv[idx], died=data_tr.died[idx], K=data_tr.K)
    draws = load_draws(sorted(glob.glob("results_eval/M3_train_*.npz")), n_draws=N_DRAWS)
    both, only = joint_law(draws, sub, int(ages[0]), T_MAX, age_limit=AGE_LIMIT)
    group = {k: v[idx] for k, v in S["train"].items() if k != "lin"}
    opt = np.load("results_xai/optimum_train.npz")
    return idx, group, both, only, {k: opt[k][idx] for k in ("alpha", "phi", "value")}


def spend_rates(both, only, par: Params):
    """Share of the carried wealth consumed by the simple rule, from the population law of the couples.
    rateB[n, t]   : both spouses alive at the start of year t
    rateS[n, s, t]: spouse s alone at the start of year t
    Rule "af": 1 / annuity factor at the current ages (see the header), with independent spouses as in
    the population table. Rule "flat10": 0.10 everywhere."""
    N, T = both.shape[0], both.shape[1] - 1
    if RULE == "flat10":
        return np.full((N, T), M0), np.full((N, 2, T), M0)
    share = (par.n / 2.0) ** (1.0 / par.g) / par.n                          # consumption of a survivor relative to the couple's
    surv = both[:, None, :] + only.sum(axis=2)                              # (N, 2, T + 1): spouse s alive after t years
    v = (1.0 + par.r) ** -np.arange(T + 1)
    rateB, rateS = np.ones((N, T)), np.ones((N, 2, T))
    for t in range(T):
        alive = surv[:, :, t] > 0
        cond = np.where(alive[:, :, None], surv[:, :, t:] / np.where(alive, surv[:, :, t], 1.0)[:, :, None], 0.0)   # alive after u years given alive after t
        disc = v[:T + 1 - t]
        rateS[:, :, t] = 1.0 / np.maximum((cond * disc).sum(axis=2), 1.0)
        two = cond[:, 0] * cond[:, 1]
        rateB[:, t] = 1.0 / np.maximum(((two + share * (cond[:, 0] + cond[:, 1] - 2.0 * two)) * disc).sum(axis=1), 1.0)
    return rateB, rateS


# ----------------------------------------------------------------------------- simulated lives
def sample_life(key, n, sim, group):
    """Years of death (husband, wife) of one simulated life of couple n; T_MAX if alive at the end
    (impossible with the limiting age when the couple is 50 or older)."""
    k1, k2, k3 = jax.random.split(key, 3)
    d = jax.random.randint(k1, (), 0, sim["kappa"].shape[0])
    kap = sim["kappa"][d]
    z = jnp.where(kap > 1e-10, jax.random.gamma(k2, 1.0 / jnp.maximum(kap, 1e-10)) * kap, 1.0)
    u = jax.random.uniform(k3, (T_MAX, 2))
    age0, lin = group["age0"][n], group["lin"][d, n]
    n_ages = sim["theta"].shape[2]

    def step(carry, inp):
        alive, K = carry
        t, u_t = inp
        age = age0 + t
        widowed = K[::-1] < t                                          # partner died in an earlier year
        eta = sim["theta"][d, jnp.arange(2), jnp.clip(age - sim["age_min"], 0, n_ages - 1)] + lin + sim["omega"][d] * widowed
        h = jnp.where(age >= AGE_LIMIT - 1, 1.0, 1.0 - jnp.exp(-z * jnp.exp(eta)))
        die = alive & (u_t < h)
        return (alive & ~die, jnp.where(die, t, K)), None

    (_, K), _ = lax.scan(step, (jnp.ones(2, dtype=bool), jnp.full(2, T_MAX)), (jnp.arange(T_MAX), u))
    return K


def status_at(K, t):
    """0 both alive, 1 husband alone, 2 wife alone, 3 nobody, at the start of year t."""
    h, w = K[0] >= t, K[1] >= t
    return jnp.where(h & w, 0, jnp.where(h, 1, jnp.where(w, 2, 3)))


# ----------------------------------------------------------------------------- economics
def incomes_one(par: Params, aB, aA, alpha, phi):
    """Income while both are alive, income of a survivor, cash at time 0 (one couple, base problem)."""
    y = alpha * (1.0 - par.xi) * (1.0 - par.load) / (aB + phi * aA)
    y_pre = par.xi / (aB + PHI_PRE * aA)
    return y + y_pre, phi * y + PHI_PRE * y_pre, (1.0 - par.xi) * (1.0 - alpha) + y + y_pre


def utility(par: Params, c, status):
    """Utility of a year: 2 u(c / n) for a couple, u(c) for a survivor."""
    u = lambda x: x ** (1.0 - par.g) / (1.0 - par.g)
    return jnp.where(status == STATUS_BOTH, 2.0 * u(c / par.n), u(c))


# ----------------------------------------------------------------------------- observations and networks
def dyn_features(t, status, since, cash, yB, yS, phi, rate):
    """Features of the year. The problem is homogeneous in (cash, incomes), so the wealth carried from
    the previous year enters relative to the income of the year, and the two incomes through their ratio.
    With the rule "af", the rate of the rule (1 / annuity factor of the population table) is a feature."""
    income = jnp.where(status == STATUS_BOTH, yB, yS)
    wealth = jnp.maximum(cash - income, 0.0)
    f = [t / 30.0, 1.0 * (status == STATUS_BOTH), 1.0 * (status == STATUS_H), 1.0 * (status == STATUS_W),
         since / 10.0, jnp.log1p(wealth / income), yS / yB]
    return jnp.stack(f + [jnp.log(rate) / 3.0] if RULE == "af" else f)


def rate_at(group_rateB, group_rateS, t, status):
    """Rate of the simple rule in year t for the household's status (both alive, husband alone, wife alone)."""
    tt = jnp.clip(t, 0, T_MAX - 1)
    return jnp.where(status == STATUS_BOTH, group_rateB[tt], group_rateS[jnp.clip(status - 1, 0, 1), tt])


def init_mlp(key, sizes, last_scale=0.01, last_bias=0.0):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        w = jax.random.normal(keys[i], (m, n)) * (last_scale if last else np.sqrt(1.0 / m))
        params.append((w, jnp.full(n, last_bias) if last else jnp.zeros(n)))
    return params


def mlp(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def init_policy(key, hidden=(64, 64)):
    """ann : features of the couple -> 4 alpha logits (one per survivor share) and 4 logits of the survivor share
       cons: features of the couple and of the year -> correction to the logit of the share of cash consumed"""
    k1, k2 = jax.random.split(key)
    return dict(ann=init_mlp(k1, (N_STATIC,) + hidden + (2 * len(MENU),), last_bias=0.0),
                cons=init_mlp(k2, (N_STATIC + N_DYN,) + hidden + (1,), last_bias=0.0))


def annuity_choice(params, feat):
    """Deterministic choice: survivor share with the largest logit, and its alpha."""
    out = mlp(params["ann"], feat)
    k = jnp.argmax(out[len(MENU):])
    return jax.nn.sigmoid(out[k]), k


def base_logit(cash, income, rate):
    """Logit of the share of cash consumed by the simple rule: the income of the year plus the share
    `rate` of the wealth carried from the previous year. The consumption network outputs a correction
    to this rule."""
    w = jnp.maximum(cash - income, 0.0)
    return jnp.minimum(jnp.log(income + rate * w + 1e-12) - jnp.log((1.0 - rate) * w + 1e-12), LOGIT_CAP)


def share_logit(params, feat, dyn, cash, income, rate):
    return base_logit(cash, income, rate) + mlp(params["cons"], jnp.concatenate([feat, dyn]))[0]


# ----------------------------------------------------------------------------- one simulated life under a policy
def life_return(params, par: Params, group, n, K, alpha, phi):
    """Discounted sum of scaled utilities of one simulated life (years of death K) when the couple
    chooses (alpha, phi) and then follows the consumption network. Differentiable in the parameters
    and in alpha."""
    yB, yS, x0 = incomes_one(par, group["aB"][n], group["aA"][n], alpha, phi)
    feat, R = group["feat"][n], 1.0 + par.r

    def step(cash, t):
        status = status_at(K, t)
        alive = status < 3
        since = jnp.where(status == STATUS_H, t - 1 - K[1], jnp.where(status == STATUS_W, t - 1 - K[0], 0))
        rate = rate_at(group["rateB"][n], group["rateS"][n], t, status)
        dyn = dyn_features(t, status, since, cash, yB, yS, phi, rate)
        c = jax.nn.sigmoid(share_logit(params, feat, dyn, cash, jnp.where(status == STATUS_BOTH, yB, yS), rate)) * cash
        r = jnp.where(alive, utility(par, jnp.where(alive, c, 1.0), status), 0.0)
        nxt = status_at(K, t + 1)
        cash_next = R * (cash - c) + jnp.where(nxt == STATUS_BOTH, yB, yS)
        return jnp.where(alive, cash_next, cash), par.beta ** t * r

    _, rewards = lax.scan(step, x0, jnp.arange(T_MAX))
    return group["scale"][n] * jnp.sum(rewards)


# ----------------------------------------------------------------------------- exact evaluation on the tree of histories
def tree_value(params, par: Params, feat, aB, aA, rateB, rateS, both, only, alpha, phi):
    """Exact expected utility (true scale) of one couple that chooses (alpha, phi) and follows the
    consumption network, under the law (both, only). rateB (T,), rateS (2, T): rates of the simple rule."""
    yB, yS, x0 = incomes_one(par, aB, aA, alpha, phi)
    R, T = 1.0 + par.r, T_MAX

    def step_b(x, t):
        dyn = dyn_features(t, STATUS_BOTH, 0, x, yB, yS, phi, rateB[t])
        c = jax.nn.sigmoid(share_logit(params, feat, dyn, x, yB, rateB[t])) * x
        w = R * (x - c)
        return w + yB, (c, w)

    _, (cb, w_after) = lax.scan(step_b, x0, jnp.arange(T))
    d_idx = jnp.arange(T)

    def step_s(x, t):                                                  # x (2, T): cash by survivor and year of the partner's death
        def one(s, d, xq):
            dyn = dyn_features(t, STATUS_H + s, t - 1 - d, xq, yB, yS, phi, rateS[s, t])
            return jax.nn.sigmoid(share_logit(params, feat, dyn, xq, yS, rateS[s, t])) * xq
        c = jax.vmap(lambda s, row: jax.vmap(lambda d, xq: one(s, d, xq))(d_idx, row))(jnp.arange(2), x)
        active = (d_idx < t)[None, :]
        return jnp.where(active, R * (x - c) + yS, x), jnp.where(active, c, 1.0)

    _, cs = lax.scan(step_s, jnp.broadcast_to(w_after + yS, (2, T)), jnp.arange(T))
    return value(cb, cs, both, only, par)


def make_exact_evaluator(par: Params, chunk=64):
    """Returns evaluate(params, test, both, only) -> (alpha, index of the survivor share, exact value)
    for every held-out couple; the consumption network is followed along every history."""
    @jax.jit
    def run(params, feat, aB, aA, rB, rS, b, o):
        def one(f, ab, aa, rb, rs, bb, oo):
            alpha, k = annuity_choice(params, f)
            return alpha, k, tree_value(params, par, f, ab, aa, rb, rs, bb, oo, alpha, jnp.asarray(MENU)[k])
        return jax.vmap(one)(feat, aB, aA, rB, rS, b, o)

    def evaluate(params, test, both, only):
        N = test["feat"].shape[0]
        alpha, k, val = [], [], []
        for lo in range(0, N, chunk):
            idx = np.minimum(np.arange(lo, lo + chunk), N - 1)         # pad the last chunk
            a, kk, v = run(params, test["feat"][idx], test["aB"][idx], test["aA"][idx], test["rateB"][idx], test["rateS"][idx],
                           jnp.asarray(both[idx]), jnp.asarray(only[idx]))
            m = min(chunk, N - lo)
            alpha.append(np.asarray(a)[:m]), k.append(np.asarray(kk)[:m]), val.append(np.asarray(v)[:m])
        return np.concatenate(alpha), np.concatenate(k), np.concatenate(val)

    return evaluate


# ----------------------------------------------------------------------------- optimiser and storage
def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return dict(m=zeros, v=zeros, t=jnp.zeros((), dtype=jnp.int32))


def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8, max_norm=None):
    if max_norm is not None:                                           # global-norm clipping
        norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        grads = jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, max_norm / (norm + 1e-12)), grads)
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g ** 2, state["v"], grads)
    step = lr * jnp.sqrt(1 - b2 ** t) / (1 - b1 ** t)
    params = jax.tree_util.tree_map(lambda p, m_, v_: p - step * m_ / (jnp.sqrt(v_) + eps), params, m, v)
    return params, dict(m=m, v=v, t=t)


def save_params(path, params, **extra):
    leaves, _ = jax.tree_util.tree_flatten(params)
    np.savez(path, **{f"leaf_{i:03d}": np.asarray(x) for i, x in enumerate(leaves)}, rule=np.array(RULE), **extra)


def load_params(path, template):
    z = np.load(path)
    rule = str(z["rule"]) if "rule" in z.files else "flat10"            # the files of step 4 have no rule: flat 10 %
    if rule != RULE:
        raise RuntimeError(f"{path} was trained with the rule '{rule}'; set CARUP_RULE={rule} to use it")
    _, treedef = jax.tree_util.tree_flatten(template)
    n = len(jax.tree_util.tree_leaves(template))
    return jax.tree_util.tree_unflatten(treedef, [jnp.asarray(z[f"leaf_{i:03d}"]) for i in range(n)])
