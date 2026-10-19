"""check_law.py -- the exact law of the couple's history (decision_law.joint_law) against
(1) the prediction formula already validated (pipeline.predict_states) and
(2) brute-force simulation of couples with frailty and widowhood effect, history by history:
    probability that spouse s is alone after t years with the partner dead since year d."""
import numpy as np

from dyadic_mortality import CoupleData
from pipeline import load_couples, split_indices, fit_scaler, make_data, prior_arrays, age_max_of, predict_states
from decision_law import joint_law

d = load_couples("cohort_w10.csv")
tr, te = split_indices(d)
sc = fit_scaler(d.iloc[tr])
ages, pstar, _ = prior_arrays("prior_table.csv", age_max_of(d))
test = make_data(d.iloc[te].reset_index(drop=True), sc)
age_min = int(ages[0])
theta = np.log(-np.log1p(-pstar))


def subset(pick):
    return CoupleData(age0=test.age0[pick], x_own=test.x_own[pick], x_cross=test.x_cross[pick],
                      n_surv=test.n_surv[pick], died=test.died[pick], K=test.K)


# ---- 1. marginals against predict_states (three draws with different frailty variances, 200 couples, 40 years)
T = 40
rng = np.random.default_rng(5)
draws = dict(p0=np.stack([pstar * f for f in (1.0, 0.9, 1.1)]),
             beta=rng.normal(0, 0.3, (3, 2, 6)), gamma=rng.normal(0, 0.2, (3, 2, 2)),
             omega=rng.normal(0.2, 0.1, (3, 2)), kappa=np.array([0.0, 0.05, 0.40]))
sub = subset(np.arange(200))
both, only = joint_law(draws, sub, age_min, T)
ref = predict_states(draws, sub, age_min, T, chunk=1)
alone = only.sum(axis=2)                                           # (N, 2, T+1)
gap = max(np.abs(both[:, 1:] - ref["joint_life"]).max(),
          np.abs(both[:, 1:] + alone[:, 0, 1:] - ref["husband"]).max(),
          np.abs(both[:, 1:] + alone[:, 1, 1:] - ref["wife"]).max(),
          np.abs(both[:, 1:] + alone[:, 0, 1:] + alone[:, 1, 1:] - ref["last_survivor"]).max())
print(f"1. largest gap with predict_states over 200 couples x 40 years x 4 states: {gap:.2e}")
print(f"   probabilities stay in [0, 1]: {bool((both >= 0).all() and (only >= -1e-15).all() and ((both + alone.sum(1)) <= 1 + 1e-12).all())}")

# ---- 2. history by history, against simulation
H, R, REPS = 14, 250_000, 16
worst_z, n_cells, n_big = 0.0, 0, 0
for kappa in (0.05, 0.40):
    draw = dict(p0=pstar[None], beta=np.array([[[0.02, 0.6, 0.3, 0.4, -0.1, -0.15], [0.1, 1.0, 0.2, 0.5, -0.1, -0.2]]]),
                gamma=np.array([[[0.1, 0.35], [0.0, 0.35]]]), omega=np.array([[0.30, 0.25]]), kappa=np.array([kappa]))
    for couple in (50, 7):
        s1 = subset(np.array([couple]))
        b, o = joint_law(draw, s1, age_min, H)
        lin = (np.einsum("nsp,sp->ns", s1.x_own, draw["beta"][0]) + np.einsum("nsq,sq->ns", s1.x_cross, draw["gamma"][0]))
        count = np.zeros((2, H, H + 1))
        count_both = np.zeros(H + 1)
        for r in range(REPS):
            g = np.random.default_rng(3000 + r)
            Z = g.gamma(1.0 / kappa, kappa, R)
            alive = np.ones((R, 2), dtype=bool)
            death_year = np.full((R, 2), 10**6)
            count_both[0] += R
            for t in range(H):
                idx = np.clip(s1.age0[0] + t - age_min, 0, pstar.shape[1] - 1)
                widowed = death_year[:, ::-1] < t
                eta = theta[np.arange(2), idx][None, :] + lin[0][None, :] + draw["omega"][0][None, :] * widowed
                die = alive & (g.random((R, 2)) < 1.0 - np.exp(-Z[:, None] * np.exp(eta)))
                death_year[die] = t
                alive &= ~die
                count_both[t + 1] += alive.all(1).sum()
                for s in range(2):
                    m = alive[:, s] & ~alive[:, 1 - s]
                    count[s, :, t + 1] += np.bincount(death_year[m, 1 - s], minlength=H)[:H]
        n = R * REPS
        est, est_b = count / n, count_both / n
        se = np.sqrt(np.maximum(o[0] * (1 - o[0]), 1e-300) / n)
        valid = o[0] * n > 50                                      # cells with enough expected histories
        z = np.abs(est - o[0])[valid] / se[valid]
        zb = np.abs(est_b - b[0])[1:] / np.sqrt(b[0][1:] * (1 - b[0][1:]) / n)
        worst_z, n_cells, n_big = max(worst_z, z.max(), zb.max()), n_cells + z.size + zb.size, n_big + int((z > 3).sum() + (zb > 3).sum())
        print(f"2. kappa {kappa:.2f}, couple {couple} (ages {s1.age0[0, 0]}/{s1.age0[0, 1]}): {z.size} histories, "
              f"largest |z| {z.max():.2f}, largest gap {np.abs(est - o[0]).max():.2e}; both alive: largest |z| {zb.max():.2f}")
print(f"   {n_cells} probabilities compared with {R * REPS:,} simulated couples each; largest |z| {worst_z:.2f}; |z| > 3: {n_big}")
