"""check_predict.py -- the exact prediction formula against brute-force simulation of couples.

Two held-out couples, two values of the frailty variance, 24 independent simulations of 250,000
couples each. For every survival probability at 12 years the script prints the exact value, the
simulated mean and the z-score of their difference (standard error from the 24 replications)."""
import numpy as np

from dyadic_mortality import CoupleData
from pipeline import load_couples, split_indices, fit_scaler, make_data, prior_arrays, age_max_of, predict_states

d = load_couples("cohort_w10.csv")
tr, te = split_indices(d)
sc = fit_scaler(d.iloc[tr])
ages, pstar, _ = prior_arrays("prior_table.csv", age_max_of(d))
test = make_data(d.iloc[te].reset_index(drop=True), sc)
H, R, REPS = 12, 250_000, 24
theta = np.log(-np.log1p(-pstar))
worst = 0.0
for kappa in (0.05, 0.40):
    draw = dict(p0=pstar[None], beta=np.array([[[0.02, 0.6, 0.3, 0.4, -0.1, -0.15], [0.1, 1.0, 0.2, 0.5, -0.1, -0.2]]]),
                gamma=np.array([[[0.1, 0.35], [0.0, 0.35]]]), omega=np.array([[0.30, 0.25]]), kappa=np.array([kappa]))
    for couple in (50, 7):
        pick = np.array([couple])
        sub = CoupleData(age0=test.age0[pick], x_own=test.x_own[pick], x_cross=test.x_cross[pick],
                         n_surv=test.n_surv[pick], died=test.died[pick], K=test.K)
        exact = predict_states(draw, sub, int(ages[0]), H)
        lin = (np.einsum("nsp,sp->ns", sub.x_own, draw["beta"][0]) + np.einsum("nsq,sq->ns", sub.x_cross, draw["gamma"][0]))
        est = {k: [] for k in exact}
        for r in range(REPS):
            rng = np.random.default_rng(1000 + r)
            Z = rng.gamma(1.0 / kappa, kappa, R)
            alive = np.ones((R, 2), dtype=bool)
            death_year = np.full((R, 2), 10**6)
            for t in range(H):
                idx = np.clip(sub.age0[0] + t - int(ages[0]), 0, pstar.shape[1] - 1)
                widowed = death_year[:, ::-1] < t                      # partner died in an earlier year
                eta = theta[np.arange(2), idx][None, :] + lin[0][None, :] + draw["omega"][0][None, :] * widowed
                die = alive & (rng.random((R, 2)) < 1.0 - np.exp(-Z[:, None] * np.exp(eta)))
                death_year[die] = t
                alive &= ~die
            est["husband"].append(alive[:, 0].mean())
            est["wife"].append(alive[:, 1].mean())
            est["joint_life"].append(alive.all(1).mean())
            est["last_survivor"].append(alive.any(1).mean())
        msg = []
        for k in exact:
            x = np.array(est[k])
            z = (x.mean() - exact[k][0, H - 1]) / (x.std(ddof=1) / np.sqrt(REPS))
            worst = max(worst, abs(x.mean() - exact[k][0, H - 1]))
            msg.append(f"{k} {exact[k][0, H - 1]:.4f} / {x.mean():.4f} (z {z:+.1f})")
        print(f"kappa {kappa:.2f}, couple {couple} (ages {sub.age0[0, 0]}/{sub.age0[0, 1]}), exact / simulated: " + " | ".join(msg))
print(f"largest |exact - simulated|: {worst:.4f}")
