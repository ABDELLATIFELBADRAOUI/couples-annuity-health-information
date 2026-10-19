"""
deep_models.py -- DeepSurv and DeepHit on the same couples and the same split as the other models.

Implementation : pycox (models CoxPH = DeepSurv, DeepHitSingle), networks of torchtuples.
One model per sex. Features: age and the six own covariates; the "+partner" variants add the
partner's BMI and smoking, exactly as the Cox models of evaluate.py. Couple targets are obtained
under independence of the two spouses, as for the Cox models.

Protocol (the held-out couples are never used here):
    - 20 % of the training couples are set aside for validation (early stopping and selection);
    - a small grid (width 32 or 64, dropout 0.1 or 0.4; two hidden layers, batch normalisation,
      Adam with learning rate 0.01, batches of 256) is compared on the validation couples;
    - the selected configuration is trained with five random seeds and the five predicted survival
      curves are averaged.
DeepSurv is selected on its validation loss (partial likelihood). DeepHit works in discrete time with
one-year intervals; its loss is alpha * likelihood + (1 - alpha) * ranking, with a ranking scale sigma,
so its validation loss cannot be compared across values of alpha. DeepHit is therefore selected on the
integrated Brier score of the validation couples (years 1-10, censoring weights from the other training
couples): first the architecture at the values of the pycox example (alpha 0.2, sigma 0.1), then
(alpha, sigma) among TUNING with that architecture.

Output: <results dir>/deep_predictions.npz (survival after 1..12 years of the held-out couples) and
deep_models.json (selected configurations, validation losses). evaluate.py reads the predictions.
"""
import json
import os
import time

import numpy as np
import torch
import torchtuples as tt
from pycox.models import CoxPH, DeepHitSingle

from metrics_util import TIMES, truncate, censoring_survival, ibs_ipcw
from pipeline import load_couples, split_indices, fit_scaler, make_data, outcomes, SPLIT_SEED

RES = os.environ.get("CARUP_RESULTS_DIR", "results_eval")
HORIZON, SEEDS, VAL_FRACTION = 12, (0, 1, 2, 3, 4), 0.20
GRID = [dict(nodes=n, dropout=p) for n in (32, 64) for p in (0.1, 0.4)]
TUNING = [(0.2, 0.5), (0.5, 0.1), (0.5, 0.5), (0.8, 0.1), (0.8, 0.5), (1.0, 0.1)]        # (alpha, sigma) tried after (0.2, 0.1); sigma is unused at alpha = 1
LR, BATCH, EPOCHS, PATIENCE = 0.01, 256, 500, 10
torch.set_num_threads(1)

d = load_couples("cohort_w10.csv")
tr, te = split_indices(d)
dtr, dte = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)
sc = fit_scaler(dtr)
data_tr, data_te = make_data(dtr, sc), make_data(dte, sc)
o_tr = outcomes(dtr)

rng = np.random.default_rng(2026)                       # validation couples, the same for every model
is_val = np.zeros(len(dtr), dtype=bool)
is_val[rng.permutation(len(dtr))[: int(round(VAL_FRACTION * len(dtr)))]] = True
age_mean, age_sd = data_tr.age0[~is_val].mean(axis=0), data_tr.age0[~is_val].std(axis=0)


def design(data, s, with_partner):
    cols = [((data.age0[:, s] - age_mean[s]) / age_sd[s])[:, None], data.x_own[:, s, :]]
    if with_partner:
        cols.append(data.x_cross[:, s, :])
    return np.concatenate(cols, axis=1).astype("float32")


def curve_at_years(surv_df):
    """Survival after 1..HORIZON years from a pycox survival data frame (rows: times)."""
    times = surv_df.index.values.astype(float)
    k = np.searchsorted(times, np.arange(1, HORIZON + 1), side="right") - 1
    return np.where(k[:, None] >= 0, surv_df.values[np.maximum(k, 0)], 1.0).T          # (n, HORIZON)


def train(kind, x_fit, y_fit, x_val, y_val, cfg, seed, cuts=None, alpha=0.2, sigma=0.1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_features = 1 if kind == "DeepSurv" else len(cuts)
    net = tt.practical.MLPVanilla(x_fit.shape[1], [cfg["nodes"], cfg["nodes"]], out_features, batch_norm=True,
                                  dropout=cfg["dropout"], output_bias=(kind != "DeepSurv"))
    if kind == "DeepSurv":
        model = CoxPH(net, tt.optim.Adam)
    else:
        model = DeepHitSingle(net, tt.optim.Adam, alpha=alpha, sigma=sigma, duration_index=cuts)
    model.optimizer.set_lr(LR)
    log = model.fit(x_fit, y_fit, BATCH, EPOCHS, [tt.callbacks.EarlyStopping(patience=PATIENCE)], verbose=False,
                    val_data=(x_val, y_val), val_batch_size=4096)
    val_loss = float(log.to_pandas()["val_loss"].min())
    if kind == "DeepSurv":
        model.compute_baseline_hazards()
    return model, val_loss, int(len(log.to_pandas()))


t0 = time.time()
pred, info = {}, {}
for kind in ("DeepSurv", "DeepHit"):
    for with_partner in (False, True):
        name = kind + ("+partner" if with_partner else "")
        curves = []
        for s, sex in enumerate(["husband", "wife"]):
            t, e = o_tr[sex]
            x_all, x_test = design(data_tr, s, with_partner), design(data_te, s, with_partner)
            cuts = None
            if kind == "DeepSurv":
                y_all = (t.astype("float32"), e.astype("float32"))
            else:
                cuts = np.arange(0, int(np.ceil(t.max())) + 1, dtype="float32")
                idx, ev = DeepHitSingle.label_transform(cuts).fit_transform(t.astype("float32"), e.astype("float32"))
                y_all = (idx, ev)
            fit = lambda y: (y[0][~is_val], y[1][~is_val])
            val = lambda y: (y[0][is_val], y[1][is_val])
            tt_, ee_ = truncate((t, e))                                       # same administrative censoring as in the evaluation
            G = censoring_survival(tt_[~is_val], ee_[~is_val])
            val_ibs = lambda model: ibs_ipcw(G, tt_[is_val], ee_[is_val], curve_at_years(model.predict_surv_df(x_all[is_val]))[:, TIMES - 1],
                                             TIMES.astype(float))
            scores, tuned = [], dict(alpha=0.2, sigma=0.1)
            for cfg in GRID:
                model, loss, _ = train(kind, x_all[~is_val], fit(y_all), x_all[is_val], val(y_all), cfg, 0, cuts)
                scores.append(loss if kind == "DeepSurv" else val_ibs(model))
            best = GRID[int(np.argmin(scores))]
            extra = {}
            if kind == "DeepHit":
                tuning = {(0.2, 0.1): min(scores)}
                for a_, s_ in TUNING:
                    model, _, _ = train(kind, x_all[~is_val], fit(y_all), x_all[is_val], val(y_all), best, 0, cuts, a_, s_)
                    tuning[(a_, s_)] = val_ibs(model)
                a_, s_ = min(tuning, key=tuning.get)
                tuned = dict(alpha=a_, sigma=s_)
                extra = dict(selected_alpha_sigma=tuned, validation_ibs_by_alpha_sigma={f"{k[0]}|{k[1]}": round(v, 5) for k, v in tuning.items()})
            per_seed, epochs = [], []
            for seed in SEEDS:
                model, _, n_ep = train(kind, x_all[~is_val], fit(y_all), x_all[is_val], val(y_all), best, seed, cuts, **tuned)
                per_seed.append(curve_at_years(model.predict_surv_df(x_test)))
                epochs.append(n_ep)
            curves.append(np.mean(per_seed, axis=0))
            info[f"{name}|{sex}"] = dict(selected=best, criterion="validation loss" if kind == "DeepSurv" else "validation integrated Brier score",
                                         validation_score_by_config=[round(x, 5) for x in scores], epochs_by_seed=epochs,
                                         survival_10y_sd_across_seeds=float(np.std([p[:, 9] for p in per_seed], axis=0).mean()), **extra)
            print(f"{name:18s} {sex:7s}: selected {best}{' ' + str(tuned) if kind == 'DeepHit' else ''}, validation scores {np.round(scores, 4)}, "
                  f"epochs {epochs}, {time.time() - t0:.0f}s", flush=True)
        h, w = curves
        pred[name] = {"husband": h, "wife": w, "joint_life": h * w, "last_survivor": h + w - h * w}

np.savez(f"{RES}/deep_predictions.npz", split_seed=SPLIT_SEED, **{f"{m}|{tg}": x for m, p in pred.items() for tg, x in p.items()})
json.dump(dict(setting=dict(train=int((~is_val).sum()), validation=int(is_val.sum()), test=len(dte), grid=GRID, tuning=TUNING, lr=LR,
                            batch=BATCH, patience=PATIENCE, seeds=list(SEEDS), torch=torch.__version__),
               models=info), open(f"{RES}/deep_models.json", "w"), indent=2)
print(f"saved {RES}/deep_predictions.npz in {time.time() - t0:.0f}s")
