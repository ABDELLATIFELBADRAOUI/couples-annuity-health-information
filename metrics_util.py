"""metrics_util.py -- censoring-aware metrics shared by the evaluation scripts (scikit-survival)."""
import numpy as np
from sksurv.metrics import (brier_score, concordance_index_ipcw, cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.nonparametric import kaplan_meier_estimator

TIMES = np.arange(1, 11)                     # evaluation years 1..10


def truncate(pair, tau=10.5):
    """Administrative censoring just after the 10-year horizon: later information is not used, and
    the censoring distribution stays positive at every evaluation time."""
    t, e = pair
    late = t > tau
    return np.where(late, tau, t), np.where(late, False, e)


def metrics(y_train, y_test, S):
    """S: (n, >= 10) survival probabilities after 1, 2, ... years."""
    St = S[:, TIMES - 1]
    risk = 1.0 - St
    auc, _ = cumulative_dynamic_auc(y_train, y_test, risk[:, [4, 9]], [5.0, 10.0])
    return dict(ibs=float(integrated_brier_score(y_train, y_test, St, TIMES.astype(float))),
                brier10=float(brier_score(y_train, y_test, St[:, [9]], [10.0])[1][0]),
                auc5=float(auc[0]), auc10=float(auc[1]),
                c_uno=float(concordance_index_ipcw(y_train, y_test, risk[:, 9], tau=10.0)[0]))


def metrics_pair(y_train, y_test, S):
    """The two metrics used in the bootstrap (same computations as in metrics)."""
    St = S[:, TIMES - 1]
    auc, _ = cumulative_dynamic_auc(y_train, y_test, 1.0 - St[:, [4, 9]], [5.0, 10.0])
    return dict(ibs=float(integrated_brier_score(y_train, y_test, St, TIMES.astype(float))), auc10=float(auc[1]))


def km_at(time, event, t):
    x, s = kaplan_meier_estimator(event, time)
    k = np.searchsorted(x, t, side="right") - 1
    return 1.0 if k < 0 else float(s[k])


def calibration(time, event, S10):
    risk = 1.0 - S10
    cuts = np.quantile(risk, [1 / 3, 2 / 3])
    grp = np.digitize(risk, cuts)
    rows = []
    for g in range(3):
        m = grp == g
        rows.append(dict(predicted=float(risk[m].mean()), observed=1.0 - km_at(time[m], event[m], 10.0), n=int(m.sum())))
    return dict(tertiles=rows, mean_abs_error=float(np.mean([abs(r["predicted"] - r["observed"]) for r in rows])),
                overall_predicted=float(risk.mean()), overall_observed=1.0 - km_at(time, event, 10.0))


# ----------------------------------------------------------------------------- the same metrics without scikit-survival
# scikit-survival refuses to compute a Brier score when an evaluation time is earlier than the first
# time observed in the test sample. For the second death almost nobody is observed before one year,
# so many bootstrap resamples were refused. The functions below follow the same definitions (inverse-
# probability-of-censoring weights from the Kaplan-Meier estimate of the censoring law in the training
# sample) without that restriction; check_metrics.py verifies that they return the values of
# scikit-survival whenever scikit-survival accepts the sample.
def censoring_survival(time_train, event_train):
    """Kaplan-Meier estimate of P(censoring time > t) from the training sample, as a step function."""
    x, s = kaplan_meier_estimator(~np.asarray(event_train, dtype=bool), np.asarray(time_train, dtype=float))

    def G(t):
        k = np.searchsorted(x, np.asarray(t, dtype=float), side="right") - 1
        return np.where(k < 0, 1.0, s[np.maximum(k, 0)])

    return G


def brier_ipcw(G, time, event, St, times):
    """Brier score at each time of `times` (St: (n, len(times)) predicted survival), Graf et al. (1999)."""
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=bool)
    g_own = G(time)
    out = []
    for j, t in enumerate(times):
        case, control = (time <= t) & event, time > t
        with np.errstate(divide="ignore", invalid="ignore"):
            w_case = np.where(case, 1.0 / g_own, 0.0)
        out.append(np.mean(St[:, j] ** 2 * w_case + (1.0 - St[:, j]) ** 2 * control / G(t)))
    return np.array(out)


def ibs_ipcw(G, time, event, St, times):
    b = brier_ipcw(G, time, event, St, times)
    return float(np.sum(0.5 * (b[1:] + b[:-1]) * np.diff(times)) / (times[-1] - times[0]))


def auc_ipcw(G, time, event, risk, t):
    """Cumulative/dynamic AUC at time t (Uno et al. 2007; Hung and Chiang 2010): cases died by t and are
    weighted by 1 / G(time of death); controls are alive after t; ties in the risk count for one half."""
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=bool)
    case, control = (time <= t) & event, time > t
    w = 1.0 / G(time[case])
    rc, rk = risk[case], np.sort(risk[control])
    below = np.searchsorted(rk, rc, side="left")                       # controls with a strictly smaller risk
    ties = np.searchsorted(rk, rc, side="right") - below
    return float(np.sum(w * (below + 0.5 * ties)) / (np.sum(w) * rk.size))


def metrics_own(G, time, event, S):
    """Integrated Brier score (years 1-10), Brier score at 10 years and AUC at 5 and 10 years."""
    St = S[:, TIMES - 1]
    b = brier_ipcw(G, time, event, St, TIMES.astype(float))
    return dict(ibs=float(np.sum(0.5 * (b[1:] + b[:-1])) / (TIMES[-1] - TIMES[0])), brier10=float(b[9]),
                auc5=auc_ipcw(G, time, event, 1.0 - St[:, 4], 5.0), auc10=auc_ipcw(G, time, event, 1.0 - St[:, 9], 10.0))
