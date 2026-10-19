"""
build_assets.py -- every number, table and figure of the manuscript, from the result files.

Run from the project folder:   python paper/build_assets.py [part ...]
It writes paper/numbers.tex (one macro per number quoted in the text, used as \\val{key}), the tables
paper/tables/*.tex and the figures paper/figures/*.pdf. Nothing is typed by hand in the manuscript:
a part whose result files are missing is skipped with a message, and its numbers then appear as
??key?? in the PDF.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

OUT = "paper"
V = {}                                     # key -> text of the macro
SKIPPED = []


# ----------------------------------------------------------------------------- formatting
def put(key, text):
    V[key] = str(text)


def f(x, d=3):
    """Fixed number of decimals, with a true minus sign in math mode handled by the caller."""
    return f"{x:.{d}f}"


def fi(n):
    return f"{int(round(n)):,}".replace(",", "{,}")


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def pct0(x):
    """Percentage without decimals; nan as 'nan' (replaced by -- in the tables), and no '-0'."""
    if x != x:
        return "nan"
    v = round(100 * x)
    return str(0 if v == 0 else v)


def sgn(x, d=3):
    return ("$-$" if x < 0 else "") + f"{abs(x):.{d}f}"


NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def word(n):
    """Small integers in words, as in the text ("five seeds")."""
    return NUMWORD.get(int(n), fi(n))


def write(name, lines, folder="tables"):
    os.makedirs(f"{OUT}/{folder}", exist_ok=True)
    open(f"{OUT}/{folder}/{name}.tex", "w").write("\n".join(lines) + "\n")


# Springer Nature template (sn-jnl.cls): the class sets the table fonts (8 bp), centres the table and prints
# \footnotetext{...} as a note under it; rules are \toprule, \midrule and \botrule. Text width: 31 pc.
SN_TABLE = {                               # name -> overrides for the Springer version: colspec, colsep, env, placement
    "simulation": {"colsep": 2},
    "oos": {"colsep": 2.5},
    "splits": {"colsep": 3.5},
    "trees": {"colsep": 2.5, "rows_to_note": ["Annuity choice of the gradient agent"]},
    "decision": {"colsep": 2},
    "sensitivity": {"colsep": 3},
    "agents": {"colsep": 2.5},
    "groups": {"colsep": 2},
    "hyper": {"colspec": ">{\\raggedright\\arraybackslash}p{3.6cm}>{\\raggedright\\arraybackslash}p{3.4cm}>{\\raggedright\\arraybackslash}p{4.4cm}"},
    "prior": {"colsep": 2.5},
    "effects": {"env": "sidewaystable", "placement": "p"},
    "effects_phi": {"env": "sidewaystable", "placement": "p"},
}
SN_PLACEMENT = "htbp"                      # default float placement of the Springer tables


def table_sn(name, caption, label, colspec, header, rows, notes=None, placement="t", colsep=None):
    o = SN_TABLE.get(name, {})
    env = o.get("env", "table")
    L = [f"\\begin{{{env}}}[{o.get('placement', SN_PLACEMENT)}]", f"\\caption{{{caption}}}\\label{{{label}}}"]
    cs = o.get("colsep", colsep)
    if cs is not None:
        L.append(f"\\setlength{{\\tabcolsep}}{{{cs}pt}}")
    moved = []                                                         # long one-line rows go to the note
    for key in o.get("rows_to_note", []):
        hit = [r for r in rows if key in r]
        assert len(hit) == 1, (name, key)
        m = re.match(r"\\multicolumn\{\d+\}\{l\}\{(.*)\} \\\\$", hit[0])
        assert m, hit[0]
        t = m.group(1).rstrip(".")
        moved.append("For comparison: " + t[0].lower() + t[1:] + ".")
        rows = [r for r in rows if r is not hit[0]]
    while rows and rows[-1] == "\\midrule":
        rows = rows[:-1]
    if moved:
        notes = " ".join(moved) + (" " + notes if notes else "")
    L += [f"\\begin{{tabular}}{{{o.get('colspec', colspec)}}}", "\\toprule"] + header + ["\\midrule"] + rows + ["\\botrule", "\\end{tabular}"]
    if notes:
        L.append(f"\\footnotetext{{{notes}}}")
    L.append(f"\\end{{{env}}}")
    write(name, L, folder="tables_sn")


def table(name, caption, label, colspec, header, rows, notes=None, size="\\small", env="table", placement="t", colsep=None):
    table_sn(name, caption, label, colspec, header, rows, notes=notes, placement=placement, colsep=colsep)
    L = [f"\\begin{{{env}}}[{placement}]", "\\centering", size, f"\\caption{{{caption}}}\\label{{{label}}}"]
    if colsep is not None:
        L.append(f"\\setlength{{\\tabcolsep}}{{{colsep}pt}}")
    if notes:
        L += ["\\begin{threeparttable}"]
    L += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"] + header + ["\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    if notes:
        L += ["\\begin{tablenotes}[flushleft]\\footnotesize", f"\\item {notes}", "\\end{tablenotes}", "\\end{threeparttable}"]
    L += [f"\\end{{{env}}}"]
    write(name, L)


def boot_ci(x, n=2000, seed=7):
    idx = np.random.default_rng(seed).integers(0, x.size, (n, x.size))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def part(fn):
    def run():
        try:
            fn()
            print(f"[ok]      {fn.__name__}")
        except FileNotFoundError as e:
            SKIPPED.append(fn.__name__)
            print(f"[skipped] {fn.__name__}: missing {e.filename}")
    run.__name__ = fn.__name__
    return run


# ----------------------------------------------------------------------------- data
@part
def cohort():
    from pipeline import load_couples, split_indices
    raw = pd.read_csv("cohort_w10.csv")
    d = load_couples("cohort_w10.csv")
    tr, te = split_indices(d)
    put("n.raw", fi(len(raw)))
    put("n.couples", fi(len(d)))
    put("n.train", fi(len(tr)))
    put("n.test", fi(len(te)))
    put("n.maxyears", int(max(d.n_surv_h.max(), d.n_surv_w.max())))
    put("n.deaths.h", fi(d.died_h.sum()))
    put("n.deaths.w", fi(d.died_w.sum()))
    put("n.both", fi((d.died_h & d.died_w).sum()))
    put("n.none", fi((~d.died_h & ~d.died_w).sum()))
    # spouses with at least one year at risk after the year of the partner's death
    wid_h = d.died_w & ((d.n_surv_h > d.n_surv_w + 1) | (d.died_h & (d.n_surv_h == d.n_surv_w + 1)))
    wid_w = d.died_h & ((d.n_surv_w > d.n_surv_h + 1) | (d.died_w & (d.n_surv_w == d.n_surv_h + 1)))
    put("n.widowers", fi(wid_h.sum()))
    put("n.widows", fi(wid_w.sum()))
    put("n.deaths.widowers", fi((wid_h & d.died_h).sum()))
    put("n.deaths.widows", fi((wid_w & d.died_w).sum()))
    put("n.wealth.nonpos", fi((d.wealth <= 0).sum()))
    # what the administrative end of follow-up removes: deaths of the years 11 and later in the file built without it,
    # against the number expected in those years from the age- and sex-specific rates of the years 0 to 10
    old = "archive_before_cap/cohort_w10_before_cap.csv"
    if os.path.exists(old):
        o = load_couples(old)
        n = np.concatenate([o.n_surv_h.values, o.n_surv_w.values]); e = np.concatenate([o.died_h.values, o.died_w.values])
        age = np.concatenate([o.age_h.values, o.age_w.values]); sex = np.repeat([0, 1], len(o))
        late = int((e & (n >= 11)).sum())
        num, den = np.zeros((2, 130)), np.zeros((2, 130))
        for t in range(11):
            risk = (e & (n == t)) | (n > t)
            np.add.at(den, (sex[risk], age[risk] + t), 1.0)
            np.add.at(num, (sex[e & (n == t)], age[e & (n == t)] + t), 1.0)
        k = np.ones(5)
        rate = np.stack([np.convolve(num[s], k, "same") / np.maximum(np.convolve(den[s], k, "same"), 1.0) for s in range(2)])   # five-year moving window
        expected = sum(rate[sex[r], age[r] + t].sum() for t in (11, 12, 13) for r in [(e & (n == t)) | (n > t)])
        put("cap.deaths.removed", fi(late))
        put("cap.deaths.expected", fi(expected))
        put("cap.excess.pct", f(100 * (late - expected) / (int(e.sum()) - (late - expected)), 0))
        put("cap.deaths.before", fi(e.sum()))
    # couples whose two spouses die in the same year of follow-up: observed number, and number expected if the two deaths
    # were independent given age and sex (one-year death probabilities by sex and age, five-year moving window)
    n2 = d[["n_surv_h", "n_surv_w"]].values; e2 = d[["died_h", "died_w"]].values.astype(bool); a2 = d[["age_h", "age_w"]].values
    last = np.where(e2, n2, n2 - 1)                                     # last year at risk of each spouse
    num, den = np.zeros((2, 130)), np.zeros((2, 130))
    for t in range(int(last.max()) + 1):
        for s in range(2):
            risk = last[:, s] >= t
            np.add.at(den[s], a2[risk, s] + t, 1.0)
            np.add.at(num[s], a2[e2[:, s] & (n2[:, s] == t), s] + t, 1.0)
    k = np.ones(5)
    q = np.stack([np.convolve(num[s], k, "same") / np.maximum(np.convolve(den[s], k, "same"), 1.0) for s in range(2)])
    expected = 0.0
    for t in range(int(last.max()) + 1):
        both = (last[:, 0] >= t) & (last[:, 1] >= t)                    # both spouses at risk in year t
        expected += float((q[0, a2[both, 0] + t] * q[1, a2[both, 1] + t]).sum())
    same = int((e2.all(axis=1) & (n2[:, 0] == n2[:, 1])).sum())
    put("n.sameyear", same)
    put("n.sameyear.expected", f(expected, 0))
    ms = lambda x, k=1: f"{x.mean():.{k}f} ({x.std():.{k}f})"
    p = lambda x: f"{100 * x.mean():.1f}"
    rows = [f"Age at baseline, years & {ms(d.age_h)} & {ms(d.age_w)} \\\\",
            f"Body mass index, kg/m$^2$ & {ms(d.bmi_h)} & {ms(d.bmi_w)} \\\\",
            f"Current smoker, \\% & {p(d.smoker_h)} & {p(d.smoker_w)} \\\\",
            f"High blood pressure, \\% & {p(d.hyp_h)} & {p(d.hyp_w)} \\\\",
            f"Diabetes, \\% & {p(d.diab_h)} & {p(d.diab_w)} \\\\",
            f"Education, years & {ms(d.educ_h)} & {ms(d.educ_w)} \\\\",
            f"Deaths observed & {fi(d.died_h.sum())} & {fi(d.died_w.sum())} \\\\",
            f"Person-years at risk & {fi((d.n_surv_h + d.died_h).sum())} & {fi((d.n_surv_w + d.died_w).sum())} \\\\",
            "\\midrule",
            f"Household net wealth, thousand USD & \\multicolumn{{2}}{{c}}{{{d.wealth.median() / 1000:.0f} "
            f"[{d.wealth.quantile(.25) / 1000:.0f}; {d.wealth.quantile(.75) / 1000:.0f}]}} \\\\",
            f"Couples & \\multicolumn{{2}}{{c}}{{{fi(len(d))}}} \\\\",
            f"Couples with both deaths observed & \\multicolumn{{2}}{{c}}{{{fi((d.died_h & d.died_w).sum())}}} \\\\"]
    table("cohort", "Description of the cohort at baseline (2010) and over the follow-up.", "tab:cohort", "lcc",
          ["Characteristic & Husbands & Wives \\\\"], rows,
          notes="Mean (standard deviation), percentage or count; wealth: median [first quartile; third quartile].")
    z = open("results_real/build_prior.txt").read()
    put("prior.n", fi(int(re.search(r"married Wave 4 respondents: (\d+)", z).group(1))))
    put("prior.maxyears", int(re.search(r"administrative end of follow-up: (\d+) completed years", z).group(1)))
    m = re.search(r"deaths observed: men (\d+), women (\d+)", z)
    put("prior.deaths.m", fi(int(m.group(1))))
    put("prior.deaths.w", fi(int(m.group(2))))


# ----------------------------------------------------------------------------- estimates of the full model
EFFECTS = ["Body mass index (+1 SD)", "Current smoker", "High blood pressure", "Diabetes", "Education (+1 SD)", "Household wealth (+1 SD)",
           "Partner's body mass index (+1 SD)", "Partner smokes", "Widowhood"]


@part
def estimates():
    j = json.load(open("results_eval/full_model_summary.json"))
    e = j["effects"]
    assert len(e) == 18
    hr = lambda r: f"{r['hazard_ratio']:.2f} [{r['hr_q025']:.2f}; {r['hr_q975']:.2f}]"
    rows = [f"{EFFECTS[k]} & {hr(e[k])} & {hr(e[9 + k])} \\\\" for k in range(9)]
    k = j["kappa"]
    rows += ["\\midrule", f"Variance of the shared frailty $\\kappa$ & \\multicolumn{{2}}{{c}}{{{k['mean']:.3f} [{k['q025']:.3f}; {k['q975']:.3f}]}} \\\\"]
    table("effects_model", "Hazard ratios of the full model, all couples.", "tab:effects.model", "lcc", ["Effect & Husbands & Wives \\\\"], rows,
          notes="Posterior median [95\\% credible interval]; posterior mean for $\\kappa$. SD: standard deviation.")
    put("hr.h.smoker", f(e[1]["hazard_ratio"], 2))
    put("hr.w.smoker", f(e[10]["hazard_ratio"], 2))
    chronic = [e[i]["hazard_ratio"] for i in (2, 3, 11, 12)]
    put("hr.min.chronic.pct", f(100 * (min(chronic) - 1), 0))
    put("hr.max.chronic.pct", f(100 * (max(chronic) - 1), 0))
    put("hr.h.wealth.pct", f(100 * (1 - e[5]["hazard_ratio"]), 0))
    put("hr.w.wealth.pct", f(100 * (1 - e[14]["hazard_ratio"]), 0))
    put("hr.h.partnersmoker.pct", f(100 * (e[7]["hazard_ratio"] - 1), 0))
    put("hr.w.partnersmoker.pct", f(100 * (e[16]["hazard_ratio"] - 1), 0))
    for s, i in (("h", 8), ("w", 17)):
        put(f"hr.{s}.widow", f(e[i]["hazard_ratio"], 2))
        put(f"hr.{s}.widow.lo", f(e[i]["hr_q025"], 2))
        put(f"hr.{s}.widow.hi", f(e[i]["hr_q975"], 2))
    put("kappa.mean", f(k["mean"]))
    put("kappa.lo", f(k["q025"]))
    put("kappa.hi", f(k["q975"]))
    for s, i in (("h", 8), ("w", 17)):
        put(f"hr.{s}.widow.ppos", f(e[i]["prob_positive"], 2))
    # Kendall's tau implied by a shared gamma frailty of variance kappa (Clayton dependence): kappa / (kappa + 2)
    kd = np.concatenate([np.load(q)["kappa"].reshape(-1) for q in sorted(glob.glob("results_eval/M3_full_*.npz"))])
    tau = kd / (kd + 2.0)
    put("kappa.tau", f(tau.mean(), 2))
    put("kappa.tau.lo", f(np.percentile(tau, 2.5), 2))
    put("kappa.tau.hi", f(np.percentile(tau, 97.5), 2))
    for i, key in ((0, "h.bmi"), (4, "h.educ"), (6, "h.partnerbmi"), (9, "w.bmi"), (13, "w.educ"), (15, "w.partnerbmi")):
        put(f"hr.{key}", f(e[i]["hazard_ratio"], 2))
        put(f"hr.{key}.lo", f(e[i]["hr_q025"], 2))
        put(f"hr.{key}.hi", f(e[i]["hr_q975"], 2))


@part
def cap9():
    """Full model refitted with the follow-up ended after nine years (results_cap9), against the fit of the paper."""
    def load(pattern):
        parts = [np.load(q) for q in sorted(glob.glob(pattern))]
        if len(parts) != 2:
            raise FileNotFoundError(2, "missing", pattern)
        return {k: np.concatenate([q[k].reshape((-1,) + q[k].shape[2:]) for q in parts], axis=0) for k in ("beta", "gamma", "omega", "kappa")}
    a, b = load("results_eval/M3_full_*.npz"), load("results_cap9/M3_full_*.npz")
    from pipeline import load_couples
    d9 = load_couples("results_cap9/cohort_cap9.csv")
    put("cap9.years", int(max(d9.n_surv_h.max(), d9.n_surv_w.max())))
    put("cap9.deaths", fi(d9.died_h.sum() + d9.died_w.sum()))
    put("cap9.couples", fi(len(d9)))
    for s, name in enumerate("hw"):
        q = np.exp(np.percentile(b["omega"][:, s], [2.5, 50, 97.5]))
        put(f"cap9.hr.{name}.widow", f(q[1], 2))
        put(f"cap9.hr.{name}.widow.lo", f(q[0], 2))
        put(f"cap9.hr.{name}.widow.hi", f(q[2], 2))
    put("cap9.kappa.mean", f(b["kappa"].mean()))
    put("cap9.kappa.lo", f(np.percentile(b["kappa"], 2.5)))
    put("cap9.kappa.hi", f(np.percentile(b["kappa"], 97.5)))
    cov_a = np.concatenate([a["beta"].reshape(len(a["beta"]), -1), a["gamma"].reshape(len(a["gamma"]), -1)], axis=1)
    cov_b = np.concatenate([b["beta"].reshape(len(b["beta"]), -1), b["gamma"].reshape(len(b["gamma"]), -1)], axis=1)
    shift = np.abs(cov_b.mean(axis=0) - cov_a.mean(axis=0)) / cov_a.std(axis=0)
    put("cap9.cov.maxshift", f(shift.max(), 1))
    put("cap9.cov.n", cov_a.shape[1])
    put("cap9.cov.maxratio.pct", f(100 * np.abs(np.exp(cov_b.mean(axis=0) - cov_a.mean(axis=0)) - 1).max(), 0))


@part
def diagnostics():
    j = json.load(open("results_eval/diagnostics.json"))
    put("diag.rhat.max", f(max(r["max_rhat"] for r in j.values())))
    put("diag.ess.min", fi(min(r["min_ess_bulk"] for r in j.values())))
    put("diag.div", int(sum(r["divergences"] for r in j.values())))
    put("diag.nfits", len(j))
    name = lambda k: k.replace(" (tous les couples)", ", all couples")
    rows = [f"{name(k)} & {fi(r['couples'])} & {r['parameters']} & {r['divergences']} & {r['max_rhat']:.3f} & {fi(r['min_ess_bulk'])} & {fi(r['min_ess_tail'])} \\\\"
            for k, r in j.items()]
    table("diagnostics", "Convergence diagnostics of the fits (main split and all couples).", "tab:diag", "lrrrrrr",
          ["Model & Couples & Parameters & Divergences & Largest $\\widehat{R}$ & \\multicolumn{2}{c}{Smallest ESS} \\\\",
           " & & & & & bulk & tail \\\\"], rows,
          notes="Four chains of 1,000 draws after 1,000 warm-up iterations. ESS: effective sample size.")


# ----------------------------------------------------------------------------- decision analysis
BELIEF = {"POP": "POP: age and sex", "MD": "MD: dependence only", "M0": "M0: own health only", "M1": "M1: M0 + partner",
          "M2": "M2: M1 + widowhood", "MARG": "MARG: no dependence", "TRUE": "Truth: full model \\Mfull{}"}


def _losses(files):
    """Losses (in %) of every belief found in the scenario files, on the couples of the first file."""
    from decision_dp import Params, wealth_equivalent_loss
    z0 = np.load(files[0])
    par = Params(g=float(z0["params"][0]), n=float(z0["params"][1]), r=float(z0["params"][2]), beta=float(z0["params"][3]),
                 load=float(z0["params"][4]), xi=float(z0["params"][5]), phis=tuple(z0["phis"]))
    out, dec = {}, {}
    for fpath in files:
        z = np.load(fpath)
        assert np.array_equal(z["index"], z0["index"])
        assert np.allclose(z["true|TRUE"], z0["true|TRUE"], rtol=1e-9), "the optimum differs between files of the same scenario"
        for key in z.files:
            if key.startswith("true|"):
                m = key.split("|")[1]
                dec[m] = (z[f"alpha|{m}"], z[f"phi|{m}"])
                if m != "TRUE":
                    out[m] = 100 * wealth_equivalent_loss(z[key], z["true|TRUE"], par)
                    out[m + "|choice"] = 100 * wealth_equivalent_loss(z[f"true_action|{m}"], z["true|TRUE"], par)
    return out, dec, len(z0["index"])


def _stats(x):
    lo, hi = boot_ci(x)
    return dict(mean=x.mean(), lo=lo, hi=hi, median=np.median(x), q90=np.percentile(x, 90), max=x.max(), above1=np.mean(x > 1.0), above01=np.mean(x > 0.1))


@part
def decision():
    rows = []
    for tag, beliefs, title in (("base", ["POP", "MD", "M0", "M1", "M2", "MARG"], "A. No pre-existing annuity"),
                                ("pre50", ["POP", "MD", "M0", "MARG"], "B. Half of the resources already annuitised")):
        loss, dec, n = _losses([f"results_decision/{tag}_limit.npz"])
        assert all(m in loss for m in beliefs)
        a_true, p_true = dec["TRUE"]
        buy_true = a_true > 0
        rows += [f"\\multicolumn{{9}}{{l}}{{\\textit{{{title}}}}} \\\\"]

        def choice_cols(m):
            a, p = dec[m]
            buy = a > 0
            same = np.mean(np.isclose(p, p_true)[buy & buy_true]) if m != "TRUE" else np.nan
            put(f"dec.{tag}.{m}.alpha", pct(a.mean()))
            put(f"dec.{tag}.{m}.phi50", pct(np.mean(np.isclose(p[buy], 0.5))))
            put(f"dec.{tag}.{m}.below50", pct(np.mean(a < 0.5)))
            if m != "TRUE":
                put(f"dec.{tag}.{m}.same", pct(same))
            return f"{pct(a.mean())} & {pct(np.mean(np.isclose(p[buy], 0.5)))} & " + ("--" if m == "TRUE" else pct(same))

        rows.append(f"{BELIEF['TRUE']} & -- & -- & -- & -- & -- & {choice_cols('TRUE')} \\\\")
        for m in beliefs:
            if m not in loss:
                continue
            s = _stats(loss[m])
            for k in ("mean", "lo", "hi", "median", "q90"):
                put(f"loss.{tag}.{m}.{k}", f(s[k]))
            put(f"loss.{tag}.{m}.max", f(s["max"], 1))
            put(f"loss.{tag}.{m}.above1", pct(s["above1"]))
            put(f"loss.{tag}.{m}.choice", pct(loss[m + "|choice"].mean() / loss[m].mean(), 0))
            rows.append(f"{BELIEF[m]} & {f(s['mean'])} & [{f(s['lo'])}; {f(s['hi'])}] & {f(s['median'])} & {f(s['q90'])} & {pct(s['above1'])} & {choice_cols(m)} \\\\")
        for a, b in (("POP", "M0"), ("POP", "MARG"), ("POP", "MD")):
            if a in loss and b in loss:
                put(f"loss.{tag}.ratio.{a}.{b}", f(loss[a].mean() / loss[b].mean(), 1))
        if tag == "base":
            rows.append("\\midrule")
            put("dec.ncouples", fi(n))
    table("decision", "Decisions and wealth-equivalent losses by belief (held-out couples).", "tab:decision", "lcccccccc",
          ["Belief & \\multicolumn{5}{c}{Loss, \\% of resources} & \\multicolumn{3}{c}{Decision} \\\\", "\\cmidrule(lr){2-6}\\cmidrule(lr){7-9}",
           " & Mean & 95\\% interval & Median & P90 & $>$1\\%, \\% & $\\bar\\alpha$, \\% & $\\phi{=}50\\%$, \\% & Same $\\phi$, \\% \\\\"], rows,
          notes="Loss: share of the initial resources a household would give up to decide with the full model. Intervals: bootstrap over couples. "
                "P90: 90th percentile. $>$1\\%: share of households losing more than 1\\%. $\\bar\\alpha$: mean annuitised share (panel B: share of the resources not already annuitised). "
                "$\\phi{=}50\\%$: share of annuity buyers choosing the 50\\% survivor share. Same $\\phi$: share choosing the survivor share of the truth, "
                "among the households that buy an annuity under the belief and under the truth. --: not applicable (the truth is the reference). "
                "Panel B: M1 and M2 not computed.",
          size="\\footnotesize", colsep=3)
    j = json.load(open("results_decision/decision_summary_limit.json"))
    q = j["base"]["by_relative_health"]
    for k, name in ((0, "q1"), (2, "q3"), (4, "q5")):
        put(f"loss.base.POP.{name}", f(q[k]["loss_POP"], 2))
    assert abs(j["base"]["beliefs"]["POP"]["mean"] - float(V["loss.base.POP.mean"])) < 6e-4            # same numbers as the project's own summary
    put("dp.bracket", f(100 * 0.6180339887 ** 13, 2))
    z = open("results_decision/check_decision.txt").read()
    put("dp.check.improve", f"{float(re.search(r'largest improvement found on the DP policy: ([0-9.e+-]+) %', z).group(1)):.5f}\\%")
    put("dp.check.grid", f(max(float(x) for x in re.findall(r"largest change for one couple ([0-9.e+-]+) points", z)), 3))
    if os.path.exists("results_decision/base_truthM0_limit.npz"):
        loss, _, _ = _losses(["results_decision/base_truthM0_limit.npz"])
        s = _stats(loss["POP"])
        for k in ("mean", "lo", "hi"):
            put(f"loss.truthM0.POP.{k}", f(s[k]))


SCEN = [("base", "Base case"), ("g2", "Risk aversion 2"), ("g5", "Risk aversion 5"), ("load0", "No loading"), ("load20", "Loading of 20\\%"),
        ("freephi", "13 survivor shares")]


@part
def sensitivity():
    cols = ["POP", "MD", "M0", "MARG"]
    base, _, _ = _losses(["results_decision/base_limit.npz"])
    rows, res, index = [], {}, None
    for s, title in SCEN[1:]:
        loss, _, n = _losses([f"results_decision/{s}_limit.npz"])
        idx = np.load(f"results_decision/{s}_limit.npz")["index"]
        assert index is None or np.array_equal(idx, index)
        index = idx
        res[s] = {m: loss[m].mean() for m in cols}
    res = {"base": {m: base[m][index].mean() for m in cols}, **res}
    for s, title in SCEN:
        rows.append(f"{title} & " + " & ".join(f(res[s][m]) for m in cols) + f" & {res[s]['POP'] / res[s]['M0']:.1f} & {res[s]['POP'] / res[s]['MARG']:.1f} \\\\")
    put("sens.ncouples", fi(len(index)))
    # value of information as differences of losses (differences of certainty-equivalent resources)
    put("sens.health.base", f(res["base"]["POP"] - res["base"]["MARG"]))
    put("value.health", f(base["POP"].mean() - base["MARG"].mean()))
    put("value.dep", f(base["MARG"].mean()))
    put("value.dep.nohealth", sgn(base["POP"].mean() - base["MD"].mean()))
    put("value.health.withdep", f(base["MD"].mean()))
    put("value.ratio", f((base["POP"].mean() - base["MARG"].mean()) / base["MARG"].mean(), 0))
    pre, _, _ = _losses(["results_decision/pre50_limit.npz"])
    health = [base["POP"].mean() - base["MARG"].mean(), pre["POP"].mean() - pre["MARG"].mean()] + [res[s]["POP"] - res[s]["MARG"] for s, _ in SCEN[1:]]
    dep = [base["MARG"].mean(), pre["MARG"].mean()] + [res[s]["MARG"] for s, _ in SCEN[1:]]
    put("range.health.min", f(min(health), 2))
    put("range.health.max", f(max(health), 2))
    put("range.dep.min", f(min(dep), 2))
    put("range.dep.max", f(max(dep), 2))
    for m in cols:
        put(f"sens.base.{m}.mean", f(res["base"][m]))
    if os.path.exists("results_decision/strong_limit.npz"):
        # stress case: stronger dependence imposed on the truth; loss of the belief that ignores it (same couples)
        from decision_setup import STRONG
        z = np.load("results_decision/strong_limit.npz")
        assert np.array_equal(z["index"], index) and str(z["truth"]) == "STRONG"
        loss, _, _ = _losses(["results_decision/strong_limit.npz"])
        st = _stats(loss["MARGS"])
        for k in ("mean", "lo", "hi", "median"):
            put(f"loss.strong.MARGS.{k}", f(st[k]))
        put("loss.strong.MARGS.max", f(st["max"], 1))
        if "POP" in loss and "MD" in loss:
            put("loss.strong.POP.mean", f(loss["POP"].mean()))
            put("loss.strong.MD.mean", f(loss["MD"].mean()))
            put("strong.value.health", f(loss["POP"].mean() - loss["MARGS"].mean()))
            put("strong.value.dep", f(loss["MARGS"].mean()))
        put("strong.hr", f(STRONG["hazard_ratio"], 1))
        put("strong.kappa", f(STRONG["kappa"], 1))
        put("loss.strong.ratio.base", f(st["mean"] / res["base"]["MARG"], 0))
    alt = [s for s, _ in SCEN[1:]]
    for m in cols:
        put(f"sens.{m}.min", f(min(res[s][m] for s in alt), 2))
        put(f"sens.{m}.max", f(max(res[s][m] for s in alt), 2))
    ratios = [res[s]["POP"] / res[s]["M0"] for s in alt]
    put("sens.ratio.min", f(min(ratios), 1))
    put("sens.ratio.max", f(max(ratios), 1))
    put("sens.order.holds", "yes" if all(res[s]["POP"] > res[s]["MD"] > res[s]["M0"] > res[s]["MARG"] for s in res) else "no")
    put("sens.POP.largest", "yes" if all(res[s]["POP"] > max(res[s][m] for m in ("MD", "M0", "MARG")) for s in res) else "no")
    FREE = np.load("results_decision/freephi_limit.npz")["phis"]
    assert len(FREE) == 13 and np.allclose(np.diff(FREE), 0.05), FREE            # the label of the scenario says 13 shares
    table("sensitivity", "Mean wealth-equivalent loss by scenario.", "tab:sens", "lcccccc",
          ["Scenario & \\multicolumn{4}{c}{Mean loss, \\% of resources} & \\multicolumn{2}{c}{Ratio of mean losses} \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}", " & POP & MD & M0 & MARG & POP / M0 & POP / MARG \\\\"], rows,
          notes=f"{fi(len(index))} held-out couples drawn at random, the same in every scenario. Base case: setting of Table~\\ref{{tab:decision}}, panel A, "
                f"on these couples. 13 survivor shares: {pct(min(FREE), 0)}\\% to {pct(max(FREE), 0)}\\% by steps of 5 points.")
    j = json.load(open("results_decision/uncertainty_summary.json"))
    put("unc.draws", j["draws"])
    put("unc.couples", j["couples"])
    for m, r in j["beliefs"].items():
        for k in ("mean", "sd", "min", "max"):
            put(f"unc.{m}.{k}", f(r[k]))
        if "main_analysis_same_couples" in r:
            put(f"unc.{m}.main", f(r["main_analysis_same_couples"]))
    for k, v in j["share_of_draws"].items():
        put("unc.share." + k.replace(">", "."), pct(v, 0))


# ----------------------------------------------------------------------------- assessment of the mortality model
SIM_LABEL = {"base": "Estimated values", "indep": "No dependence", "frailty": "Frailty, no widowhood effect", "widow": "Widowhood effect, no frailty",
             "both": "Frailty and widowhood effect", "small": "Estimated values, 1,000 couples", "short": "Estimated values, 6 years"}


@part
def simulation():
    full = json.load(open("results_sim/sim_summary.json"))
    j = {k: v for k, v in full["scenarios"].items() if k in SIM_LABEL}
    if len(j) < len(SIM_LABEL):
        raise FileNotFoundError(2, "missing", "results_sim/<scenario>_*.npz (all seven scenarios)")
    reps = sorted({s["replications"] for s in j.values()})
    put("sim.reps", reps[0] if len(reps) == 1 else f"{reps[0]} to {reps[-1]}")
    rows = []
    cov = lambda x: "--" if x is None or (isinstance(x, float) and np.isnan(x)) else pct(x, 0)
    for sc, r in j.items():
        g = r["groups"]
        om, ka = g["omega"], g["kappa"]
        est = lambda grp, k: grp["truth"][k] + grp["bias"][k]
        rows.append(f"{SIM_LABEL[sc]} & {r['replications']} & {sgn(om['truth'][0], 2)} & {sgn(est(om, 0), 2)} & {sgn(om['truth'][1], 2)} & {sgn(est(om, 1), 2)} & "
                    f"{sgn(ka['truth'][0], 2)} & {sgn(est(ka, 0), 2)} & {cov(om['coverage'])} & {cov(g['beta']['coverage'])} & {cov(g['p0']['coverage'])} & {r['max_rhat']:.3f} \\\\")
    table("simulation", "Simulation study: recovery of the dependence parameters and coverage.", "tab:sim", "lrcccccccccc",
          ["Scenario & Rep. & \\multicolumn{2}{c}{$\\omega_h$} & \\multicolumn{2}{c}{$\\omega_w$} & \\multicolumn{2}{c}{$\\kappa$} & "
           "\\multicolumn{3}{c}{Coverage, \\%} & $\\widehat{R}$ \\\\", "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-11}",
           " & & true & est. & true & est. & true & est. & $\\omega$ & $\\beta$ & $p$ & max \\\\"], rows,
          notes="Rep.: replications. est.: mean of the posterior means over the replications. Coverage: share of the 95\\% credible intervals that contain "
                "the true value, pooled over the parameters of the group and the replications ($\\omega$: 2 parameters; $\\beta$: 12; $p$: baseline "
                "probabilities at ages 55, 65, 75 and 85 for each sex). $\\widehat{R}$: largest value over the replications.", size="\\footnotesize")
    fr_, wi, ind = j["frailty"]["groups"], j["widow"]["groups"], j["indep"]["groups"]
    put("sim.frailty.omega.mean", sgn(np.mean(fr_["omega"]["bias"]), 2))
    put("sim.frailty.kappa.mean", f(fr_["kappa"]["truth"][0] + fr_["kappa"]["bias"][0], 2))
    put("sim.widow.omega.true", f(wi["omega"]["truth"][0], 2))
    put("sim.widow.omega.mean", f(wi["omega"]["truth"][0] + np.mean(wi["omega"]["bias"]), 2))
    put("sim.widow.kappa.mean", f(wi["kappa"]["bias"][0], 2))
    put("sim.indep.omega.reject", pct(np.mean(ind["omega"]["excludes_zero"]), 0))
    put("sim.indep.kappa.mean", f(ind["kappa"]["bias"][0], 2))
    put("sim.indep.omega.mean", sgn(np.mean(ind["omega"]["bias"]), 2))
    covs = [r["groups"]["beta"]["coverage"] for r in j.values()]
    put("sim.cov.beta.min", pct(min(covs), 0))
    put("sim.cov.beta.max", pct(max(covs), 0))
    covs = [r["groups"]["omega"]["coverage"] for r in j.values()]
    put("sim.cov.omega.min", pct(min(covs), 0))
    put("sim.cov.omega.max", pct(max(covs), 0))
    put("sim.pred.mae.max", f(100 * max(v["mae"] for r in j.values() for v in r["prediction"].values()), 1))
    put("sim.rhat.max", f(max(r["max_rhat"] for r in j.values()), 2))
    put("sim.rhat.share105", pct(np.mean([r["share_rhat_above_1_05"] for r in j.values()]), 0))
    put("sim.div", int(sum(r["divergences"] for r in j.values())))
    put("sim.nfits", fi(sum(r["replications"] for r in j.values())))
    put("sim.base.deaths.h", fi(j["base"]["deaths_mean"][0]))
    put("sim.base.deaths.w", fi(j["base"]["deaths_mean"][1]))
    put("sim.base.both", fi(j["base"]["both_died_mean"]))
    put("sim.base.kappa.mean", f(j["base"]["groups"]["kappa"]["truth"][0] + j["base"]["groups"]["kappa"]["bias"][0], 3))
    put("sim.base.kappa.true", f(j["base"]["groups"]["kappa"]["truth"][0], 3))
    put("sim.base.omega.cov", pct(j["base"]["groups"]["omega"]["coverage"], 0))
    for sc in ("short", "small"):
        g = j[sc]["groups"]
        put(f"sim.{sc}.omega.h", f(g["omega"]["truth"][0] + g["omega"]["bias"][0], 2))
        put(f"sim.{sc}.omega.w", f(g["omega"]["truth"][1] + g["omega"]["bias"][1], 2))
        put(f"sim.{sc}.kappa", f(g["kappa"]["truth"][0] + g["kappa"]["bias"][0], 2))
    put("sim.base.omega.h.true", f(j["base"]["groups"]["omega"]["truth"][0], 2))
    put("sim.base.omega.w.true", f(j["base"]["groups"]["omega"]["truth"][1], 2))
    put("sim.base.kappa.true2", f(j["base"]["groups"]["kappa"]["truth"][0], 2))
    put("sim.base.omega.power.h", pct(j["base"]["groups"]["omega"]["excludes_zero"][0], 0))
    put("sim.base.omega.power.w", pct(j["base"]["groups"]["omega"]["excludes_zero"][1], 0))
    # replications whose short run did not converge and that were refitted with the reference protocol
    put("sim.refits", len(glob.glob("results_sim/failed/*_short_run.npz")))
    pc = full.get("protocol_check", {})
    if pc:
        put("sim.protocol.reps", " and ".join(str(v["replications"]) for v in pc.values()))
        reps = [int(v["replications"]) for v in pc.values()]
        put("sim.protocol.reps.text", f"{word(reps[0])} data sets of each" if len(set(reps)) == 1
            else " and ".join(map(str, reps)) + " data sets, respectively,")
        put("sim.protocol.maxsd", f(max(v[k]["max_difference_in_posterior_sd"] for v in pc.values() for k in ("omega", "kappa", "beta")), 2))


PRIOR_LABEL = {"base": "Reference", "c_quarter": "Urn strength $\\div\\,4$", "c_times4": "Urn strength $\\times\\,4$", "centre_low": "Prior guess $\\times\\,0.7$",
               "centre_high": "Prior guess $\\times\\,1.3$", "scale05": "Effects: Normal(0, 0.5)", "scale2": "Effects: Normal(0, 2)",
               "student": "Effects: Student, 3 d.f.", "kappa05": "Frailty: half-normal(0.5)", "kappa2": "Frailty: half-normal(2)"}


@part
def prior():
    j = json.load(open("results_sens/sens_summary.json"))["settings"]
    if "base" not in j or len(j) < len(PRIOR_LABEL):
        raise FileNotFoundError(2, "missing", "results_sens/<setting>.npz (all ten settings)")
    hr = lambda r: f"{r['median']:.2f} [{r['q025']:.2f}; {r['q975']:.2f}]"
    rows = []
    for name, r in j.items():
        d = r["survival10_abs_difference"]
        rows.append(f"{PRIOR_LABEL[name]} & {hr(r['widowhood_hr'][0])} & {hr(r['widowhood_hr'][1])} & {r['kappa']['mean']:.3f} [{r['kappa']['q025']:.3f}; {r['kappa']['q975']:.3f}] & "
                    f"{100 * r['p0_relative_difference']:.1f} & {100 * d['joint_life']['mean']:.2f} \\\\")
    # the integrated Brier scores barely move: their range goes to the note, and the table fits the width of the page
    ibs = {tg: [r["metrics"][tg]["ibs"] for r in j.values()] for tg in ("joint_life", "last_survivor")}
    span = lambda v: f"{min(v):.4f}" if f"{min(v):.4f}" == f"{max(v):.4f}" else f"{min(v):.4f} to {max(v):.4f}"
    chains = sorted({r["chains"] for r in j.values()})
    draws = sorted({r["draws"] for r in j.values()})
    assert len(chains) == 1 and len(draws) == 1
    table("prior", "Sensitivity of the full model to the prior.", "tab:prior", "lccccc",
          ["Prior & \\multicolumn{2}{c}{Widowhood hazard ratio} & $\\kappa$ & Baseline & Survival \\\\",
           "\\cmidrule(lr){2-3}", " & husbands & wives & & diff., \\% & diff., pp \\\\"], rows,
          notes=f"Estimation couples; one choice changed at a time; {word(chains[0])} chains of {fi(draws[0])} draws after 1{{,}}000 warm-up iterations per fit. "
                "Reference: the first two of the four chains of the fit used as the truth in Section~\\ref{sec:decision}. "
                "Posterior median [95\\% interval] of the hazard ratios; posterior mean [95\\% interval] of $\\kappa$. "
                "Baseline diff.: mean absolute relative difference of the posterior mean baseline death probabilities from the reference, ages 50 to 90, both sexes. "
                "Survival diff.: mean absolute difference of the predicted ten-year joint survival from the reference, in percentage points, held-out couples. "
                f"Integrated Brier score of the held-out couples in every fit: {span(ibs['joint_life'])} (first death) and {span(ibs['last_survivor'])} (second death).",
          size="\\footnotesize", colsep=3)
    # the truth of the decision analysis pools the four chains of the full model on the estimation couples (decision_setup.py)
    parts = [np.load(p) for p in sorted(glob.glob("results_eval/M3_train_*.npz"))]
    assert len(parts) == 2
    om = np.concatenate([np.exp(z["omega"]).reshape(-1, 2) for z in parts])
    ka = np.concatenate([z["kappa"].ravel() for z in parts])
    for s, k in (("h", 0), ("w", 1)):
        put(f"truth.widow.{s}", f(np.median(om[:, k]), 2))
        put(f"truth.widow.{s}.lo", f(np.quantile(om[:, k], 0.025), 2))
        put(f"truth.widow.{s}.hi", f(np.quantile(om[:, k], 0.975), 2))
    put("truth.kappa", f(ka.mean()))
    alt = [r for n, r in j.items() if n != "base"]
    for s, k in (("h", 0), ("w", 1)):
        put(f"sens.widow.{s}.min", f(min(r["widowhood_hr"][k]["median"] for r in alt), 2))
        put(f"sens.widow.{s}.max", f(max(r["widowhood_hr"][k]["median"] for r in alt), 2))
    put("sens.kappa.min", f(min(r["kappa"]["mean"] for r in alt)))
    put("sens.kappa.max", f(max(r["kappa"]["mean"] for r in alt)))
    put("sens.pred.max", f(100 * max(r["survival10_abs_difference"]["joint_life"]["mean"] for r in alt), 2))
    put("sens.ibs.maxdiff", f(max(abs(r["metrics"][tg]["ibs"] - j["base"]["metrics"][tg]["ibs"]) for r in alt for tg in r["metrics"]), 4))
    put("sens.div", int(sum(r["divergences"] for r in j.values())))
    put("sens.rhat.max", f(max(r["max_rhat"] for r in j.values())))
    worst = max((n for n in j if n != "base"), key=lambda n: j[n]["survival10_abs_difference"]["joint_life"]["mean"])
    put("sens.most", PRIOR_LABEL[worst])


MODEL_LABEL = {"M0": "M0", "M1": "M1", "M2": "M2", "M3": "\\Mfull{} (full model)", "MD": "MD (no covariates)", "Cox": "Cox", "Cox+partner": "Cox + partner",
               "Gompertz": "Gompertz", "Copula": "Gompertz--Frank copula", "Copula+partner": "Copula + partner", "DeepSurv": "DeepSurv",
               "DeepSurv+partner": "DeepSurv + partner", "DeepHit": "DeepHit", "DeepHit+partner": "DeepHit + partner"}
TG = ["husband", "wife", "joint_life", "last_survivor"]
TGK = {"husband": "h", "wife": "w", "joint_life": "joint", "last_survivor": "last"}


@part
def oos():
    e = json.load(open("results_eval/evaluation.json"))
    if "MD" not in e["targets"]["husband"]["models"] or "Copula" not in e["targets"]["husband"]["models"] or not os.path.exists("results_eval/evaluation_boot.npz"):
        raise FileNotFoundError(2, "missing", "results_eval/evaluation.json with MD, the copula model and the bootstrap file")
    boot = np.load("results_eval/evaluation_boot.npz")
    models = [m for m in MODEL_LABEL if m in e["targets"]["husband"]["models"]]
    put("eval.nboot", fi(e["setting"]["bootstrap"]))
    rows = []
    for m in models:
        r = {tg: e["targets"][tg]["models"][m] for tg in TG}
        rows.append(f"{MODEL_LABEL[m]} & " + " & ".join(f"{r[tg]['ibs']:.4f}" for tg in TG) + " & " + " & ".join(f"{r[tg]['auc10']:.3f}" for tg in TG)
                    + f" & {r['joint_life']['calibration']['mean_abs_error']:.3f} & {r['last_survivor']['calibration']['mean_abs_error']:.3f} \\\\")
        if m in ("MD", "Cox+partner", "Copula+partner"):
            rows.append("\\addlinespace")
        for tg in TG:
            put(f"eval.{TGK[tg]}.{m.replace('+', '')}.ibs", f(r[tg]["ibs"], 4))
            put(f"eval.{TGK[tg]}.{m.replace('+', '')}.auc", f(r[tg]["auc10"]))
            put(f"eval.{TGK[tg]}.{m.replace('+', '')}.cal", f(r[tg]["calibration"]["mean_abs_error"]))
    ev = "; ".join(f"{name} {e['targets'][tg]['events_by_10']}" for tg, name in zip(TG, ("husband", "wife", "first death", "second death")))
    table("oos", "Out-of-sample accuracy on the held-out couples (main split).", "tab:oos", "lcccccccccc",
          ["Model & \\multicolumn{4}{c}{Integrated Brier score} & \\multicolumn{4}{c}{AUC at 10 years} & \\multicolumn{2}{c}{Calibration} \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-11}", " & H & W & First & Second & H & W & First & Second & First & Second \\\\"], rows,
          notes=f"H: death of the husband; W: death of the wife; First, Second: first and second death of the couple. {fi(e['setting']['test'])} held-out couples; "
                f"deaths within ten years: {ev}. Calibration: calibration error at ten years. Lower is better for the Brier score and the calibration error, higher for the AUC.", size="\\footnotesize", colsep=4)

    withcov = [e["targets"]["joint_life"]["models"][m]["ibs"] for m in models if m != "MD"]      # every model except MD has covariates
    put("eval.joint.cov.ibs.min", f(min(withcov), 4))
    put("eval.joint.cov.ibs.max", f(max(withcov), 4))
    nowid = [m for m in models if m not in ("M2", "M3", "MD")]
    cal = [e["targets"]["last_survivor"]["models"][m]["calibration"]["mean_abs_error"] for m in nowid]
    put("eval.last.cal.nowidow.min", f(min(cal)))
    put("eval.last.cal.nowidow.max", f(max(cal)))

    def diff(tg, a, b, k="ibs"):
        x = boot[f"{tg}|{a}|{k}"] - boot[f"{tg}|{b}|{k}"]
        return float(x.mean()), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    for tg in TG:
        for a, b, name in (("M3", "M0", "dM3M0"), ("M3", "MD", "dM3MD"), ("M3", "Cox", "dM3Cox"), ("M3", "Copula", "dM3Copula"), ("DeepSurv", "M3", "dDeepSurvM3"),
                           ("DeepHit", "M3", "dDeepHitM3"), ("DeepSurv", "Cox", "dDeepSurvCox"), ("Copula", "Gompertz", "dCopulaGompertz")):
            if f"{tg}|{a}|ibs" in boot.files and f"{tg}|{b}|ibs" in boot.files:
                for k, d in (("ibs", 4), ("auc10", 3)):
                    mean, lo, hi = diff(tg, a, b, k)
                    kk = "ibs" if k == "ibs" else "auc"
                    put(f"eval.{TGK[tg]}.{name}.{kk}", sgn(mean, d))
                    put(f"eval.{TGK[tg]}.{name}.{kk}.lo", sgn(lo, d))
                    put(f"eval.{TGK[tg]}.{name}.{kk}.hi", sgn(hi, d))
    c = json.load(open("results_eval/copula_fit.json"))
    put("copula.theta", f(c["Copula"]["theta"], 2))
    put("copula.theta.se", f(c["Copula"]["theta_se"], 2))
    put("copula.tau", f(c["Copula"]["kendall_tau"], 2))
    put("copula.lr", f(c["Copula"]["lr_test_vs_independence"], 2))
    put("copula.recovery.reps", c["recovery"]["theta_as_fitted"]["replications"])
    put("copula.recovery.mean", f(c["recovery"]["theta_as_fitted"]["theta_mean"], 2))
    put("copula.recovery.mean3", f(c["recovery"]["theta_3"]["theta_mean"], 2))
    d = json.load(open("results_eval/deep_models.json"))
    for name in ("DeepHit|husband", "DeepHit|wife"):
        sel = d["models"][name]["selected_alpha_sigma"]
        put(f"deephit.{name.split('|')[1]}.alpha", sel["alpha"])
        put(f"deephit.{name.split('|')[1]}.sigma", sel["sigma"])


@part
def splits():
    j = json.load(open("results_seeds/splits_summary.json"))
    if "Copula" not in j["models"]:                       # MD is fitted on the main split only
        raise FileNotFoundError(2, "missing", "results_seeds/splits_summary.json with the copula model")
    rows = []
    for m in [m for m in MODEL_LABEL if m in j["models"]]:
        r = {tg: j["metrics"][tg][m] for tg in TG}
        rows.append(f"{MODEL_LABEL[m]} & " + " & ".join(f"{r[tg]['ibs']['mean']:.4f}" for tg in TG) + " & " + " & ".join(f"{r[tg]['auc10']['mean']:.3f}" for tg in TG) + " \\\\")
    put("splits.n", len(j["splits"]))
    absent = [m for m in ("M1", "M2", "MD") if m not in j["models"]]
    assert absent == ["M1", "M2", "MD"], absent
    table("splits", "Out-of-sample accuracy averaged over five random splits.", "tab:splits", "lcccccccc",
          ["Model & \\multicolumn{4}{c}{Integrated Brier score} & \\multicolumn{4}{c}{AUC at 10 years} \\\\", "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
           " & H & W & First & Second & H & W & First & Second \\\\"], rows,
          notes="Mean over the five splits; each model of the table is refitted on each split. M1, M2 and MD are fitted on the main split only "
                "(Table~\\ref{tab:oos}). Columns as the Brier score and AUC columns of Table~\\ref{tab:oos}.", size="\\footnotesize", placement="h")
    for tg in TG:
        d = j["differences_vs_M0"][tg]
        for m in ("M3", "DeepSurv", "DeepSurvpartner", "DeepHit", "DeepHitpartner", "Copula", "Cox"):
            mm_ = m.replace("partner", "+partner")
            if mm_ in d:
                put(f"splits.{TGK[tg]}.{m}.dibs", sgn(d[mm_]["d_ibs_vs_M0"]["mean"], 4))
                put(f"splits.{TGK[tg]}.{m}.better", d[mm_]["d_ibs_vs_M0"]["splits_better"])
                put(f"splits.{TGK[tg]}.{m}.worse", d[mm_]["d_ibs_vs_M0"]["splits_worse"])
                put(f"splits.{TGK[tg]}.{m}.dauc", sgn(d[mm_]["d_auc10_vs_M0"]["mean"], 3))
                put(f"splits.{TGK[tg]}.{m}.auc.better", d[mm_]["d_auc10_vs_M0"]["splits_better"])
        for m in j["models"]:
            put(f"splits.{TGK[tg]}.{m.replace('+', '')}.auc", f(j["metrics"][tg][m]["auc10"]["mean"]))
            put(f"splits.{TGK[tg]}.{m.replace('+', '')}.ibs", f(j["metrics"][tg][m]["ibs"]["mean"], 4))
    worst = {tg: max(j["metrics"][tg][m]["ibs"]["max"] - j["metrics"][tg][m]["ibs"]["min"] for m in j["models"]) for tg in TG}
    put("splits.ibs.range.max", f(max(worst.values()), 4))


@part
def annuity():
    j = json.load(open("results_eval/annuity_gap.json"))
    signed = lambda x: ("+" if x >= 0 else "$-$") + f"{abs(100 * x):.1f}"
    for tg, key in (("joint_life", "joint"), ("last_survivor", "last")):
        r = j["B_within_M3"][tg]
        put(f"gap.{key}.mean", signed(r["portfolio_mean"]))
        put(f"gap.{key}.lo", signed(r["portfolio_q025"]))
        put(f"gap.{key}.hi", signed(r["portfolio_q975"]))
        r = j["A_M3_vs_M0"][tg]
        put(f"gap.{key}.vsM0.mean", signed(r["portfolio_mean"]))
        put(f"gap.{key}.vsM0.lo", signed(r["portfolio_q025"]))
        put(f"gap.{key}.vsM0.hi", signed(r["portfolio_q975"]))
    put("gap.rate", f"{100 * j['setting']['rate']:.0f}")


# ----------------------------------------------------------------------------- agents
KIND = {"apg": "Gradient through the simulator", "ppoa": "PPO, annealed noise", "ppo": "PPO, learned noise"}
KIND_SHORT = {"apg": "Gradient agent", "ppoa": "PPO, annealed noise", "ppo": "PPO, learned noise"}
million = lambda n: f"{n / 1e6:.2f}"


@part
def agents():
    j = json.load(open("results_rl/agents_summary.json"))
    rule = json.load(open("results_rl/rule_baseline_af.json"))
    A = j["agents"]
    groups = {}
    for name, r in A.items():
        groups.setdefault((r["kind"], r["lives"]), []).append(r)
    if not any(k[0].startswith("ppo") for k in groups):
        raise FileNotFoundError(2, "missing", "results_rl/ppo*_seed*.npz (PPO agents)")
    rows = []
    mean = lambda rs, fn: float(np.mean([fn(r) for r in rs]))
    order = sorted(groups, key=lambda k: (list(KIND).index(k[0]), -k[1] if k[0] == "apg" else k[1]))
    for kind, lives in order:
        rs = groups[(kind, lives)]
        g = [r["gap"]["mean"] for r in rs]
        rng = f"{min(g):.3f} to {max(g):.3f}" if len(g) > 1 else "--"
        rows.append(f"{KIND_SHORT[kind]} & {million(lives)} & {len(rs)} & {np.mean(g):.3f} & {rng} & {mean(rs, lambda r: r['gap_choice']['mean']):.3f} & "
                    f"{mean(rs, lambda r: r['gap_consumption_mean']):.3f} & {100 * mean(rs, lambda r: r['same_share']):.1f} \\\\")
    rows.append("\\midrule")
    # by construction: the rule keeps the optimal annuity choice, and without annuity the whole gap comes from the annuity choice
    rows.append(f"Rule, optimal annuity & -- & -- & {rule['optimal_annuity']['mean']:.3f} & -- & 0.000 & {rule['optimal_annuity']['mean']:.3f} & 100.0 \\\\")
    # the beliefs of Table 6 on the same couples: their annuity choice followed by the optimal consumption
    bl, bdec, _ = _losses(["results_decision/base_limit.npz"])
    a_true, p_true = bdec["TRUE"]
    for key in ("MARG", "M0", "POP"):
        assert abs(bl[key].mean() - j["references"][key]["gap"]["mean"]) < 1e-9          # same couples, same losses as the evaluation of the agents
        a, p = bdec[key]
        same = np.mean(np.isclose(p, p_true)[a_true > 0])
        rows.append(f"Belief {key} & -- & -- & {bl[key].mean():.3f} & -- & {bl[key + '|choice'].mean():.3f} & {bl[key].mean() - bl[key + '|choice'].mean():.3f} & {100 * same:.1f} \\\\")
    gna = j["references"]["no_annuity"]["gap"]["mean"]
    rows.append(f"No annuity & -- & -- & {gna:.3f} & -- & {gna:.3f} & 0.000 & -- \\\\")
    table("agents", "Gap of the learned policies to the optimum, with exact expectations (held-out couples).", "tab:agents", "lccccccc",
          ["Policy & Lives & Seeds & \\multicolumn{2}{c}{Gap, \\% of resources} & \\multicolumn{2}{c}{Part due to} & Same $\\phi$ \\\\",
           "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}", " & millions & & mean & range & annuity & consumption & \\% \\\\"], rows,
          notes="Gap: wealth-equivalent loss with respect to the optimum of dynamic programming, mean over the held-out couples and over the seeds; range: "
                "smallest and largest mean across seeds. Part due to the annuity: loss of the annuity choice followed by the optimal consumption; the remainder is "
                "due to consumption. Same $\\phi$: share of couples for which the policy chooses the optimal survivor share. Gradient agent: gradient through the "
                "simulator. Rule, optimal annuity: the simple consumption rule alone, with the optimal annuity choice. Belief: household that decides with this "
                "belief (Table~\\ref{tab:decision}). No annuity: no annuity and the optimal consumption. --: not applicable.", size="\\footnotesize", colsep=3)
    put("rl.nstatic", 16)
    put("rl.ndyn", 8)
    put("rl.nmon", 300)
    put("rl.rule.optimal", f(rule["optimal_annuity"]["mean"], 2))
    put("rl.rule.allin", f(rule["all_in_two_thirds"]["mean"], 2))
    apg = max((k for k in groups if k[0] == "apg"), key=lambda k: k[1])
    rs = groups[apg]
    info = json.load(open("results_rl/apg_seed0.json"))
    put("rl.apg.lives", fi(info["lives_per_step"]))
    put("rl.apg.steps", fi(info["steps"]))
    put("rl.apg.total", f"{million(apg[1])} million")
    put("rl.apg.nseeds", len(rs))
    put("rl.apg.nseeds.word", word(len(rs)))
    g = [r["gap"]["mean"] for r in rs]
    put("rl.apg.gap.mean", f(np.mean(g)))
    put("rl.apg.gap.min", f(min(g)))
    put("rl.apg.gap.max", f(max(g)))
    put("rl.apg.gap.sd", f(np.std(g, ddof=1)) if len(g) > 1 else "--")
    put("rl.apg.choice.mean", f(mean(rs, lambda r: r["gap_choice"]["mean"])))
    put("rl.apg.cons.mean", f(mean(rs, lambda r: r["gap_consumption_mean"])))
    put("rl.apg.same.mean", pct(mean(rs, lambda r: r["same_share"])))
    put("rl.apg.gap.train", f(mean(rs, lambda r: r["gap_training_monitoring"])))
    put("rl.apg.q90.mean", f(mean(rs, lambda r: r["gap"]["q90"])))
    put("rl.apg.median.mean", f(mean(rs, lambda r: r["gap"]["median"])))
    put("rl.apg.max", f(max(r["gap"]["max"] for r in rs), 1))
    put("rl.apg.minutes", f(mean(rs, lambda r: r["seconds"]) / 60, 0))
    ppo_short = min(k[1] for k in groups if k[0].startswith("ppo"))
    at = [[c["gap_mean"] for c in r["curve"] if c["lives"] == ppo_short] for r in rs]
    assert all(len(a) == 1 for a in at), "the learning curves of the gradient agents have no point at the budget of PPO"
    put("rl.apg.gap.atppo", f(np.mean([a[0] for a in at])))
    for kind in ("ppoa", "ppo"):
        if (kind, ppo_short) in groups:
            rs2 = groups[(kind, ppo_short)]
            g = [r["gap"]["mean"] for r in rs2]
            put(f"rl.{kind}.nseeds", len(rs2))
            put(f"rl.{kind}.nseeds.word", word(len(rs2)))
            put(f"rl.{kind}.gap.mean", f(np.mean(g)))
            put(f"rl.{kind}.gap.min", f(min(g)))
            put(f"rl.{kind}.gap.max", f(max(g)))
            put(f"rl.{kind}.choice.mean", f(mean(rs2, lambda r: r["gap_choice"]["mean"])))
            put(f"rl.{kind}.minutes", f(mean(rs2, lambda r: r["seconds"]) / 60, 0))
    put("rl.ppo.total", f"{million(ppo_short)} million")
    infos = [json.load(open(p)) for p in sorted(glob.glob("results_rl/ppo*_seed*.json"))]
    assert len({i["lives_per_iteration"] for i in infos}) == 1 and len({i["width"] for i in infos} | {info["width"]}) == 1
    put("rl.ppo.lives", fi(infos[0]["lives_per_iteration"]))
    put("rl.ppo.iters", fi(ppo_short // infos[0]["lives_per_iteration"]))
    put("rl.width", info["width"])
    noise = sorted({i["noise"] for i in infos if i["noise"] != "learned"})
    assert noise == ["annealed from exp(-0.5) to exp(-3.0)"], noise
    long = sorted(k for k in groups if k[0] in ("ppoa", "ppo") and k[1] > ppo_short)
    assert len(long) <= 1
    if long:
        choice = json.load(open("results_rl/ppo_long_choice.json"))
        assert choice["chosen"] == long[0][0]
        put("rl.ppo.long.total", f"{million(long[0][1])} million")
        put("rl.long.gap", f(np.mean([r["gap"]["mean"] for r in groups[long[0]]])))
        put("rl.long.kind", "annealed" if long[0][0] == "ppoa" else "learned")
        put("rl.long.nseeds", len(groups[long[0]]))
        nl = len(groups[long[0]])
        put("rl.long.seedtext", "with one seed" if nl == 1 else f"on average over {word(nl)} seeds")
        put("rl.long.paren", "one run" if nl == 1 else f"{word(nl)} seeds")
        put("rl.long.runs", "a longer run uses" if nl == 1 else f"{word(nl)} longer runs use")
        put("rl.ratio.long.apg", f(float(V["rl.long.gap"]) / float(V["rl.apg.gap.mean"]), 0))
        for k, v in choice["monitoring_gap"].items():
            put(f"rl.{k}.monitor.gap", f(v))
    put("rl.ratio.ppoa.apg", f(float(V["rl.ppoa.gap.mean"]) / float(V["rl.apg.gap.mean"]), 0) if "rl.ppoa.gap.mean" in V else "--")
    rows = [f"Simulated lives per update & {V['rl.apg.lives']} & {V['rl.ppo.lives']} \\\\",
            f"Updates & {V['rl.apg.steps']} steps & {V['rl.ppo.iters']} iterations"
            + (f" ({fi(long[0][1] // infos[0]['lives_per_iteration'])} for the longer run{'s' if len(groups[long[0]]) > 1 else ''})" if long else "")
            + " of 4 epochs $\\times$ 8 minibatches \\\\",
            "Optimiser & Adam & Adam \\\\",
            "Learning rate & $10^{-3}$ to $10^{-5}$, cosine & $3\\times10^{-4}$ to 0, linear \\\\",
            "Gradient clipping (global norm) & 1 & 0.5 \\\\",
            f"Hidden layers & 2 $\\times$ {V['rl.width']}, tanh & 2 $\\times$ {V['rl.width']}, tanh (policy and value networks) \\\\",
            "Discrete survivor share & regression of the advantage of each share & categorical policy \\\\",
            "Clipping of the surrogate objective & -- & 0.2 \\\\",
            "Advantage estimation & -- & GAE, $\\lambda=0.95$, discount factor $1/(1+r)$ \\\\",
            "Entropy coefficient & -- & $10^{-3}$ \\\\",
            "Exploration noise (log standard deviation) & none & learned from $-0.5$, or annealed from $-0.5$ to $-3$ \\\\",
            "Reward used for learning & scaled utility & scaled utility divided by 5, floored at $-50$ \\\\"]
    table("hyper", "Settings of the two learning algorithms.", "tab:hyper", "p{4.3cm}p{4.3cm}p{5.4cm}",
          ["Setting & Gradient through the simulator & PPO \\\\"], rows,
          notes="GAE: generalised advantage estimation; $\\lambda$: its smoothing parameter. $r$: interest rate of Section~\\ref{sec:decision}. --: not applicable.",
          size="\\footnotesize", placement="h")


# ----------------------------------------------------------------------------- explanations
G_FR = ["Âge du mari", "Âge de la femme", "Santé du mari", "Santé de la femme", "Études et patrimoine"]
G_EN = ["Age of the husband", "Age of the wife", "Health of the husband", "Health of the wife", "Education and wealth"]
G_TAB = ["Age, husband", "Age, wife", "Health, husband", "Health, wife", "Education, wealth"]
G_KEY = ["ah", "aw", "hh", "hw", "ses"]
C_EN = {"age_h": "Age of the husband", "age_w": "Age of the wife", "bmi_h": "BMI of the husband", "bmi_w": "BMI of the wife", "smoker_h": "Husband smokes",
        "smoker_w": "Wife smokes", "hyp_h": "Hypertension, husband", "hyp_w": "Hypertension, wife", "diab_h": "Diabetes, husband", "diab_w": "Diabetes, wife",
        "educ_h": "Education of the husband", "educ_w": "Education of the wife", "wealth": "Household wealth"}
C_CHANGE = {"age_h": "+5 years", "age_w": "+5 years", "bmi_h": "+5 kg/m$^2$", "bmi_w": "+5 kg/m$^2$", "smoker_h": "no to yes", "smoker_w": "no to yes",
            "hyp_h": "no to yes", "hyp_w": "no to yes", "diab_h": "no to yes", "diab_w": "no to yes", "educ_h": "+4 years", "educ_w": "+4 years", "wealth": "$\\times$ 10"}
REF_AGENT = "apg_seed0"
ci = lambda r, d=2: f"{r['mean']:.{d}f} [{r['ci'][0]:.{d}f}; {r['ci'][1]:.{d}f}]"
mm = lambda x, d=2: f"{x:.{d}f}".replace("-", "$-$")


@part
def xai():
    j = json.load(open("results_xai/xai_shap.json"))
    apg = [a for a in j["agents"] if a.startswith("apg")]
    ppo = [a for a in j["agents"] if a.startswith("ppo")]
    if json.load(open("results_rl/apg_seed0.json"))["steps"] != 12000 or len(apg) < 5:
        raise FileNotFoundError(2, "missing", "results_xai/xai_shap.json computed with the five final gradient agents")
    put("xai.pairs.all", fi(j["pairs_A"]))
    # without the shortcuts: every ordered pair of distinct couples, 2^13 coalitions each
    put("xai.evals.full", fi(2 * j["pairs_A"] * 2 ** 13 / 1e6))
    put("xai.evals.done", fi(j["A_coalitions_evaluated"] / 1e6))
    put("xai.pairs", j["pairs"])
    put("xai.dp.solved", fi(len(np.load("results_xai/hybrids.npz")["Z"]) + len(np.load("results_xai/effects.npz")["Z"])))
    hz, ez = np.load("results_xai/hybrids.npz"), np.load("results_xai/effects.npz")
    used = np.vstack([hz["Z"][np.unique(hz["row"][:, :32])], ez["Z"]])                # the 32 group hybrids of each pair and the stated changes
    put("xai.dp.used", fi(len(np.unique(np.round(used, 10), axis=0))))
    put("xai.nseeds", len(apg))
    rows = []
    for dec in ("alpha", "phi"):
        B = j["B"][dec]
        ex, ag = B["exact"], B[REF_AGENT]
        for g, key in zip(G_FR, G_KEY):
            put(f"xai.B.{dec}.exact.share.{key}", pct(ex["share"][g], 0))
            put(f"xai.B.{dec}.agent.share.{key}", pct(ag["share"][g], 0))
            put(f"xai.B.{dec}.exact.{key}", f(ex["mean_abs"][g]["mean"], 2))
            put(f"xai.B.{dec}.agent.{key}", f(ag["mean_abs"][g]["mean"], 2))
            put(f"xai.B.{dec}.corr.{key}", f(ag["corr"][g]["value"], 2))
        put(f"xai.B.{dec}.corr", f(ag["corr_all"]["value"], 2))
        put(f"xai.B.{dec}.corr.lo", f(ag["corr_all"]["ci"][0], 2))
        put(f"xai.B.{dec}.corr.hi", f(ag["corr_all"]["ci"][1], 2))
        cs = [B[a]["corr_all"]["value"] for a in apg]
        put(f"xai.B.{dec}.corr.min", f(min(cs), 2))
        put(f"xai.B.{dec}.corr.max", f(max(cs), 2))
        lo_, hi_ = f(min(cs), 2), f(max(cs), 2)
        put(f"xai.B.{dec}.corr.seedtext", f"is {lo_} for each of the {word(len(apg))} seeds" if lo_ == hi_ else f"lies between {lo_} and {hi_} across the {word(len(apg))} seeds")
        for a in ppo:
            put(f"xai.B.{dec}.corr.agent.{a.split('_')[0]}", f(B[a]["corr_all"]["value"], 2))
        if len(ppo) == 2:
            a_, b_ = (f(B[a]["corr_all"]["value"], 2) for a in sorted(ppo, key=lambda a: a.split("_")[0] != "ppoa"))
            put(f"xai.B.{dec}.corr.ppo.pair", f"{a_} in both cases" if a_ == b_ else f"{a_} and {b_}")
        if ppo:
            cs = [B[a]["corr_all"]["value"] for a in ppo]
            put(f"xai.B.{dec}.corr.ppo", f(cs[0], 2) if len(cs) == 1 or f(min(cs), 2) == f(max(cs), 2) else f"{min(cs):.2f} to {max(cs):.2f}")
    for g, name in zip(G_FR, G_TAB):
        a, p = j["B"]["alpha"], j["B"]["phi"]
        rows.append(f"{name} & {ci(a['exact']['mean_abs'][g])} & {ci(a[REF_AGENT]['mean_abs'][g])} & {a[REF_AGENT]['corr'][g]['value']:.2f} & "
                    f"{ci(p['exact']['mean_abs'][g])} & {ci(p[REF_AGENT]['mean_abs'][g])} & {p[REF_AGENT]['corr'][g]['value']:.2f} \\\\")
    tot = {(dec, who): sum(j["B"][dec][who]["mean_abs"][g]["mean"] for g in G_FR) for dec in ("alpha", "phi") for who in ("exact", REF_AGENT)}
    rows += ["\\midrule", f"All groups & {tot['alpha', 'exact']:.2f} & {tot['alpha', REF_AGENT]:.2f} & {j['B']['alpha'][REF_AGENT]['corr_all']['value']:.2f} & "
                           f"{tot['phi', 'exact']:.2f} & {tot['phi', REF_AGENT]:.2f} & {j['B']['phi'][REF_AGENT]['corr_all']['value']:.2f} \\\\"]
    table("groups", "Weights of five groups of characteristics: exact policy and gradient agent.", "tab:groups", "lcccccc",
          ["Group & \\multicolumn{3}{c}{Annuitised share} & \\multicolumn{3}{c}{Survivor share} \\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           " & exact & agent & corr. & exact & agent & corr. \\\\"], rows,
          notes=f"Weight: mean absolute Shapley contribution of the group to the difference of decisions between two couples, in percentage points, over "
                f"{j['pairs']} pairs of held-out couples [95\\% bootstrap interval]. agent: gradient agent, reference seed. corr.: correlation between the "
                "contribution according to the agent and the exact contribution. All groups: sum of the weights; correlation over the pairs and the five groups together.",
          size="\\footnotesize")
    # all-pairs SHAP weights of the 13 characteristics (gradient agents)
    for dec in ("alpha", "phi"):
        w = j["A"][REF_AGENT][dec]["mean_abs"]
        top = sorted(w, key=w.get, reverse=True)
        put(f"xai.A.{dec}.top.weight", f(w[top[0]], 2))
        for k in range(4):
            put(f"xai.A.{dec}.top{k + 1}", C_EN[top[k]].lower().replace("bmi", "BMI"))
        put(f"xai.A.{dec}.seedcorr.min", f(min(v[dec]["corr_values"] for v in j["A_between_agents"].values()), 3))
        put(f"xai.A.{dec}.sd", f(j["A"][REF_AGENT]["decision_sd"][0 if dec == "alpha" else 1], 1))
    # stated changes
    D = j["D"]
    sign = np.load("results_xai/effects.npz")["sign"]                 # -1: the characteristic was lowered (raising it would leave the observed range)
    RAWK = ["age_h", "age_w", "bmi_h", "bmi_w", "smoker_h", "smoker_w", "hyp_h", "hyp_w", "diab_h", "diab_w", "educ_h", "educ_w", "wealth"]
    lowered = {c: float((sign[:, k] < 0).mean()) for k, c in enumerate(RAWK)}
    assert lowered["bmi_h"] == 0 and lowered["bmi_w"] == 0, lowered
    agemax = max(lowered["age_h"], lowered["age_w"])
    low = {"educ_h": pct(lowered["educ_h"], 0), "educ_w": pct(lowered["educ_w"], 0), "wealth": pct(lowered["wealth"], 0),
           "age": "below 1\\%" if agemax < 0.01 else pct(agemax, 0) + "\\% at most"}
    rows, rows_phi = [], []
    for c in j["characteristics"]:
        a, p = D["alpha"][c], D["phi"][c]
        rng = lambda d: f"{mm(min(d[x]['mean'] for x in apg))} to {mm(max(d[x]['mean'] for x in apg))}"
        cr = [a[x]["corr"] for x in apg]
        rows.append(f"{C_EN[c]} & {C_CHANGE[c]} & {mm(a['exact']['mean'])} [{mm(a['exact']['ci'][0])}; {mm(a['exact']['ci'][1])}] & "
                    f"{mm(a[REF_AGENT]['mean'])} [{mm(a[REF_AGENT]['ci'][0])}; {mm(a[REF_AGENT]['ci'][1])}] & {rng(a)} & {min(cr):.2f} to {max(cr):.2f} \\\\")
        rows_phi.append(f"{C_EN[c]} & {C_CHANGE[c]} & {mm(p['exact']['mean'])} [{mm(p['exact']['ci'][0])}; {mm(p['exact']['ci'][1])}] & {pct(p['exact']['share_nonzero'], 0)} & "
                        f"{mm(p[REF_AGENT]['mean'])} [{mm(p[REF_AGENT]['ci'][0])}; {mm(p[REF_AGENT]['ci'][1])}] & {rng(p)} \\\\")
        key = c.replace("_", "")
        put(f"xai.D.alpha.exact.{key}", f(abs(a["exact"]["mean"]), 2))
        put(f"xai.D.alpha.agent.{key}", f(abs(a[REF_AGENT]["mean"]), 2))
        put(f"xai.D.phi.exact.{key}", f(abs(p["exact"]["mean"]), 2))
        put(f"xai.D.phi.exact.{key}.sign", "raises" if p["exact"]["mean"] > 0 else "lowers")
        put(f"xai.D.phi.changed.{key}", pct(p["exact"]["share_nonzero"], 0))
    put("xai.D.alpha.maxdiff", f(max(abs(D["alpha"][c][REF_AGENT]["mean"] - D["alpha"][c]["exact"]["mean"]) for c in j["characteristics"]), 2))
    put("xai.D.alpha.corr.min", f(min(D["alpha"][c][REF_AGENT]["corr"] for c in j["characteristics"]), 2))
    put("xai.D.alpha.corr.minchar", C_EN[min(j["characteristics"], key=lambda c: D["alpha"][c][REF_AGENT]["corr"])].lower().replace("bmi", "BMI"))
    chronic = [abs(D["alpha"][c]["exact"]["mean"]) for c in ("hyp_h", "hyp_w", "diab_h", "diab_w")]
    put("xai.D.alpha.exact.minchronic", f(min(chronic), 2))
    put("xai.D.alpha.exact.maxchronic", f(max(chronic), 2))
    table("effects", "Effect of a stated change of one characteristic on the annuitised share (percentage points).", "tab:effects", "llcccc",
          ["Characteristic & Change & Exact policy & Agent & Agent, range & Corr., range \\\\"], rows,
          notes=f"{D['couples']} held-out couples; mean change of the annuitised share when one characteristic is changed and the others are kept [95\\% bootstrap interval]. "
                "Yes/no characteristics are switched to the other state and the effect is expressed for the change from no to yes. The other characteristics are raised "
                f"by the stated amount, or lowered by it when raising would leave the observed range (education of the husband: {low['educ_h']}\\%, of the wife: {low['educ_w']}\\% of the couples; "
                f"wealth: {low['wealth']}\\%; ages: {low['age']}), and the effect is expressed for the increase. For wealth the change is $\\ln 10$ on the "
                "inverse-hyperbolic-sine scale, a tenfold increase for large positive wealth. Agent: reference seed; ranges: over the "
                f"{word(len(apg))} seeds. Corr.: correlation across couples between the effect according to the agent and the exact effect.", size="\\scriptsize", colsep=3)
    table("effects_phi", "Effect of a stated change of one characteristic on the survivor share (percentage points).", "tab:effects.phi", "llcccc",
          ["Characteristic & Change & Exact policy & Changed, \\% & Agent & Agent, range \\\\"], rows_phi,
          notes=f"As Table~\\ref{{tab:effects}}, for the survivor share and without the correlation column. Changed: share of the couples whose exact "
                "survivor share changes.", size="\\footnotesize", placement="h", colsep=3)
    G = j["gap_hybrids"]
    put("xai.gap.hybrid", f(G[REF_AGENT]["hybrid"], 2))
    put("xai.gap.real", f(G[REF_AGENT]["real"], 2))


@part
def trees():
    t = json.load(open("results_xai/xai_tree.json"))
    if REF_AGENT not in t["teachers"] or len([a for a in t["teachers"] if a.startswith("apg")]) < 5:
        raise FileNotFoundError(2, "missing", "results_xai/xai_tree.json computed with the five final gradient agents")
    T, direct = t["teachers"][REF_AGENT], t["direct"]
    rows = []
    for d in map(str, t["depths"]):
        rows.append(f"{d} & {T[d]['adjusted']['leaves']} & {T[d]['imitation']['loss']['mean']:.3f} & {T[d]['adjusted']['loss']['mean']:.3f} & "
                    f"{direct[d]['leaves']} & {direct[d]['loss']['mean']:.3f} & {pct0(T[d]['imitation'].get('fidelity_r2_alpha', float('nan')))} & "
                    f"{100 * T[d]['imitation'].get('fidelity_same_phi', float('nan')):.0f} \\\\".replace("nan", "--"))
    R = t["references"]
    rows += ["\\midrule", f"\\multicolumn{{8}}{{l}}{{Annuity choice of the gradient agent (reference seed): {R[REF_AGENT]['mean']:.3f}; of the belief M0: {R['M0']['mean']:.3f}; "
                          f"of the belief POP: {R['POP']['mean']:.3f}}} \\\\"]
    table("trees", "Loss of short decision trees, with exact expectations (\\% of resources, held-out couples).", "tab:trees", "cccccccc",
          ["Depth & \\multicolumn{3}{c}{Tree extracted from the agent} & \\multicolumn{2}{c}{Direct tree} & \\multicolumn{2}{c}{Fidelity, \\%} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}", " & leaves & imitation & adjusted & leaves & loss & $R^2$ of $\\alpha$ & same $\\phi$ \\\\"], rows,
          notes="Loss of the annuity choice of the tree followed by the optimal consumption. Trees are learned on the estimation couples. Imitation: leaves take the "
                "mean decision of the agent; adjusted: leaves take the best action of a grid. Fidelity: agreement of the imitation tree with the agent on the held-out "
                "couples ($R^2$ of the annuitised share; share of couples with the same survivor share). "
                "Depth 0: one action for all couples. Agent: gradient agent, reference seed.", size="\\footnotesize")
    put("xai.tree.actions", t["grid_actions"])
    put("xai.tree.d0", f(direct["0"]["loss"]["mean"], 2))
    put("xai.tree.d3.leaves", word(T["3"]["adjusted"]["leaves"]))
    put("xai.tree.d3.adjusted", f(T["3"]["adjusted"]["loss"]["mean"], 2))
    put("xai.tree.d3.imitation", f(T["3"]["imitation"]["loss"]["mean"], 2))
    put("xai.tree.d3.direct", f(direct["3"]["loss"]["mean"], 2))
    put("xai.tree.d6.best", f(min(T["6"]["adjusted"]["loss"]["mean"], direct["6"]["loss"]["mean"]), 2))
    put("xai.tree.d6.leaves", direct["6"]["leaves"])
    put("xai.tree.ref.agent", f(R[REF_AGENT]["mean"], 2))
    put("xai.tree.ref.POP", f(R["POP"]["mean"], 2))
    put("xai.tree.recovered", pct(1 - T["3"]["adjusted"]["loss"]["mean"] / R["POP"]["mean"], 0))
    put("xai.tree.ratio.agent", f(T["3"]["adjusted"]["loss"]["mean"] / R[REF_AGENT]["mean"], 0))
    noun = {"age_h": "the age of the husband", "age_w": "the age of the wife", "bmi_h": "the BMI of the husband", "bmi_w": "the BMI of the wife",
            "smoker_h": "the smoking status of the husband", "smoker_w": "the smoking status of the wife", "hyp_h": "the hypertension of the husband",
            "hyp_w": "the hypertension of the wife", "diab_h": "the diabetes of the husband", "diab_w": "the diabetes of the wife",
            "educ_h": "the education of the husband", "educ_w": "the education of the wife", "wealth": "household wealth", "age_gap": "the age difference"}
    root = T["3"]["adjusted"]["tree"]
    first = list(dict.fromkeys([root["feature"]] + [n["feature"] for n in (root["left"], root["right"]) if not n["leaf"]]))
    put("xai.tree.first", " and ".join([", ".join(noun[c] for c in first[:-1]), noun[first[-1]]]) if len(first) > 1 else noun[first[0]])


# ----------------------------------------------------------------------------- numerical checks quoted in the appendix
@part
def checks():
    grab = lambda path, pattern, k=1: re.search(pattern, open(path).read()).group(k)
    put("chk.lik.z", grab("results/check_likelihood.txt", r"largest \|closed - MC\| in Monte Carlo standard errors: ([0-9.]+)"))
    # largest standardised difference over all urns (check_urn_conjugacy.py reports the urns with and without a death ball apart)
    urn = re.findall(r"max \|posterior mean - urn mean\| / urn sd\s*: ([0-9.]+)", open("results/check_urn_conjugacy_jax0.10.2.txt").read())
    put("chk.urn.mean", f(max(float(x) for x in urn)))
    z = open("results_decision/check_law.txt").read()
    m = re.search(r"(\d+) probabilities compared with ([0-9,]+) simulated couples each; largest \|z\| ([0-9.]+)", z)
    put("chk.law.nprob", m.group(1))
    put("chk.law.nsim", m.group(2).replace(",", "{,}"))
    put("chk.law.z", m.group(3))
    z = open("results_decision/check_decision.txt").read()
    put("chk.dp.closed", "$" + re.sub(r"e-0?(\d+)", r"\\times10^{-\1}", re.search(r"closed form without income: largest relative gap ([0-9.e-]+)", z).group(1)) + "$")
    put("chk.dp.gold", f"{abs(float(re.search(r'gap between (-?[0-9.e+-]+) %', z).group(1))):.4f}")
    z = open("results_rl/check_rl.txt").read()
    m = re.search(r"simulated lives against the exact law \(([0-9,]+) lives per couple\)", z)
    put("chk.rl.nsim", m.group(1).replace(",", "{,}"))
    zs = [float(x) for x in re.findall(r"probabilities, largest \|z\| ([0-9.]+)", z)]
    put("chk.rl.nprob", sum(int(x) for x in re.findall(r": (\d+) probabilities, largest", z)))
    put("chk.rl.z", f(max(zs), 2))
    put("chk.rl.zobj", f(max(abs(float(x)) for x in re.findall(r"standard error [0-9.]+, z ([+-][0-9.]+)\)", z)), 2))
    # independent binomial regression (no frailty) against the Bayesian fit without socio-economic covariates
    g = open("results_real/check_glm.txt").read()
    glm = {}
    for block, sex in (("husbands", "mari"), ("wives", "femme")):
        part = g.split(block, 1)[1].split("wives" if block == "husbands" else "\n\n", 1)[0]
        for name, hr in re.findall(r"^\s+(\w+)\s+HR ([0-9.]+)", part, flags=re.M):
            glm[(sex, name)] = float(hr)
    bayes = {r["parameter"]: r["hazard_ratio"] for r in json.load(open("results_real/posterior_summary.json"))["effects"]}
    label = {"bmi": "BMI (+1 écart-type)", "smoker": "fumeur", "hyp": "hypertension", "diab": "diabète", "p_bmi": "BMI du conjoint (+1 écart-type)",
             "p_smoker": "conjoint fumeur"}
    diffs = [abs(glm[(sx, n)] - bayes[f"{sx} : {lab}"]) for sx in ("mari", "femme") for n, lab in label.items()]
    put("chk.glm.maxdiff", f(max(diffs), 2))
    put("chk.glm.ncov", len(diffs))
    z = open("results_xai/check_xai.txt").read()
    sci = lambda x: "$" + re.sub(r"e-0?(\d+)", r"\\times10^{-\1}", x) + "$"
    put("chk.xai.shap", sci(re.search(r"écart maximal avec notre calcul sur les mêmes couples ([0-9.e-]+)", z).group(1)))
    put("chk.xai.short", sci(re.search(r"contre énumération des 8 192 coalitions : écart maximal ([0-9.e-]+)", z).group(1)))
    put("chk.xai.eff", sci(max(re.findall(r"somme des valeurs SHAP = décision du couple - décision moyenne des couples, écart maximal ([0-9.e-]+)", z), key=float)))


# ----------------------------------------------------------------------------- figures
COL = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F", "black": "#000000"}


FIG_W = 119 / 25.4                         # final figure width in inches (119 mm)


def _plain_log(axis, ticks=None):
    """Log axis labelled with plain decimals (0.001, 0.01, ...) instead of powers of ten."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
    axis.set_major_locator(FixedLocator(ticks) if ticks is not None else LogLocator(base=10))
    axis.set_major_formatter(FuncFormatter(lambda v, _: np.format_float_positional(v, trim="-")))
    axis.set_minor_formatter(NullFormatter())


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # journal artwork rules: Helvetica or Arial lettering of 8 to 12 pt at the final size; figures are drawn at their final
    # width, FIG_W (119 mm, the width of small-format Springer journals), and included at that width without scaling
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "lines.linewidth": 1.2, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": None, "pdf.fonttype": 42,
                         "font.family": "Liberation Sans", "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                         "mathtext.it": "Liberation Sans:italic"})
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    return plt


@part
def fig_baseline():
    plt = _plt()
    parts = [np.load(p) for p in sorted(glob.glob("results_eval/M3_full_*.npz"))]
    if not parts:
        raise FileNotFoundError(2, "missing", "results_eval/M3_full_*.npz")
    p0 = np.concatenate([p["p0"].reshape((-1,) + p["p0"].shape[2:]) for p in parts], axis=0)          # (draws, 2, ages)
    ages, pstar = parts[0]["ages"], parts[0]["pstar"]
    fig, ax = plt.subplots(1, 2, figsize=(FIG_W, 2.5), sharey=True, layout="constrained")
    for s, (name, col) in enumerate((("Husbands", COL["blue"]), ("Wives", COL["red"]))):
        q = np.percentile(p0[:, s], [2.5, 50, 97.5], axis=0)
        ax[s].fill_between(ages, q[0], q[2], color=col, alpha=0.25, lw=0)
        ax[s].plot(ages, q[1], color=col, label="posterior")
        ax[s].plot(ages, pstar[s], color=COL["black"], ls="--", lw=0.9, label="prior guess")
        ax[s].set_yscale("log")
        ax[s].set_ylim(2e-5, 1.0)
        _plain_log(ax[s].yaxis)
        ax[s].set_xlabel("Age, years")
        ax[s].set_title(name)
        ax[s].legend(loc="upper left")
    ax[0].set_ylabel("One-year death probability")
    fig.savefig(f"{OUT}/figures/baseline.pdf")
    plt.close(fig)
    put("model.agemin", int(ages[0]))
    put("model.agemax", int(ages[-1]))
    put("model.nurns", 2 * len(ages))
    c = parts[0]["c"]
    put("prior.c.max", fi(c.max()))
    from pipeline import load_couples, split_indices, fit_scaler, make_data
    from dyadic_mortality import build_arrays
    d = load_couples("cohort_w10.csv")
    data = make_data(d, fit_scaler(d))
    arr = build_arrays(data, int(ages[0]), int(ages[-1]))
    risk = np.zeros((2, len(ages)))
    for s in range(2):
        np.add.at(risk[s], arr["age_idx"][:, s, :][(arr["surv_mask"][:, s, :] + arr["event_mask"][:, s, :]) > 0], 1.0)
    band = (ages >= 55) & (ages <= 85)
    put("data.py.min", fi(risk[:, band].min()))
    put("data.py.max", fi(risk[:, band].max()))
    put("prior.c.band.max", fi(c[:, band].max()))
    # weight of the prior in an urn: pseudo-observations over pseudo-observations plus person-years
    w = c / (c + risk)
    share = lambda m: 100 * c[:, m].sum() / (c[:, m] + risk[:, m]).sum()
    put("prior.w.6089.max", f(100 * w[:, (ages >= 60) & (ages <= 89)].max(), 0))
    put("prior.w.5054", f(share((ages >= 50) & (ages <= 54)), 0))
    put("prior.w.5559", f(share((ages >= 55) & (ages <= 59)), 0))
    put("prior.w.9094", f(share((ages >= 90) & (ages <= 94)), 0))
    put("prior.w.9599", f(share((ages >= 95) & (ages <= 99)), 0))
    put("prior.w.100", f(share(ages >= 100), 0))
    # observed deaths of the cohort against the number expected from the prior guess (all ages, by sex)
    dth = np.zeros((2, len(ages)))
    for s in range(2):
        np.add.at(dth[s], arr["age_idx"][:, s, :][arr["event_mask"][:, s, :] > 0], 1.0)
    oe = dth.sum(axis=1) / (risk * pstar).sum(axis=1)
    put("prior.oe.h", f(oe[0], 2))
    put("prior.oe.w", f(oe[1], 2))


@part
def fig_losses():
    plt = _plt()
    from decision_setup import law
    from decision_dp import Params, annuity_due
    par = Params()
    loss, _, n = _losses(["results_decision/base_limit.npz"])
    bt, ot = law("TRUE", 110)
    bp, op = law("POP", 110)
    ratio = annuity_due(bt, ot, 2.0 / 3.0, par.r) / annuity_due(bp, op, 2.0 / 3.0, par.r)
    edges = np.quantile(ratio, np.linspace(0, 1, 11))
    grp = np.clip(np.digitize(ratio, edges[1:-1]), 0, 9)
    x = np.array([ratio[grp == g].mean() for g in range(10)])
    fig, ax = plt.subplots(figsize=(FIG_W, 3.0), layout="constrained")
    ax.scatter(ratio, np.maximum(loss["POP"], 1e-3), s=3, color=COL["grey"], alpha=0.25, lw=0, rasterized=True)
    for m, col, mk in (("POP", COL["red"], "o"), ("MD", COL["orange"], "D"), ("M0", COL["blue"], "s"), ("MARG", COL["green"], "^")):
        ax.plot(x, [loss[m][grp == g].mean() for g in range(10)], color=col, marker=mk, ms=3.5, label=m)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 30)
    _plain_log(ax.yaxis)
    ax.set_xlabel("Joint annuity value: truth / pricing table")
    ax.set_ylabel("Loss, % of resources")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.6)
    fig.savefig(f"{OUT}/figures/losses.pdf", dpi=300)
    plt.close(fig)
    put("health.ratio.d1", f(x[0], 2))
    put("health.ratio.d10", f(x[-1], 2))
    put("health.ratio.min", f(ratio.min(), 2))
    put("health.ratio.max", f(ratio.max(), 2))
    for m in ("POP", "MD", "M0", "MARG"):
        put(f"loss.base.{m}.d1", f(loss[m][grp == 0].mean(), 2))
        put(f"loss.base.{m}.d10", f(loss[m][grp == 9].mean(), 2))
        put(f"loss.base.{m}.dmin", f(min(loss[m][grp == g].mean() for g in range(10)), 2))


@part
def fig_curves():
    plt = _plt()
    files = sorted(glob.glob("results_rl/apg_seed*.json")) + sorted(glob.glob("results_rl/ppoa_seed*.json")) + sorted(glob.glob("results_rl/ppo_seed*.json"))
    kinds = {os.path.basename(p).split("_")[0] for p in files}
    if not {"apg", "ppoa"} <= kinds or len(glob.glob("results_rl/apg_seed*.json")) < 5:
        raise FileNotFoundError(2, "missing", "results_rl/*.json of the final agents (five gradient seeds and PPO)")
    style = {"apg": (COL["blue"], "-", "Gradient through the simulator"), "ppoa": (COL["red"], "-", "PPO, annealed noise"),
             "ppoa_long": (COL["red"], ":", "PPO, annealed noise, longer runs"), "ppo": (COL["green"], "--", "PPO, learned noise")}
    fig, ax = plt.subplots(figsize=(FIG_W, 3.0), layout="constrained")
    seen = set()
    short = min(json.load(open(p))["curve"][-1]["lives"] for p in files if os.path.basename(p).startswith("ppo"))
    for p in files:
        kind = os.path.basename(p).split("_")[0]
        c = json.load(open(p))["curve"]
        if kind == "ppoa" and c[-1]["lives"] > short:
            kind = "ppoa_long"
        col, ls, label = style[kind]
        x = np.array([k["lives"] for k in c][1:]) / 1e6
        y = np.array([k["gap_mean"] for k in c][1:])
        ax.plot(x, y, color=col, ls=ls, lw=0.9, alpha=0.85, label=None if kind in seen else label)
        seen.add(kind)
    ax.set_yscale("log")
    _plain_log(ax.yaxis, [0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    ax.set_xlabel("Simulated lives (millions)")
    ax.set_ylabel("Gap to the optimum, % of resources")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.24))           # in the empty band between the two groups of curves
    fig.savefig(f"{OUT}/figures/learning_curves.pdf")
    plt.close(fig)


@part
def fig_shap():
    plt = _plt()
    j = json.load(open("results_xai/xai_shap.json"))
    apg = [a for a in j["agents"] if a.startswith("apg")]
    if json.load(open("results_rl/apg_seed0.json"))["steps"] != 12000 or len(apg) < 5:
        raise FileNotFoundError(2, "missing", "results_xai/xai_shap.json computed with the five final gradient agents")
    fig, ax = plt.subplots(2, 1, figsize=(FIG_W, 5.4), layout="constrained")
    for k, (dec, title) in enumerate((("alpha", "Annuitised share"), ("phi", "Survivor share"))):
        w = j["A"][REF_AGENT][dec]["mean_abs"]
        order = sorted(w, key=w.get)
        lo = np.array([min(j["A"][a][dec]["mean_abs"][c] for a in apg) for c in order])
        hi = np.array([max(j["A"][a][dec]["mean_abs"][c] for a in apg) for c in order])
        val = np.array([w[c] for c in order])
        y = np.arange(len(order))
        ax[k].barh(y, val, color=COL["blue"], height=0.62)
        ax[k].hlines(y, lo, hi, color=COL["black"], lw=1.0)
        ax[k].set_yticks(y)
        ax[k].set_yticklabels([C_EN[c] for c in order])
        ax[k].set_xlabel("Percentage points")
        ax[k].set_title(title)
    fig.savefig(f"{OUT}/figures/shap_weights.pdf")
    plt.close(fig)


@part
def fig_tree():
    plt = _plt()
    t = json.load(open("results_xai/xai_tree.json"))
    if REF_AGENT not in t["teachers"] or len([a for a in t["teachers"] if a.startswith("apg")]) < 5:
        raise FileNotFoundError(2, "missing", "results_xai/xai_tree.json computed with the five final gradient agents")
    tree = t["teachers"][REF_AGENT]["3"]["adjusted"]["tree"]
    short = {"age_h": "age of the husband", "age_w": "age of the wife", "bmi_h": "BMI of the husband", "bmi_w": "BMI of the wife", "smoker_h": "husband smokes",
             "smoker_w": "wife smokes", "hyp_h": "husband has hypertension", "hyp_w": "wife has hypertension", "diab_h": "husband has diabetes",
             "diab_w": "wife has diabetes", "educ_h": "education of the husband", "educ_w": "education of the wife", "wealth": "wealth", "age_gap": "age difference"}
    binary = {"smoker_h", "smoker_w", "hyp_h", "hyp_w", "diab_h", "diab_w"}

    def question(node):
        c, th = node["feature"], node["threshold"]
        if c in binary:
            return f"{short[c]}?", "no", "yes"
        if c == "wealth":
            return f"wealth > {th / 1000:,.0f}k USD?", "no", "yes"
        return f"{short[c]} above {th:.1f}?", "no", "yes"

    import textwrap
    leaves_ = []

    # horizontal layout: the root on the left, one column per depth, the leaves stacked on the right
    def place(node, depth):
        if node["leaf"]:
            node["y"], node["d"] = -len(leaves_), depth + 0.3             # the leaves a little further right: room for the labels of the last edges
            leaves_.append(node)
        else:
            place(node["left"], depth + 1)
            place(node["right"], depth + 1)
            node["y"], node["d"] = 0.5 * (node["left"]["y"] + node["right"]["y"]), depth
    place(tree, 0)
    depth_max = max(n["d"] for n in leaves_)
    fig, ax = plt.subplots(figsize=(FIG_W, 4.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    box = dict(boxstyle="round,pad=0.3", fc="white", ec=COL["grey"], lw=0.6)

    def draw(node):
        x, y = node["d"], node["y"]
        if node["leaf"]:
            ax.text(x, y, f"$\\alpha$ = {100 * node['alpha']:.0f}%, $\\phi$ = {100 * node['phi']:.0f}%\nn = {node['n_test']}", ha="center",
                    va="center", bbox=dict(boxstyle="round,pad=0.3", fc="#EAF2FA", ec=COL["blue"], lw=0.6))
            return
        q, l, r = question(node)
        for child, lab in ((node["left"], l), (node["right"], r)):
            ax.plot([x, child["d"]], [y, child["y"]], color=COL["grey"], lw=0.7, zorder=0)
            fr = 0.45 if child["leaf"] else 0.42                    # short last edges: in the gap between the two boxes
            ax.text(x + fr * (child["d"] - x), y + fr * (child["y"] - y), lab, ha="center", va="center", color=COL["grey"],
                    bbox=dict(fc="white", ec="none", pad=0.6))
            draw(child)
        ax.text(x, y, textwrap.fill(q, 15), ha="center", va="center", bbox=box)
    draw(tree)
    ax.set_xlim(-0.55, depth_max + 0.55)
    ax.set_ylim(-len(leaves_) + 0.45, 0.55)
    ax.axis("off")
    fig.savefig(f"{OUT}/figures/tree.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
PARTS = [cohort, estimates, cap9, diagnostics, simulation, prior, oos, splits, annuity, decision, sensitivity, agents, xai, trees, checks,
         fig_baseline, fig_losses, fig_curves, fig_shap, fig_tree]

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    wanted = sys.argv[1:]
    for p in PARTS:
        if not wanted or p.__name__ in wanted:
            p()
    if wanted and os.path.exists(f"{OUT}/numbers.json"):                          # partial run: keep the other parts' numbers
        old = json.load(open(f"{OUT}/numbers.json"))
        old.update(V)
        V = old
    json.dump(V, open(f"{OUT}/numbers.json", "w"), indent=0, sort_keys=True)
    with open(f"{OUT}/numbers.tex", "w") as out:
        for k in sorted(V):
            out.write(f"\\expandafter\\def\\csname val@{k}\\endcsname{{{V[k]}}}\n")
    print(f"{len(V)} numbers written; parts skipped: {SKIPPED or 'none'}")
