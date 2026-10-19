"""
build_prior.py -- prior guess for the urns, from the mortality experience of HRS before 2010.

Population : married respondents at Wave 4 (INW4 = 1, R4MSTAT = 1), 1998.
Follow-up  : Waves 5-10 (2000-2010), with the status codes and the dates of build_cohort.py (RwIWSTAT
             codes 1, 4, 5, 6, 7; death month and year from RADMONTH / RADYEAR; discrete time in
             completed years since the Wave 4 interview month).
             Administrative end of follow-up, as in build_cohort.py: every person is censored after
             MAX_YEARS completed years and a later death is not used. MAX_YEARS is the largest number
             of years reached by at least 95 % of the persons known alive at the last wave (the same
             rule gives 11 years for the analysis cohort).
Output     : deaths and persons at risk by sex and attained age, and a Gompertz curve per sex
             q(a) = 1 - exp(-exp(alpha + beta * (a - 70))) fitted by binomial maximum likelihood
             on ages 50-95. The curve is the prior guess p*(a) of each urn.

The analysis cohort starts at Wave 10 (2010), so no death analysed there enters this prior.
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if len(sys.argv) < 2:
    raise SystemExit('usage: python build_prior.py "path/to/randhrs1992_2022v1.dta" [prior_table.csv]')
DTA = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "prior_table.csv"
BASE, WAVES = 4, range(5, 11)
FIT_AGES = (50, 95)

wanted = (["HHIDPN", "RAGENDER", "RADYEAR", "RADMONTH", f"INW{BASE}", f"R{BASE}MSTAT", f"R{BASE}AGEY_E",
           f"R{BASE}IWENDY", f"R{BASE}IWENDM"]
          + [f"R{w}IWSTAT" for w in WAVES] + [f"R{w}IWENDY" for w in WAVES] + [f"R{w}IWENDM" for w in WAVES])
with pd.read_stata(DTA, iterator=True) as reader:
    actual = {c.upper(): c for c in reader.read(1, convert_categoricals=False).columns}
missing = [v for v in wanted if v not in actual]
if missing:
    raise SystemExit(f"variables not found: {missing}")
with pd.read_stata(DTA, columns=[actual[v] for v in wanted], convert_categoricals=False,
                   convert_missing=False, chunksize=4000) as reader:
    df = pd.concat(list(reader), ignore_index=True)
df.columns = [c.upper() for c in df.columns]

pop = df[(df[f"INW{BASE}"] == 1) & (df[f"R{BASE}MSTAT"] == 1) & df[f"R{BASE}AGEY_E"].notna()
         & df[f"R{BASE}IWENDY"].notna() & df[f"R{BASE}IWENDM"].notna()].reset_index(drop=True)
print(f"married Wave {BASE} respondents: {len(pop)}")

med_month = {w: int(np.nanmedian(df.loc[df[f"R{w}IWSTAT"] == 1, f"R{w}IWENDY"] * 12
                                 + df.loc[df[f"R{w}IWSTAT"] == 1, f"R{w}IWENDM"])) for w in WAVES}
origin = (pop[f"R{BASE}IWENDY"] * 12 + pop[f"R{BASE}IWENDM"]).astype(int).values
stat = np.stack([pop[f"R{w}IWSTAT"].values for w in WAVES], axis=1)
iwm = np.stack([(pop[f"R{w}IWENDY"] * 12 + pop[f"R{w}IWENDM"]).values for w in WAVES], axis=1)
dy, dm = pop.RADYEAR.values, pop.RADMONTH.values

n = len(pop)
end = origin.astype(float).copy()
died = np.zeros(n, dtype=bool)
month_imputed = 0
for i in range(n):
    last_alive = origin[i]
    for j, w in enumerate(WAVES):
        code = stat[i, j]
        if code in (1, 4):
            last_alive = iwm[i, j] if (code == 1 and not np.isnan(iwm[i, j])) else med_month[w]
        elif code in (5, 6):
            if not np.isnan(dy[i]):
                m = dm[i]
                if np.isnan(m):
                    m, month_imputed = 6, month_imputed + 1
                death_month = dy[i] * 12 + m
                if death_month >= origin[i]:
                    died[i], last_alive = True, death_month
            break
        elif code == 7:
            break
    end[i] = last_alive
years = np.floor((end - origin) / 12).astype(int)
age0 = pop[f"R{BASE}AGEY_E"].astype(int).values
sex = (pop.RAGENDER.values == 2).astype(int)                 # 0 = men, 1 = women
# administrative end of follow-up: a survivor is censored after the completed years that precede the last
# interview, whereas a death is dated in its own year, so the deaths of the last years would be counted while
# part of the survivors are not at risk for the whole year
last = stat[:, -1]
followed = np.isin(last, (1, 4)) & ~died                    # known alive at the last wave
share = {m: float(np.mean(years[followed] >= m)) for m in range(8, 15)}
MAX_YEARS = max(m for m, v in share.items() if v >= 0.95)
print("share of the persons known alive at the last wave who reach m completed years: "
      + ", ".join(f"{m}: {v:.3f}" for m, v in share.items()))
late = died & (years >= MAX_YEARS)
print(f"administrative end of follow-up: {MAX_YEARS} completed years; deaths in later years, not used: "
      f"men {int(late[sex == 0].sum())}, women {int(late[sex == 1].sum())}")
died = died & (years < MAX_YEARS)
years = np.minimum(years, MAX_YEARS)
print(f"deaths observed: men {int(died[sex == 0].sum())}, women {int(died[sex == 1].sum())}; "
      f"death month imputed: {month_imputed}")

AGE_LO, AGE_HI = 20, 115
at_risk = np.zeros((2, AGE_HI - AGE_LO + 1))
deaths = np.zeros_like(at_risk)
for i in range(n):
    a = age0[i] - AGE_LO
    at_risk[sex[i], a:a + years[i]] += 1                    # years survived
    if died[i]:
        at_risk[sex[i], a + years[i]] += 1
        deaths[sex[i], a + years[i]] += 1
ages = np.arange(AGE_LO, AGE_HI + 1)


def gompertz_q(age, alpha, beta):
    return 1.0 - np.exp(-np.exp(alpha + beta * (age - 70.0)))


params = np.zeros((2, 2))
fit = (ages >= FIT_AGES[0]) & (ages <= FIT_AGES[1])
for s in range(2):
    d, r = deaths[s, fit], at_risk[s, fit]

    def nll(th):
        q = np.clip(gompertz_q(ages[fit], th[0], th[1]), 1e-12, 1 - 1e-12)
        return -np.sum(d * np.log(q) + (r - d) * np.log(1 - q))

    res = minimize(nll, np.array([-3.5, 0.09]), method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-8, maxiter=4000))
    params[s] = res.x
    print(f"{['men', 'women'][s]:5s}: alpha = {res.x[0]:.4f}, beta = {res.x[1]:.4f} "
          f"(person-years {int(r.sum())}, deaths {int(d.sum())}, ages {FIT_AGES[0]}-{FIT_AGES[1]})")

rows = []
for s in range(2):
    for k, a in enumerate(ages):
        rows.append(dict(sex=["men", "women"][s], age=int(a), at_risk=int(at_risk[s, k]), deaths=int(deaths[s, k]),
                         q_observed=(deaths[s, k] / at_risk[s, k]) if at_risk[s, k] > 0 else np.nan,
                         q_gompertz=gompertz_q(a, *params[s])))
tab = pd.DataFrame(rows)
tab.to_csv(OUT, index=False)

print("\nobserved versus Gompertz one-year death probability (5-year age bands):")
for s in range(2):
    for lo in range(50, 95, 5):
        m = (ages >= lo) & (ages < lo + 5)
        r, d = at_risk[s, m].sum(), deaths[s, m].sum()
        expected = (at_risk[s, m] * gompertz_q(ages[m], *params[s])).sum()
        print(f"  {['men', 'women'][s]:5s} {lo}-{lo + 4}: at risk {int(r):6d}  deaths {int(d):5d}  expected {expected:7.1f}  "
              f"q observed {d / r:.4f}")
