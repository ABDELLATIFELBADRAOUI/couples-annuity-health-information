"""
build_cohort.py -- couples cohort for the dyadic mortality model, built from the RAND HRS
Longitudinal File 1992-2022 (V1). Variable codes checked against randhrs1992_2022v1.pdf.

Cohort (baseline = Wave 10, 2010):
  husband: RAGENDER = 1 (male), INW10 = 1, R10MSTAT = 1 (married), S10HHIDPN > 0
  wife   : the record whose HHIDPN = husband's S10HHIDPN, RAGENDER = 2, INW10 = 1,
           and whose S10HHIDPN points back to the husband
  complete covariates for BOTH spouses at Wave 10: age, BMI, smoking now, ever high blood
  pressure, ever diabetes.
  Also exported, without entering the cohort definition: years of education of each spouse
  (RAEDYRS) and household net wealth at Wave 10 (H10ATOTB).

Follow-up (Waves 11-16, 2012-2022), per spouse:
  RwIWSTAT: 1 = responded alive, 4 = non-response alive, 5 = died this wave,
            6 = died in a previous wave, 7 = dropped from sample.
  death observed : a wave with code 5 or 6 appears before any code 7, and RADYEAR is known
                   (RADMONTH missing -> month 6, counted in the log)
  otherwise      : censored at the last wave with code 1 or 4 (interview end date if the person
                   responded, else the median interview month of that wave's respondents)
  time origin    : husband's Wave 10 interview end month (shared by the couple)
  discrete time  : completed years since the origin
  administrative end: every spouse is censored after MAX_YEARS = 11 completed years (see below)
  If one spouse is censored alive, the other is censored at the same time, because the
  widowhood indicator is unknown after that point.
"""
import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    raise SystemExit('usage: python build_cohort.py "path/to/randhrs1992_2022v1.dta" [cohort_w10.csv]')
DTA = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "cohort_w10.csv"
WAVES = range(11, 17)
MAX_YEARS = 11                # administrative end of follow-up, in completed years since the baseline interview

wanted = (["HHIDPN", "RAGENDER", "RADYEAR", "RADMONTH", "INW10", "R10MSTAT", "S10HHIDPN",
           "R10AGEY_E", "R10IWENDY", "R10IWENDM", "R10BMI", "R10SMOKEN", "R10HIBPE", "R10DIABE",
           "RAEDYRS", "H10ATOTB"]
          + [f"R{w}IWSTAT" for w in WAVES] + [f"R{w}IWENDY" for w in WAVES] + [f"R{w}IWENDM" for w in WAVES])

with pd.read_stata(DTA, iterator=True) as reader:
    actual = {c.upper(): c for c in reader.read(1, convert_categoricals=False).columns}
missing = [v for v in wanted if v not in actual]
if missing:
    raise SystemExit(f"variables not found: {missing}")
# read in chunks: the file is 1.7 GB and only 32 of its columns are needed
with pd.read_stata(DTA, columns=[actual[v] for v in wanted], convert_categoricals=False,
                   convert_missing=False, chunksize=4000) as reader:
    df = pd.concat(list(reader), ignore_index=True)
df.columns = [c.upper() for c in df.columns]
print(f"records read: {len(df)}")

log = {}
hus = df[(df.RAGENDER == 1) & (df.INW10 == 1) & (df.R10MSTAT == 1) & (df.S10HHIDPN > 0)]
wif = df[(df.RAGENDER == 2) & (df.INW10 == 1)]
cp = hus.merge(wif, left_on="S10HHIDPN", right_on="HHIDPN", suffixes=("_H", "_W"))
log["husbands married at W10 with spouse id"] = len(hus)
log["matched to a female W10 respondent"] = len(cp)
cp = cp[cp.S10HHIDPN_W == cp.HHIDPN_H]
log["reciprocal spouse link"] = len(cp)

cov = ["R10AGEY_E", "R10BMI", "R10SMOKEN", "R10HIBPE", "R10DIABE"]
cc = cp[[f"{c}_{s}" for c in cov for s in "HW"]].notna().all(axis=1)
cp = cp[cc & cp.R10IWENDY_H.notna() & cp.R10IWENDM_H.notna()].copy()
log["complete covariates for both spouses"] = len(cp)

# median interview month of respondents, per wave (for non-respondents known alive)
med_month = {w: int(np.nanmedian(df.loc[df[f"R{w}IWSTAT"] == 1, f"R{w}IWENDY"] * 12
                                 + df.loc[df[f"R{w}IWSTAT"] == 1, f"R{w}IWENDM"])) for w in WAVES}
origin = (cp.R10IWENDY_H * 12 + cp.R10IWENDM_H).astype(int).values


def spouse_follow_up(s):
    stat = np.stack([cp[f"R{w}IWSTAT_{s}"].values for w in WAVES], axis=1)
    iwm = np.stack([(cp[f"R{w}IWENDY_{s}"] * 12 + cp[f"R{w}IWENDM_{s}"]).values for w in WAVES], axis=1)
    n = len(cp)
    end = origin.copy().astype(float)          # default: censored at baseline
    died = np.zeros(n, dtype=bool)
    month_imputed = no_year = before_origin = 0
    for i in range(n):
        last_alive = origin[i]
        for j, w in enumerate(WAVES):
            code = stat[i, j]
            if code in (1, 4):
                last_alive = iwm[i, j] if (code == 1 and not np.isnan(iwm[i, j])) else med_month[w]
            elif code in (5, 6):
                y, m = cp[f"RADYEAR_{s}"].values[i], cp[f"RADMONTH_{s}"].values[i]
                if not np.isnan(y):
                    if np.isnan(m):
                        m, month_imputed = 6, month_imputed + 1
                    dm = y * 12 + m
                    if dm >= origin[i]:
                        died[i], last_alive = True, dm
                    else:
                        before_origin += 1     # death dated before the baseline interview: censored at the last wave seen alive
                else:
                    no_year += 1               # year of death unknown: censored at the last wave seen alive
                break
            elif code == 7:
                break
        end[i] = last_alive
    codes = {int(w): {int(k): int(v) for k, v in zip(*np.unique(stat[:, j][~np.isnan(stat[:, j])], return_counts=True))}
             for j, w in enumerate(WAVES)}
    return end, died, month_imputed, (no_year, before_origin), codes


end_h, died_h, imp_h, lost_h, codes_h = spouse_follow_up("H")
end_w, died_w, imp_w, lost_w, codes_w = spouse_follow_up("W")
log["death month imputed (husbands, wives)"] = (imp_h, imp_w)
log["deaths without a year of death, treated as censored (husbands, wives)"] = (lost_h[0], lost_w[0])
log["deaths dated before the baseline interview, treated as censored (husbands, wives)"] = (lost_h[1], lost_w[1])
for w in WAVES:
    log[f"interview status codes at wave {w} (husbands; wives)"] = (codes_h[w], codes_w[w])

yrs_h = np.floor((end_h - origin) / 12).astype(int)
yrs_w = np.floor((end_w - origin) / 12).astype(int)
# rule of the administrative end (also used in build_prior.py): the largest number of completed years reached by
# at least 95 % of the spouses known alive at the last wave; it is printed here as a check of MAX_YEARS
known = np.concatenate([np.isin(cp[f"R{WAVES[-1]}IWSTAT_{s}"].values, (1, 4)) & ~d for s, d in (("H", died_h), ("W", died_w))])
reach = {m: float(np.mean(np.concatenate([yrs_h, yrs_w])[known] >= m)) for m in range(8, 15)}
log["share of the spouses known alive at the last wave who reach m completed years"] = {m: round(v, 3) for m, v in reach.items()}
assert MAX_YEARS == max(m for m, v in reach.items() if v >= 0.95), "MAX_YEARS no longer follows the rule"
# Administrative end of follow-up. A survivor is censored after the completed years that precede the last
# interview, whereas a death is dated in its own year: in the last years of follow-up the deaths would be
# counted while most survivors are not at risk for the whole year. Every spouse is therefore censored after
# MAX_YEARS completed years, a duration that nearly all spouses still followed at the last wave reach;
# a death in a later year is not used.
log["deaths after the administrative end, before the partner rule below (husbands, wives)"] = (int((died_h & (yrs_h >= MAX_YEARS)).sum()),
                                                                          int((died_w & (yrs_w >= MAX_YEARS)).sum()))
died_h, died_w = died_h & (yrs_h < MAX_YEARS), died_w & (yrs_w < MAX_YEARS)
yrs_h, yrs_w = np.minimum(yrs_h, MAX_YEARS), np.minimum(yrs_w, MAX_YEARS)
# A spouse censored alive after n completed years is known alive at the start of year n, so the
# partner's widowhood indicator is known through year n: the partner is observed at most through
# year n (a death in year <= n is kept; follow-up beyond year n is censored at n + 1 years).
def cap(yrs_self, died_self, yrs_partner, died_partner):
    limit = np.where(~died_partner, yrs_partner, 10**6)
    lost_death = died_self & (yrs_self > limit)
    new_yrs = np.where(died_self & ~lost_death, yrs_self, np.minimum(yrs_self, limit + 1))
    new_yrs = np.where(lost_death, limit + 1, new_yrs)
    return new_yrs, died_self & ~lost_death
(yrs_w_new, died_w_new) = cap(yrs_w, died_w, yrs_h, died_h)
(yrs_h_new, died_h_new) = cap(yrs_h, died_h, yrs_w, died_w)
log["deaths censored by partner's censoring (husbands, wives)"] = (int((died_h & ~died_h_new).sum()),
                                                                   int((died_w & ~died_w_new).sum()))
yrs_h, died_h, yrs_w, died_w = yrs_h_new, died_h_new, yrs_w_new, died_w_new

out = pd.DataFrame({
    "hhidpn_h": cp.HHIDPN_H.values, "hhidpn_w": cp.HHIDPN_W.values,
    "age_h": cp.R10AGEY_E_H.astype(int).values, "age_w": cp.R10AGEY_E_W.astype(int).values,
    "bmi_h": cp.R10BMI_H.values, "bmi_w": cp.R10BMI_W.values,
    "smoker_h": cp.R10SMOKEN_H.values, "smoker_w": cp.R10SMOKEN_W.values,
    "hyp_h": cp.R10HIBPE_H.values, "hyp_w": cp.R10HIBPE_W.values,
    "diab_h": cp.R10DIABE_H.values, "diab_w": cp.R10DIABE_W.values,
    # socio-economic covariates (may be missing; they do not enter the cohort definition)
    "educ_h": cp.RAEDYRS_H.values, "educ_w": cp.RAEDYRS_W.values,      # years of education, 17 = 17 or more
    "wealth": cp.H10ATOTB_H.values,                                    # household net wealth at Wave 10 (USD)
    "n_surv_h": yrs_h, "died_h": died_h, "n_surv_w": yrs_w, "died_w": died_w,
})
keep = ~((out.n_surv_h == 0) & ~out.died_h & (out.n_surv_w == 0) & ~out.died_w)
log["couples with no follow-up dropped"] = int((~keep).sum())
out = out[keep].reset_index(drop=True)
out.to_csv(OUT, index=False)

log["final couples"] = len(out)
log["deaths observed (husbands, wives)"] = (int(out.died_h.sum()), int(out.died_w.sum()))
log["censoring rate (husbands, wives)"] = (round(1 - out.died_h.mean(), 3), round(1 - out.died_w.mean(), 3))
log["max completed years"] = int(max(out.n_surv_h.max(), out.n_surv_w.max()))
log["baseline age range (husbands, wives)"] = ((int(out.age_h.min()), int(out.age_h.max())),
                                               (int(out.age_w.min()), int(out.age_w.max())))
for k, v in log.items():
    print(f"{k}: {v}")
