"""Independent check of the effect estimates: person-year table built with pandas and a
complementary log-log binomial GLM (no frailty), fitted by sex with statsmodels.
With a small frailty variance, its hazard ratios should be close to the Bayesian model's."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

d = pd.read_csv("cohort_w10.csv")
d = d[(d.age_h >= 50) & (d.age_w >= 50)].reset_index(drop=True)
for s in "hw":
    d[f"bmiz_{s}"] = (d[f"bmi_{s}"] - d[f"bmi_{s}"].mean()) / d[f"bmi_{s}"].std()

rows = []
for s, o in (("h", "w"), ("w", "h")):
    for r in d.itertuples(index=False):
        n, died = getattr(r, f"n_surv_{s}"), getattr(r, f"died_{s}")
        pn, pdied = getattr(r, f"n_surv_{o}"), getattr(r, f"died_{o}")
        for t in range(n + (1 if died else 0)):
            rows.append((s, getattr(r, f"age_{s}") + t, int(died and t == n), getattr(r, f"bmiz_{s}"),
                         getattr(r, f"smoker_{s}"), getattr(r, f"hyp_{s}"), getattr(r, f"diab_{s}"),
                         getattr(r, f"bmiz_{o}"), getattr(r, f"smoker_{o}"), int(pdied and pn < t)))
py = pd.DataFrame(rows, columns=["sex", "age", "y", "bmi", "smoker", "hyp", "diab", "p_bmi", "p_smoker", "widowed"])
print("person-years:", len(py), " deaths:", int(py.y.sum()), " widowed person-years:", int(py.widowed.sum()))
names = ["bmi", "smoker", "hyp", "diab", "p_bmi", "p_smoker", "widowed"]
for s, label in (("h", "husbands"), ("w", "wives")):
    sub = py[py.sex == s]
    age_band = pd.get_dummies(sub.age.clip(upper=95), prefix="a", drop_first=True, dtype=float)   # one level per age, 95+ pooled
    X = sm.add_constant(pd.concat([sub[names].astype(float).reset_index(drop=True), age_band.reset_index(drop=True)], axis=1))
    res = sm.GLM(sub.y.values, X, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
    ci = res.conf_int()
    print(label)
    for nm in names:
        print(f"  {nm:9s} HR {np.exp(res.params[nm]):.2f}  [{np.exp(ci.loc[nm, 0]):.2f} ; {np.exp(ci.loc[nm, 1]):.2f}]")
