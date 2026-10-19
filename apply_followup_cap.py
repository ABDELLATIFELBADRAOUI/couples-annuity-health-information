"""
apply_followup_cap.py -- administrative end of follow-up applied to a cohort file built without it.

Why. Follow-up is counted in completed years since the baseline interview. A survivor interviewed at the
last wave after, say, 11 years and 8 months is censored after 11 completed years: the 8 months are not
used. A person who dies after 11 years and 3 months is a death of year 11. In year 11 the deaths were
therefore counted while most of the survivors were not at risk: the death rate of the last years of
follow-up was overstated (crude rate 3.7 % in year 10, 7.0 % in year 11, 27 % in year 12).
Remedy: every spouse is censored after MAX_YEARS completed years; a death in a later year is not used.
With MAX_YEARS = 11 nearly all the spouses still followed at the last wave are observed over the whole
of every year kept (years 0 to 10).

build_cohort.py now applies the same rule when it builds the file (same result: the rule commutes with
the censoring of a spouse at the partner's censoring). This script only converts a file built before.

The same script gives the file of a shorter follow-up (sensitivity analysis): the rule also commutes with
itself, so a file already limited to 11 years can be limited to 9.

Usage: python apply_followup_cap.py <old cohort csv> <new cohort csv> [max_years, default 11]
"""
import sys

import numpy as np
import pandas as pd

MAX_YEARS = 11

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        MAX_YEARS = int(sys.argv[3])
    d = pd.read_csv(src, float_precision="round_trip")
    before = (int(d.died_h.sum()), int(d.died_w.sum()))
    for s in "hw":
        late = d[f"n_surv_{s}"] >= MAX_YEARS
        d[f"died_{s}"] = d[f"died_{s}"] & ~late
        d[f"n_surv_{s}"] = np.minimum(d[f"n_surv_{s}"], MAX_YEARS)
    keep = ~((d.n_surv_h == 0) & ~d.died_h & (d.n_surv_w == 0) & ~d.died_w)
    d = d[keep].reset_index(drop=True)
    d.to_csv(out, index=False)
    print(f"couples: {len(d)}; deaths before (husbands, wives): {before}; after: ({int(d.died_h.sum())}, {int(d.died_w.sum())}); "
          f"largest number of completed years: {int(max(d.n_surv_h.max(), d.n_surv_w.max()))}")
