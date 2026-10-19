# Code for "The Value of Health Information in Couples' Annuity Choice: A Dyadic Mortality Model, an Exact Benchmark, and Explainable Reinforcement Learning"

Abdellatif El Badraoui, Yassine Mouhssine, Abdelkader El Alaoui, Said Ouatik El Alaoui

This repository contains the code that produces every result, table and figure of the article: the cohort of couples, the Bayesian mortality model of couples and its assessment, the couple's annuity decision solved by dynamic programming, the learning agents, and the explanations of the annuity decision.

## Environment

Python 3.11. The package versions used for the article are fixed in `requirements.txt`; `requirements-unpinned.txt` lists the same packages without versions.

```
pip install -r requirements.txt
```

## Data

The study uses the RAND HRS Longitudinal File 2022 (V1), the Stata file `randhrs1992_2022v1.dta`, which registered users can download from the Health and Retirement Study (https://hrsdata.isr.umich.edu). The HRS conditions of use do not permit the transfer of the data to third parties. This repository therefore contains no data and no file derived from the data. Two scripts build the analysis files from the original file:

```
python build_prior.py "path/to/randhrs1992_2022v1.dta"     # writes prior_table.csv (prior guess of the baseline mortality)
python build_cohort.py "path/to/randhrs1992_2022v1.dta"    # writes cohort_w10.csv (the couples of wave 10, 2010)
```

The result files written by the other scripts are derived from the data as well. `python run_all.py --bundle` archives them for the user's own records; they must not be redistributed.

## Check

```
python run_all.py --check
```

`run_all.py --check` reports the installed versions, the two data files above (as missing until they are built), the free disk space, and the result of short tests of the sampler (JAX/NumPyro), of the deep survival models (PyTorch/pycox) and of scikit-survival.

## Running the computations

```
python run_all.py                  # every computation, in parallel; can be stopped and restarted
python run_all.py --workers 6      # number of jobs at a time (default: number of cores - 1, at most 8)
python run_all.py --list           # the jobs and their state
```

Each job is a script of the repository. The jobs, their arguments and their dependencies are listed in `run_all.py`. The output of a job goes to `results_queue/<job>.log`, and a finished job is not run again.

Two checks of the implementation use simulated data only. They are run separately and write to the folder `results`, which `run_all.py` creates:

```
python check_likelihood.py > results/check_likelihood.txt
python check_urn_conjugacy.py > results/check_urn_conjugacy_jax0.10.2.txt
```

Finally, the numbers, tables and figures of the article are written from the result files:

```
python paper/build_assets.py       # paper/numbers.tex, paper/tables*/, paper/figures/
```

Indicative run times on the authors' laptop (20 logical processors): about 192 minutes for one run of the gradient agent, about 65 minutes for one run of proximal policy optimisation with 1.02 million simulated lives, and about 100 minutes for one run with 3.07 million lives.

## Contents

| Stage | Scripts |
|---|---|
| Data | `build_prior.py`, `build_cohort.py`, `apply_followup_cap.py` (end of follow-up after a given number of completed years; shorter follow-up of the sensitivity analysis) |
| Mortality model | `dyadic_mortality.py` (model, closed-form likelihood, simulator), `pipeline.py` (data, nested variants, exact predictions), `fit_models.py`, `fit_real.py`, `fit_sens.py` (other priors), `sim_study.py` (simulation study) |
| Out-of-sample comparison | `evaluate.py`, `copula_baseline.py` (Frank copula with Gompertz margins), `deep_models.py` (DeepSurv, DeepHit), `run_deep_all.py`, `metrics_util.py` |
| Annuity decision | `decision_law.py` (exact law of a couple's survival history), `decision_dp.py` (dynamic programming), `decision_setup.py`, `run_decision.py`, `run_uncertainty.py`, `annuity_gap.py` |
| Learning agents | `couple_env.py` (simulator), `rl_core.py`, `rl_apg.py` (gradient through the simulator), `rl_ppo.py` (proximal policy optimisation), `ppo_long.py` (longer run with the treatment of the exploration noise chosen on the training couples), `rule_baseline.py` (consumption rule without learning), `evaluate_agents.py` (exact gap to the optimum) |
| Explanations | `xai_setup.py`, `xai_dp.py` (exact optimum and exact value of the agents for the training couples and the hybrid couples), `xai_shap.py` (Shapley values), `xai_tree.py` (decision trees) |
| Checks | `check_*.py` (numerical checks of the implementation; most are reported in the appendix of the article) |
| Summaries | `summarize_*.py` (summaries in JSON, and reports in Markdown written in French) |
| Article | `paper/build_assets.py` (every number, table and figure of the article) |
| Runner | `run_all.py` |

## License

MIT (see `LICENSE`).

## Citation

See `CITATION.cff`.
