"""
run_all.py -- runs every computation of the paper, in parallel, on Windows, macOS or Linux.

    python run_all.py --check          checks the Python packages and the data (one or two minutes)
    python run_all.py                  runs everything (several hours); can be stopped and restarted
    python run_all.py --workers 6      same, with 6 jobs at a time (default: number of cores - 1, at most 8)
    python run_all.py --list           lists the jobs and their state
    python run_all.py --bundle         writes two zip files of the results, for archiving (they contain results
                                       derived from the HRS data: do not redistribute them)

Each job is a script of the project. A job starts when the jobs it depends on have finished (in the order
of the list below when several can start), and its output goes to results_queue/<job>.log. When a job ends without error, the file results_queue/done/<job>.ok
is written: a job with this file is not run again, so the script can be stopped (Ctrl+C) and restarted.
A job that fails is reported at the end with its log; the jobs that depend on it are not run.
"""
import argparse
import datetime
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile

Q = "results_queue"
DONE = os.path.join(Q, "done")
PY = sys.executable
# When this script is started from Jupyter, the notebook's settings are inherited: its plotting backend
# (MPLBACKEND = matplotlib_inline, absent from this environment) and possibly its module path. They are
# replaced here, for this script and for every job it starts.
os.environ["MPLBACKEND"] = "Agg"
for _var in ("PYTHONPATH", "PYTHONHOME"):
    os.environ.pop(_var, None)


class Job:
    def __init__(self, name, args, after=(), env=None, stdout=None, weight=1):
        self.name, self.args, self.after, self.env, self.stdout, self.weight = name, args, list(after), env or {}, stdout, weight


def jobs():
    J = []
    add = lambda *a, **k: J.append(Job(*a, **k))
    fits = []
    # main split: nested models on the estimation couples, full model on all couples
    for v in ("M3", "M0", "MD", "M1", "M2"):
        for tag, seed in (("a", 101), ("b", 202)):
            add(f"fit_{v}_train_{tag}", ["fit_models.py", v, "train", str(seed), "2", f"results_eval/{v}_train_{tag}.npz"])
            fits.append(f"fit_{v}_train_{tag}")
    for tag, seed in (("a", 101), ("b", 202)):
        add(f"fit_M3_full_{tag}", ["fit_models.py", "M3", "full", str(seed), "2", f"results_eval/M3_full_{tag}.npz"])
    m3 = ["fit_M3_train_a", "fit_M3_train_b"]
    m0 = ["fit_M0_train_a", "fit_M0_train_b"]
    agents_ref = [f"apg_seed{k}" for k in range(5)] + ["ppoa_seed0", "ppo_seed0"]
    # exact optimum of the estimation couples (monitoring of the agents)
    add("xai_train", ["xai_dp.py", "train"], after=m3)
    # decision analysis
    add("dec_base", ["run_decision.py", "base_limit", "pre50_limit"], after=fits)
    # out-of-sample comparison (copula, deep models, evaluation), main split
    add("eval_main", ["run_deep_all.py", "main"], after=fits)
    # agents
    for k in range(5):
        add(f"apg_seed{k}", ["rl_apg.py", str(k), "12000", "256", "64"], after=["xai_train"])
    for k in range(5):
        add(f"ppoa_seed{k}", ["rl_ppo.py", str(k), "500", "2048", "64", "25", "-3.0"], after=["xai_train"])
        add(f"ppo_seed{k}", ["rl_ppo.py", str(k), "500", "2048", "64", "25"], after=["xai_train"])
    add("rule_af", ["rule_baseline.py"], after=["dec_base"])
    add("eval_agents_1", ["evaluate_agents.py"], after=agents_ref + ["dec_base"], env={"CARUP_AGENTS": ",".join(agents_ref)})
    # explanations (exact policy on hybrid couples, Shapley values, trees)
    add("xai_train2", ["xai_dp.py", "train"], after=agents_ref + m3)
    add("xai_grid", ["xai_dp.py", "grid"], after=["xai_train"])
    add("xai_hybrids", ["xai_dp.py", "hybrids"], after=agents_ref + m3)
    add("xai_effects", ["xai_dp.py", "effects"], after=agents_ref + m3)
    add("xai_shap", ["xai_shap.py"], after=["xai_hybrids", "xai_effects", "dec_base"] + agents_ref)
    add("xai_tree", ["xai_tree.py"], after=["xai_train2", "xai_grid", "xai_shap", "eval_agents_1"])
    add("check_xai", ["check_xai.py"], after=["xai_tree"], stdout="results_xai/check_xai.txt")
    # longer PPO run for the better treatment of the noise (chosen on the training couples), then all agents
    ppo_all = [f"{k}_seed{s}" for k in ("ppoa", "ppo") for s in range(5)]
    add("ppo_long", ["ppo_long.py"], after=ppo_all)
    add("eval_agents_final", ["evaluate_agents.py"], after=["ppo_long", "xai_tree", "check_xai"] + agents_ref + ppo_all)
    add("check_rl", ["check_rl.py"], after=["apg_seed0", "dec_base"], stdout="results_rl/check_rl.txt")
    # same budget as the gradient agents for PPO: four more runs of 1,500 iterations (3.07 million lives) with the treatment of
    # the noise chosen by ppo_long.py (annealed, results_rl/ppo_long_choice.json), then all agents evaluated again
    long_seeds = [f"ppoa_long_seed{k}" for k in range(6, 10)]
    for k in range(6, 10):
        add(f"ppoa_long_seed{k}", ["rl_ppo.py", str(k), "1500", "2048", "64", "25", "-3.0"], after=["ppo_long"])
    add("eval_agents_final2", ["evaluate_agents.py"], after=long_seeds + ["eval_agents_final"])
    # sensitivity of the decision analysis, stress case, model uncertainty (jobs listed later start later)
    add("dec_sens", ["run_decision.py", "freephi_limit", "g2_limit", "g5_limit", "load0_limit", "load20_limit", "base_truthM0_limit"], after=["dec_base"])
    add("dec_strong", ["run_decision.py", "strong_limit"], after=["dec_base"])
    add("uncertainty", ["run_uncertainty.py"], after=["dec_base"])
    # dependence and annuity values; follow-up of nine years
    add("annuity_gap", ["annuity_gap.py"], after=m0 + m3, stdout="results_eval/annuity_gap.txt")
    add("cohort_cap9", ["apply_followup_cap.py", "cohort_w10.csv", "results_cap9/cohort_cap9.csv", "9"])
    for tag, seed in (("a", 101), ("b", 202)):
        add(f"cap9_{tag}", ["fit_models.py", "M3", "full", str(seed), "2", f"results_cap9/M3_full_{tag}.npz"], after=["cohort_cap9"],
            env={"CARUP_COHORT": "results_cap9/cohort_cap9.csv"})
    # prior sensitivity
    settings = ["base", "c_quarter", "c_times4", "centre_low", "centre_high", "scale05", "scale2", "student", "kappa05", "kappa2"]
    for s in settings:
        add(f"sens_{s}", ["fit_sens.py", s, "101", f"results_sens/{s}.npz"])
    # other random splits
    seed_jobs = {k: [] for k in range(1, 5)}
    for k in range(1, 5):
        for v in ("M0", "M3"):
            for tag, seed in (("a", 101), ("b", 202)):
                name = f"seed{k}_{v}_{tag}"
                add(name, ["fit_models.py", v, "train", str(seed), "2", f"results_seeds/seed_{k}/{v}_train_{tag}.npz"], env={"CARUP_SPLIT_SEED": str(k)})
                seed_jobs[k].append(name)
        add(f"seeds_eval_{k}", ["run_deep_all.py", str(k)], after=seed_jobs[k])
    # simulation study (truth: posterior means of the full model)
    sims = []
    add("sim_truth", ["-c", "import sim_study; sim_study.truth_from_fit()"], after=m3)        # written once, before the replications
    for lo in (1, 6, 11, 16):
        for sc in ("base", "indep", "frailty", "widow", "both", "short"):
            add(f"sim_{sc}_{lo}", ["sim_study.py", sc, str(lo), str(lo + 4)], after=["sim_truth"])
            sims.append(f"sim_{sc}_{lo}")
    for sc, a, b in (("small", 1, 20), ("base_ref", 1, 5), ("indep_ref", 1, 5)):
        add(f"sim_{sc}", ["sim_study.py", sc, str(a), str(b)], after=["sim_truth"])
        sims.append(f"sim_{sc}")
    # checks of the implementation
    add("check_law", ["check_law.py"], after=m3, stdout="results_decision/check_law.txt")
    add("check_decision", ["check_decision.py"], after=["dec_base"], stdout="results_decision/check_decision.txt")
    add("check_env", ["check_env.py"], after=["dec_base"], stdout="results_decision/check_env.txt")
    add("check_predict", ["check_predict.py"], after=m3, stdout="results_eval/check_predict.txt")
    add("check_metrics", ["check_metrics.py"], after=["eval_main"])
    add("check_glm", ["check_glm.py"], stdout="results_real/check_glm.txt")
    for tag, seed in (("a", 101), ("b", 202)):
        add(f"real_{tag}", ["fit_real.py", "cohort_w10.csv", "prior_table.csv", str(seed), "2", f"results_real/chains_{tag}.npz"])
    # summaries
    add("summarize_full", ["summarize_full.py"], after=["fit_M3_full_a", "fit_M3_full_b"])
    add("summarize_diagnostics", ["summarize_diagnostics.py"], after=fits + ["fit_M3_full_a", "fit_M3_full_b"])
    add("summarize_real", ["summarize_real.py"], after=["real_a", "real_b"])
    add("summarize_sens", ["summarize_sens.py"], after=[f"sens_{s}" for s in settings] + m3)
    add("summarize_sim", ["summarize_sim.py"], after=sims)
    add("summarize_seeds", ["summarize_seeds.py"], after=["eval_main"] + [f"seeds_eval_{k}" for k in range(1, 5)])
    add("summarize_decision", ["summarize_decision.py", "limit"], after=["dec_base", "dec_sens"])
    add("summarize_uncertainty", ["summarize_uncertainty.py"], after=["uncertainty"])
    return J


FOLDERS = ["results_eval", "results_decision", "results_rl", "results_rl_exp", "results_xai", "results_sens", "results_sim", "results_cap9",
           "results_real", "results"] + [f"results_seeds/seed_{k}" for k in range(1, 5)]


def environment(extra):
    env = dict(os.environ)
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"})
    env.update(extra)
    return env


def is_done(name):
    return os.path.exists(os.path.join(DONE, name + ".ok"))


def fmt(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def run(J, workers):
    os.makedirs(DONE, exist_ok=True)
    for d in FOLDERS:
        os.makedirs(d, exist_ok=True)
    names = {j.name for j in J}
    for j in J:
        missing = [a for a in j.after if a not in names]
        assert not missing, f"{j.name} depends on unknown jobs {missing}"
    todo = [j for j in J if not is_done(j.name)]
    print(f"{len(J)} jobs, {len(J) - len(todo)} already done, {len(todo)} to run, {workers} at a time. Logs: {Q}/<job>.log", flush=True)
    running, failed, t_start = {}, {}, time.time()
    try:
        while todo or running:
            for name, (p, t0, j, log) in list(running.items()):
                code = p.poll()
                if code is None:
                    continue
                log.close()
                del running[name]
                if code == 0:
                    with open(os.path.join(DONE, name + ".ok"), "w", encoding="utf-8") as f:
                        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {time.time() - t0:.0f}s\n")
                    print(f"[{fmt(time.time() - t_start)}] done    {name} ({fmt(time.time() - t0)}); {len(todo)} waiting, {len(running)} running", flush=True)
                else:
                    failed[name] = code
                    print(f"[{fmt(time.time() - t_start)}] FAILED  {name} (exit code {code}); see {Q}/{name}.log", flush=True)
            changed = True                                   # a job that depends on a failed job is not run
            while changed:
                changed = False
                for j in list(todo):
                    if any(a in failed for a in j.after):
                        todo.remove(j)
                        failed[j.name] = "not run: a job it depends on failed"
                        changed = True
            used = sum(r[2].weight for r in running.values())
            for j in list(todo):
                if used + j.weight > workers:
                    break
                if all(is_done(a) for a in j.after):
                    todo.remove(j)
                    log = open(os.path.join(Q, j.name + ".log"), "w", encoding="utf-8")
                    out = open(j.stdout, "w", encoding="utf-8") if j.stdout else log
                    if j.stdout:
                        log.write(f"standard output written to {j.stdout}\n")
                        log.flush()
                    p = subprocess.Popen([PY, "-u"] + j.args, stdout=out, stderr=subprocess.STDOUT if not j.stdout else log, env=environment(j.env))
                    if j.stdout:
                        out.close()
                    running[j.name] = (p, time.time(), j, log)
                    used += j.weight
                    print(f"[{fmt(time.time() - t_start)}] start   {j.name}", flush=True)
            if not running and todo:
                waiting = [j.name for j in todo]
                print(f"nothing can start any more; jobs left: {waiting}", flush=True)
                for n in waiting:
                    failed.setdefault(n, "not run: a job it depends on failed")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nstopping the running jobs (they will start again from the beginning next time) ...", flush=True)
        for name, (p, t0, j, log) in running.items():
            p.terminate()
        for name, (p, t0, j, log) in running.items():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
            log.close()
        raise SystemExit(1)
    print(f"\nfinished in {fmt(time.time() - t_start)}: {len(J) - len(failed)} jobs done, {len(failed)} not done.")
    for n, why in failed.items():
        print(f"  {n}: {why if isinstance(why, str) else f'exit code {why}, see {Q}/{n}.log'}")
    if not failed:
        print("Everything is done. Tables, figures and numbers of the paper:  python paper/build_assets.py")
    return 0 if not failed else 1


def check():
    ok = True
    print(f"Python {platform.python_version()} on {platform.system()} {platform.release()}, {os.cpu_count()} logical processors")
    if sys.version_info[:2] != (3, 11):
        print("  note: the computations were prepared with Python 3.11; another version may work but is not tested")
    try:
        import importlib
        for mod, want in (("numpy", "2.4.4"), ("scipy", "1.17.1"), ("pandas", "3.0.2"), ("jax", "0.10.2"), ("numpyro", "0.22.0"), ("arviz", "0.23.4"),
                          ("torch", "2.14.1"), ("pycox", "0.3.0"), ("torchtuples", "0.2.2"), ("sksurv", "0.28.0"), ("sklearn", "1.9.1"),
                          ("statsmodels", "0.15.0"), ("shap", "0.51.0"), ("matplotlib", "3.10.9")):
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            print(f"  {mod:12s} {v:14s}" + ("" if v.split("+")[0] == want else f"  (prepared with {want})"))
    except Exception as e:
        print(f"  MISSING PACKAGE: {e!r}  ->  pip install -r requirements.txt")
        return 1
    import numpy as np
    for f in ("cohort_w10.csv", "prior_table.csv"):
        if not os.path.exists(f):
            print(f"  missing data file: {f}")
            ok = False
    t0 = time.time()
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def model(y):
        mu = numpyro.sample("mu", dist.Normal(0, 1))
        numpyro.sample("y", dist.Normal(mu, 1), obs=y)
    mcmc = MCMC(NUTS(model), num_warmup=200, num_samples=200, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), jnp.asarray(np.random.default_rng(0).normal(1.0, 1.0, 200)))
    mu = float(mcmc.get_samples()["mu"].mean())
    print(f"  JAX / NumPyro sampler: posterior mean {mu:.2f} (expected about 1), {time.time() - t0:.1f}s" + ("" if abs(mu - 1) < 0.3 else "  PROBLEM"))
    ok &= abs(mu - 1) < 0.3
    t0 = time.time()
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH
    x = np.random.default_rng(1).normal(size=(500, 3)).astype("float32")
    t = np.random.default_rng(2).exponential(1.0 / np.exp(x[:, 0])).astype("float32")
    e = np.ones(500, dtype="float32")
    net = tt.practical.MLPVanilla(3, [8], 1, False, 0.0)
    m = CoxPH(net, tt.optim.Adam(0.01))
    m.fit(x, (t, e), batch_size=128, epochs=5, verbose=False)
    m.compute_baseline_hazards()
    print(f"  torch / pycox: {time.time() - t0:.1f}s")
    from sksurv.metrics import integrated_brier_score
    from sksurv.util import Surv
    y = Surv.from_arrays(np.ones(50, bool), np.linspace(1, 10, 50))
    ibs = integrated_brier_score(y, y, np.full((50, 3), 0.5), [3, 5, 7])
    print(f"  scikit-survival: integrated Brier score of a constant 0.5 = {ibs:.3f}")
    os.makedirs(Q, exist_ok=True)
    path = os.path.join(Q, "check_utf8.txt")
    # the scripts write such characters with the default encoding: same environment as the jobs
    code = subprocess.call([PY, "-c", f"open({path!r}, 'w').write('R\u0302 \u03ba \u2265 \u00e9')"], env=environment({}))
    print("  text files with special characters: " + ("ok" if code == 0 and os.path.exists(path) else "PROBLEM"))
    ok &= code == 0
    if os.path.exists(path):
        os.remove(path)
    free = shutil.disk_usage(".").free / 1e9
    print(f"  free disk space: {free:.0f} GB" + ("" if free > 8 else "  (at least 8 GB recommended)"))
    print(f"\nRecommended: python run_all.py --workers {default_workers()}")
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def default_workers():
    return max(1, min((os.cpu_count() or 2) - 1, 8))


def bundle():
    """Two zip files: 1) the summaries, logs and small result files; 2) the posterior draws of the full model."""
    day = datetime.date.today().isoformat()
    chains = ("results_eval/M3_train_a.npz", "results_eval/M3_train_b.npz", "results_eval/M3_full_a.npz", "results_eval/M3_full_b.npz",
              "results_cap9/M3_full_a.npz", "results_cap9/M3_full_b.npz")
    small_npz = ("results_eval/evaluation_boot.npz", "results_xai/hybrids.npz", "results_xai/effects.npz", "results_rl/agents_per_couple.npz")
    part1, part2 = [], [c for c in chains if os.path.exists(c)]
    roots = sorted({d.split("/")[0] for d in FOLDERS} | {Q})
    for top in roots:
        for base, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs if d not in ("cache", "__pycache__")]
            for fn in files:
                path = os.path.join(base, fn).replace(os.sep, "/")
                if not fn.endswith(".npz"):
                    part1.append(path)
                elif (path in small_npz or path.startswith("results_decision/") or (path.startswith("results_rl/") and "_seed" in fn)
                      or (path.startswith("results_xai/shap_apg_seed"))):
                    part1.append(path)
    for k, files in ((1, part1), (2, part2)):
        name = f"resultats_carup_{day}_{k}.zip"
        with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
            for path in files:
                z.write(path)
        print(f"{name}: {len(files)} files, {os.path.getsize(name) / 1e6:.1f} MB")
    print("These files contain results derived from the HRS data: keep them private (HRS conditions of use).")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--bundle", action="store_true")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--only", default="", help="comma-separated job names (their dependencies must be done)")
    a = ap.parse_args()
    J = jobs()
    if a.check:
        raise SystemExit(check())
    if a.list:
        for j in J:
            print(f"{'done' if is_done(j.name) else '    '}  {j.name:24s} {' '.join(j.args)}")
        raise SystemExit(0)
    if a.bundle:
        bundle()
        raise SystemExit(0)
    if a.only:
        keep = set(a.only.split(","))
        J = [j for j in J if j.name in keep]
        for j in J:
            j.after = [x for x in j.after if x in keep or not is_done(x)]
    raise SystemExit(run(J, a.workers))
