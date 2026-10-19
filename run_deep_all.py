"""run_deep_all.py -- on every split: the comparison models that are missing (the copula model, DeepSurv and
DeepHit), then the evaluation table with all the models.

Run by run_all.py after the fits of the split it evaluates.
Usage: python run_deep_all.py [main] [seeds ...]      (default: main 1 2 3 4)
"""
import os
import subprocess
import sys
import time

which = sys.argv[1:] or ["main", "1", "2", "3", "4"]
jobs = [("results_eval", {}) if w == "main" else
        (f"results_seeds/seed_{w}", {"CARUP_SPLIT_SEED": w, "CARUP_MODELS": "M0,M3"}) for w in which]
for res, extra in jobs:
    chains = [f"{res}/{v}_train_{tag}.npz" for v in extra.get("CARUP_MODELS", "").split(",") if v for tag in "ab"]
    while not all(os.path.exists(f) for f in chains):                      # split still being fitted
        time.sleep(30)
    time.sleep(20 if chains else 0)
    env = dict(os.environ, CARUP_RESULTS_DIR=res, **extra)
    t0 = time.time()
    if not os.path.exists(f"{res}/copula_predictions.npz"):
        with open(f"{res}/copula.log", "w", encoding="utf-8") as f:
            code = subprocess.call([sys.executable, "-u", "copula_baseline.py"], stdout=f, stderr=subprocess.STDOUT,
                                   env=dict(env, CARUP_COPULA_RECOVERY="1" if not extra else "0"))
        print(f"{res}: copula model, exit code {code}, {time.time() - t0:.0f}s", flush=True)
        assert code == 0, f"the copula model failed: see {res}/copula.log"
    t0 = time.time()
    if not os.path.exists(f"{res}/deep_predictions.npz"):
        with open(f"{res}/deep_models.log", "w", encoding="utf-8") as f:
            code = subprocess.call([sys.executable, "-u", "deep_models.py"], stdout=f, stderr=subprocess.STDOUT, env=env)
        print(f"{res}: deep models, exit code {code}, {time.time() - t0:.0f}s", flush=True)
        assert code == 0, f"the deep models failed: see {res}/deep_models.log"
    t0 = time.time()
    with open(f"{res}/evaluation.txt.tmp", "w", encoding="utf-8") as f, open(f"{res}/evaluation.err", "w", encoding="utf-8") as g:
        code = subprocess.call([sys.executable, "evaluate.py"], stdout=f, stderr=g, env=env)
    if code == 0:
        os.replace(f"{res}/evaluation.txt.tmp", f"{res}/evaluation.txt")
    print(f"{res}: evaluation, exit code {code}, {time.time() - t0:.0f}s", flush=True)
    assert code == 0, f"the evaluation failed: see {res}/evaluation.err"
