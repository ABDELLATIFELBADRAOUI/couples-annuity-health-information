"""
ppo_long.py -- one longer run of PPO, for the treatment of the exploration noise that did best.

Rule, fixed before the runs: among the two treatments (learned noise, files ppo_seed*; annealed noise,
files ppoa_seed*), take the one with the smaller gap to the optimum on the MONITORING couples
(training couples) at the end of training, averaged over its seeds 0 to 4. The held-out couples play no
part in the choice. The run has three times the iterations of the others and is saved as seed 5.

Usage: python ppo_long.py        (folder results_rl, or the environment variable CARUP_RL_OUT)
"""
import glob
import json
import os
import subprocess
import sys

OUT = os.environ.get("CARUP_RL_OUT", "results_rl")
SEEDS, LONG_SEED, FACTOR = range(5), 5, 3
final = {}
for kind in ("ppo", "ppoa"):
    infos = [json.load(open(f"{OUT}/{kind}_seed{s}.json")) for s in SEEDS]
    assert len({(i["iterations"], i["lives_per_iteration"], i["width"]) for i in infos}) == 1, "the runs of one treatment differ"
    final[kind] = (sum(i["curve"][-1]["gap_mean"] for i in infos) / len(infos), infos[0])
    print(f"{kind}: gap on the monitoring couples at the end of training, mean over {len(infos)} seeds: {final[kind][0]:.4f} %")
kind = min(final, key=lambda k: final[k][0])
info = final[kind][1]
cmd = [sys.executable, "-u", "rl_ppo.py", str(LONG_SEED), str(FACTOR * info["iterations"]), str(info["lives_per_iteration"]),
       str(info["width"]), "25"] + (["-3.0"] if kind == "ppoa" else [])
print("longer run:", kind, " ".join(cmd), flush=True)
json.dump(dict(chosen=kind, monitoring_gap={k: v[0] for k, v in final.items()}, command=" ".join(cmd)),
          open(f"{OUT}/ppo_long_choice.json", "w"), indent=1)
if "--dry-run" in sys.argv:
    raise SystemExit(0)
raise SystemExit(subprocess.call(cmd))
