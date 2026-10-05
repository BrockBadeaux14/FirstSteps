"""Does changing only a hyperparameter force Brax PPO to recompile?

  uv run python -m sprinter.recompile_test --config configs/sprint_default.json [--impl jax]

Runs short trainings (2 PPO updates each) and records JAX's compile events:
  process 1, no persistent cache:   lr, same lr again, lr changed
  process 2, fresh XLA cache dir:   lr, same lr again, lr changed
  process 3, same cache dir:        lr with a different seed (a new job)
Results go to runs/recompile_test_<impl>.json. This decides the future server
design: whether a GPU worker can reuse compiled programs across jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

SHORT_UPDATES = 2


def _load(config_path: str, impl: str):
  from sprinter import config as cfglib  # pylint: disable=g-import-not-at-top
  cfg = cfglib.load_config(config_path)
  return cfg.model_copy(update={"sim": cfg.sim.model_copy(update={"impl": impl})})


def _worker(config_path: str, impl: str, cases: list[dict], cache_dir: str) -> None:
  from brax.training.agents.ppo import train as ppo  # pylint: disable=g-import-not-at-top
  from sprinter import train as trainlib  # pylint: disable=g-import-not-at-top

  trainlib.require_gpu()
  trainlib.enable_compile_cache(cache_dir or None)
  base = _load(config_path, impl)
  results = []
  for case in cases:
    ppo_cfg = base.ppo.model_copy(update={
        "num_timesteps": SHORT_UPDATES * base.ppo.env_steps_per_update,
        "num_evals": 2,
        "learning_rate": case["learning_rate"],
    })
    cfg = base.model_copy(update={"ppo": ppo_cfg, "seed": case.get("seed", base.seed)})
    timer = trainlib.COMPILE_TIMER
    timer.reset()
    timer.enabled = True
    t0 = time.monotonic()
    first_eval = {}

    def progress(step, unused_metrics, t0=t0, first_eval=first_eval):
      if step == 0:
        first_eval["s"] = round(time.monotonic() - t0, 1)

    env = trainlib.make_env(cfg, cfg.ppo.num_envs)
    eval_env = trainlib.make_env(cfg, cfg.ppo.num_eval_envs)
    ppo.train(environment=env, eval_env=eval_env, progress_fn=progress, **trainlib.ppo_kwargs(cfg))
    timer.enabled = False
    results.append({
        **case,
        "wall_s": round(time.monotonic() - t0, 1),
        "time_to_first_eval_s": first_eval.get("s"),
        **timer.summary(),
    })
    print(json.dumps(results[-1]), flush=True)
  print("RESULTS " + json.dumps(results), flush=True)


def _spawn(config: str, impl: str, cases: list[dict], cache_dir: str) -> list[dict]:
  cmd = [sys.executable, "-m", "sprinter.recompile_test", "--config", config, "--impl", impl,
         "--worker", json.dumps(cases), "--cache-dir", cache_dir]
  out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
  line = next(l for l in out.splitlines() if l.startswith("RESULTS "))
  return json.loads(line[len("RESULTS "):])


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--config", default="configs/sprint_default.json")
  p.add_argument("--impl", default="warp", choices=["warp", "jax"])
  p.add_argument("--worker", default=None, help=argparse.SUPPRESS)
  p.add_argument("--cache-dir", default="", help=argparse.SUPPRESS)
  a = p.parse_args()
  if a.worker:
    _worker(a.config, a.impl, json.loads(a.worker), a.cache_dir)
    return

  cfg = _load(a.config, a.impl)
  lr = cfg.ppo.learning_rate
  in_process = [
      {"case": "baseline", "learning_rate": lr},
      {"case": "same config again", "learning_rate": lr},
      {"case": "learning_rate changed", "learning_rate": lr / 2},
  ]
  cache = Path(tempfile.mkdtemp(prefix="sprinter_xla_cache_"))
  try:
    report = {
        "impl": a.impl,
        "short_training_steps": SHORT_UPDATES * cfg.ppo.env_steps_per_update,
        "no_persistent_cache": _spawn(a.config, a.impl, in_process, ""),
        "persistent_cache_same_process": _spawn(a.config, a.impl, in_process, str(cache)),
        "persistent_cache_new_process": _spawn(
            a.config, a.impl,
            [{"case": "new job, other seed", "learning_rate": lr, "seed": 1}], str(cache),
        ),
    }
  finally:
    shutil.rmtree(cache, ignore_errors=True)
  out = Path(f"runs/recompile_test_{a.impl}.json")
  out.parent.mkdir(exist_ok=True)
  out.write_text(json.dumps(report, indent=2) + "\n")
  print(f"impl={a.impl}, {report['short_training_steps']:,} training steps per case")
  print(f"{'scenario':32s} {'case':24s} {'wall':>7s} {'compile':>8s} {'xla':>5s} {'hits':>5s}")
  for scenario in ("no_persistent_cache", "persistent_cache_same_process", "persistent_cache_new_process"):
    for r in report[scenario]:
      print(f"{scenario:32s} {r['case']:24s} {r['wall_s']:6.1f}s {r['total_s']:7.1f}s "
            f"{r['xla_compiles']:5d} {r['persistent_cache_hits']:5d}")
  print(f"wrote {out}")


if __name__ == "__main__":
  main()
