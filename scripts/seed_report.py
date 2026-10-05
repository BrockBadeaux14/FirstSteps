"""Markdown table of key numbers for several runs (after sprinter.replay).

  uv run python scripts/seed_report.py runs/final-seed0 runs/final-seed1 runs/final-seed2
"""

import json
from pathlib import Path
import sys


def main(run_dirs):
  rows = []
  for run in map(Path, run_dirs):
    s = json.loads((run / "summary.json").read_text())
    t = json.loads((run / "train_stats.json").read_text())
    rows.append((s, t))
  print("| Seed | Wall time | Compile | Steps/s | Distance in 10 s (mean, min-max) | Falls | Passed | Alternation | Stance imbalance | Foot slip | Peak GPU mem (XLA) | Pass |")
  print("|---|---|---|---|---|---|---|---|---|---|---|---|")
  for s, t in rows:
    d = s["final_distance_m"]
    e = s["outputs"]["after"]["eval"]
    print(
        f"| {s['seed']} | {s['wall_time_s'] / 60:.1f} min | {s['compile_s']:.0f} s | "
        f"{s['steps_per_s']:,} | {d['mean']:.1f} m ({d['min']:.1f}-{d['max']:.1f}) | "
        f"{s['eval_falls']}/10 | {s['eval_passed_episodes']}/10 | {s['mean_alternation']:.2f} | "
        f"{e['mean_duty_imbalance']:.2f} | {e['mean_foot_slip_mps']:.2f} m/s | "
        f"{s['xla_peak_gpu_memory_gb']:.2f} GB | {'yes' if s['pass'] else 'no'} |"
    )
  s0 = rows[0][0]
  print(f"\nbackend {s0['backend']}, {s0['timesteps']:,} steps, GPU {s0['gpu'].get('name')}, "
        f"driver {s0['gpu'].get('driver')}")


if __name__ == "__main__":
  main(sys.argv[1:])
