"""Training curves from metrics.jsonl.

  uv run python -m sprinter.plots --run runs/<id>      # writes reward_curve.png

Four small multiples on a shared x-axis (no dual axes): total episode reward,
reward by named component, distance per episode, and fall rate. The data table
behind every point is metrics.jsonl itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top,wrong-import-position

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"  # single-series panels (categorical slot 1)
# Reward components keep a fixed color each (slots 2-5), whichever are present.
COMPONENT_COLORS = {
    "forward_velocity": "#eb6834",
    "alive": "#1baf7a",
    "upright": "#eda100",
    "control_cost": "#e87ba4",
}


def load_metrics(path: Path) -> list[dict]:
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _style(ax, title: str, ylabel: str) -> None:
  ax.set_facecolor(SURFACE)
  ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
  ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
  ax.grid(True, axis="y", color=GRID, linewidth=1, linestyle="-")
  ax.set_axisbelow(True)
  for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
  ax.spines["bottom"].set_color(AXIS)
  ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=4)


def _line(ax, x, y, color, label=None, fmt="{:.0f}", end_label=True):
  ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=label)
  ax.plot(x[-1:], y[-1:], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
  if end_label:
    text = fmt.format(y[-1]) if label is None else f"{label} {fmt.format(y[-1])}"
    ax.annotate(text, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK_2)


def plot_reward_curve(metrics_path: Path, out_path: Path, title: str = "") -> Path:
  rows = load_metrics(metrics_path)
  x = [r["timesteps"] / 1e6 for r in rows]
  fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True, facecolor=SURFACE)
  (ax_r, ax_c), (ax_d, ax_f) = axes

  _style(ax_r, "Episode reward (eval)", "reward per episode")
  _line(ax_r, x, [r["eval_reward"] for r in rows], SERIES)

  _style(ax_c, "Reward by component (eval)", "reward per episode")
  present = [k for k in COMPONENT_COLORS
             if any(abs(r["reward_components"].get(k) or 0.0) > 1e-9 for r in rows)]
  for k in present:
    _line(ax_c, x, [r["reward_components"][k] for r in rows], COMPONENT_COLORS[k], label=k)
  ax_c.axhline(0, color=AXIS, linewidth=1)
  ax_c.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2)

  _style(ax_d, "Distance in one episode", "meters")
  _line(ax_d, x, [r["distance_m"] for r in rows], SERIES, fmt="{:.1f} m")
  ax_d.axhline(20, color=MUTED, linewidth=1)
  ax_d.annotate("20 m bar", (x[0], 20), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=MUTED)

  _style(ax_f, "Falls", "share of eval episodes")
  _line(ax_f, x, [100 * r["fall_rate"] for r in rows], SERIES, fmt="{:.0f}%")
  ax_f.set_ylim(-5, 105)

  for ax in (ax_d, ax_f):
    ax.set_xlabel("environment steps (millions)", color=INK_2, fontsize=9)
  for ax in axes.flat:
    ax.margins(x=0.12)
  if title:
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150, facecolor=SURFACE)
  plt.close(fig)
  return out_path


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--run", required=True)
  a = p.parse_args()
  run = Path(a.run)
  out = plot_reward_curve(run / "metrics.jsonl", run / "reward_curve.png", title=run.name)
  print(out)


if __name__ == "__main__":
  main()
