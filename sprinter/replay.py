"""Replays and outputs for a finished training run.

  uv run python -m sprinter.replay --run runs/<id>

Writes into the run directory:
  replay.json / replay.mp4                 final policy, one 10 s episode
  before/replay.json / before/replay.mp4   early checkpoint (~10% of training)
  before_after.mp4                         the two side by side
  reward_curve.png                         from metrics.jsonl
  summary.json                             versions, GPU, timing, distance, pass/fail

replay.json is what the phone app draws, so it carries everything needed to
draw the runner without physics or kinematics: a skeleton (written once) and
every body's 2D world pose per control step.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import argparse  # pylint: disable=g-import-not-at-top,wrong-import-position
import json
from pathlib import Path
from typing import Any

from brax.training import checkpoint as brax_checkpoint
from brax.training import types as brax_types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sprinter import config as cfglib
from sprinter import evaluate
from sprinter import plots
from sprinter import system_info
from sprinter import train as trainlib

REPLAY_FORMAT = "sprinter-replay/1"
EVAL_EPISODES = 10
EARLY_FRACTION = 0.10
AFTER_FALL_S = 1.0  # keep showing the body this long after a fall
MAX_WALL_TIME_S = 30 * 60


# -----------------------------------------------------------------------------
# Checkpoints and policies


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
  ckpts = [(int(p.name), p) for p in (run_dir / "checkpoints").iterdir() if p.name.isdigit()]
  return sorted(ckpts)


def final_checkpoint(run_dir: Path) -> tuple[int, Path]:
  final = sorted((int(p.name), p) for p in (run_dir / "final").iterdir() if p.name.isdigit())
  return final[-1] if final else list_checkpoints(run_dir)[-1]


def load_policy(cfg: cfglib.TrainConfig, ckpt: Path, obs_size: int, action_size: int):
  params = brax_checkpoint.load(ckpt)
  normalize = (running_statistics.normalize if cfg.ppo.normalize_observations
               else brax_types.identity_observation_preprocessor)
  network = trainlib.make_network_factory(cfg.network)(
      obs_size, action_size, preprocess_observations_fn=normalize
  )
  return ppo_networks.make_inference_fn(network)(params, deterministic=True)


# -----------------------------------------------------------------------------
# replay.json


def _hex(rgba) -> str:
  return "#" + "".join(f"{int(round(255 * c)):02x}" for c in rgba[:3])


def _xz(v) -> list[float]:
  return [round(float(v[0]), 3), round(float(v[2]), 3)]


def skeleton(m: mujoco.MjModel) -> list[dict[str, Any]]:
  """One entry per body (world excluded), in MuJoCo body order = pose order."""
  bodies = []
  for b in range(1, m.nbody):
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
    assert len(geoms) == 1, "every body has exactly one geom"
    g = geoms[0]
    pos = m.geom_pos[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
      mat = np.zeros(9)
      mujoco.mju_quat2Mat(mat, m.geom_quat[g])
      axis = mat.reshape(3, 3)[:, 2] * m.geom_size[g][1]
      geom = {"type": "capsule", "from": _xz(pos - axis), "to": _xz(pos + axis),
              "radius": round(float(m.geom_size[g][0]), 3)}
    elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
      geom = {"type": "sphere", "center": _xz(pos), "radius": round(float(m.geom_size[g][0]), 3)}
    else:
      raise ValueError(f"unsupported geom type for {m.geom(g).name}")
    rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    name = m.body(b).name
    parent = m.body_parentid[b]
    bodies.append({
        "name": name,
        "parent": m.body(parent).name if parent > 0 else None,
        "offset": _xz(m.body_pos[b]),  # body origin in the parent's frame
        "side": "left" if name.startswith("left_") else "right" if name.startswith("right_") else "center",
        "depth": round(float(m.body_pos[b][1] if parent == 1 else 0.0), 3),
        "geom": geom,
        "color": _hex(rgba),
    })
  # Children inherit their limb's depth (y offset) so the app can sort draw order.
  for body in bodies:
    if body["parent"] not in (None, "torso"):
      body["depth"] = next(p["depth"] for p in bodies if p["name"] == body["parent"])
  return bodies


def build_replay(
    m: mujoco.MjModel, traj: dict[str, np.ndarray], episode: dict[str, Any], i: int,
    *, dt: float, n_frames: int, cfg: cfglib.TrainConfig, step: int,
) -> dict[str, Any]:
  # float64 before rounding, so JSON gets "0.123" rather than float32 noise.
  poses = np.round(traj["poses"][:n_frames, i].astype(np.float64), 3)
  frames = [{"t": round(k * dt, 3), "poses": poses[k].tolist()} for k in range(n_frames)]
  skel = skeleton(m)
  # Back to front as seen from the camera (camera sits at -y): left limbs first.
  draw_order = [b["name"] for b in sorted(skel, key=lambda b: -b["depth"])]
  return {
      "meta": {
          "format": REPLAY_FORMAT,
          "level": cfg.level,
          "dt": dt,
          "fps": round(1.0 / dt),
          "num_frames": n_frames,
          "units": {"length": "m", "angle": "rad", "time": "s"},
          "axes": "x forward (right on screen), z up; the runner moves toward +x",
          "pose": "[x, z, angle] per body, in skeleton order; body frame origin in world coordinates",
          "angle_convention": (
              "counter-clockwise positive with +x right and +z up; a point (u, v) in "
              "the body frame is at (x + u cos a - v sin a, z + u sin a + v cos a)"
          ),
          "skeleton_geometry": "geom from/to/center are in the body frame, [x, z]",
          "draw_order": draw_order,
          "floor_z": 0.0,
          "marker_every_m": 10,
          "distance_m": episode["distance_m"],
          "fell": episode["fell"],
          "end_s": episode["seconds"],
          "mean_speed_mps": episode["mean_speed_mps"],
          "episode_seconds": cfg.sim.episode_seconds,
          "checkpoint_step": step,
          "seed": cfg.seed,
          "config_hash": cfg.config_hash(),
      },
      "skeleton": skel,
      "frames": frames,
  }


def write_json(path: Path, obj: Any) -> int:
  path.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(obj, separators=(",", ":"))
  path.write_text(text)
  return len(text.encode())


# -----------------------------------------------------------------------------
# Video


def render(m: mujoco.MjModel, qpos: np.ndarray, width: int, height: int) -> list[np.ndarray]:
  d = mujoco.MjData(m)
  with mujoco.Renderer(m, height=height, width=width) as r:
    frames = []
    for q in qpos:
      d.qpos[:] = q
      mujoco.mj_forward(m, d)
      r.update_scene(d, camera="track")
      frames.append(r.render().copy())
  return frames


def _caption(frame: np.ndarray, text: str) -> np.ndarray:
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img)
  try:
    font = ImageFont.load_default(size=max(14, frame.shape[0] // 20))
  except TypeError:
    font = ImageFont.load_default()
  draw.rectangle([0, 0, frame.shape[1], frame.shape[0] // 11], fill=(252, 252, 251))
  draw.text((12, 4), text, fill=(11, 11, 11), font=font)
  return np.asarray(img)


def side_by_side(a: list[np.ndarray], b: list[np.ndarray], label_a: str, label_b: str):
  n = max(len(a), len(b))
  a = a + [a[-1]] * (n - len(a))
  b = b + [b[-1]] * (n - len(b))
  return [np.concatenate([_caption(x, label_a), _caption(y, label_b)], axis=1) for x, y in zip(a, b)]


# -----------------------------------------------------------------------------
# Main


def episode_frames(episode: dict[str, Any], dt: float, total_steps: int) -> int:
  steps = int(round(episode["seconds"] / dt))
  if episode["fell"]:
    steps += int(round(AFTER_FALL_S / dt))
  return min(steps, total_steps) + 1  # +1 for the reset frame


def median_episode(result: dict[str, Any]) -> int:
  d = np.array([e["distance_m"] for e in result["episodes"]])
  return int(np.argsort(d)[len(d) // 2])


def make_replays(run_dir: Path, *, video: bool = True, width: int = 1280, height: int = 720) -> dict[str, Any]:
  cfg = cfglib.load_config(run_dir / "config.json")
  env = trainlib.make_env(cfg, EVAL_EPISODES)
  m, dt, T = env.mj_model, env.dt, cfg.sim.episode_length
  obs_size, act_size = int(env.observation_size), env.action_size

  final_step, final_path = final_checkpoint(run_dir)
  ckpts = list_checkpoints(run_dir)
  early_step, early_path = min(ckpts, key=lambda c: abs(c[0] - EARLY_FRACTION * final_step))

  outputs: dict[str, Any] = {}
  results = {}
  videos = {}
  for tag, step, path, out_dir in (
      ("after", final_step, final_path, run_dir),
      ("before", early_step, early_path, run_dir / "before"),
  ):
    policy = load_policy(cfg, path, obs_size, act_size)
    result = evaluate.evaluate(env, policy, EVAL_EPISODES, seed=cfg.seed + 1000)
    results[tag] = result
    traj = result["_trajectories"]
    i = median_episode(result)
    episode = result["episodes"][i]
    n = episode_frames(episode, dt, T)
    replay = build_replay(m, traj, episode, i, dt=dt, n_frames=n, cfg=cfg, step=step)
    size = write_json(out_dir / "replay.json", replay)
    print(f"[{tag}] step {step:,}: episode {i} -> {episode['distance_m']:.1f} m, "
          f"fell={episode['fell']}; replay.json {size / 1024:.0f} KB")
    outputs[tag] = {
        "checkpoint_step": step,
        "episode": i,
        "distance_m": episode["distance_m"],
        "fell": episode["fell"],
        "replay_json": str(out_dir / "replay.json"),
        "replay_json_kb": round(size / 1024, 1),
        "eval": {k: v for k, v in evaluate.strip_trajectories(result).items() if k != "episodes"},
    }
    if video:
      frames = render(m, traj["qpos"][:n, i], width, height)
      mediapy.write_video(out_dir / "replay.mp4", frames, fps=round(1 / dt))
      outputs[tag]["replay_mp4"] = str(out_dir / "replay.mp4")
      videos[tag] = (frames, step)

  if video:
    (fa, sa), (fb, sb) = videos["before"], videos["after"]
    small = lambda fr: [np.asarray(Image.fromarray(x).resize((width // 2, height // 2))) for x in fr]
    pair = side_by_side(small(fa), small(fb),
                        f"Before: {sa / 1e6:.0f}M steps, {outputs['before']['distance_m']:.1f} m",
                        f"After: {sb / 1e6:.0f}M steps, {outputs['after']['distance_m']:.1f} m")
    mediapy.write_video(run_dir / "before_after.mp4", pair, fps=round(1 / dt))
    outputs["before_after_mp4"] = str(run_dir / "before_after.mp4")

  plots.plot_reward_curve(run_dir / "metrics.jsonl", run_dir / "reward_curve.png",
                          title=f"Sprint training: {run_dir.name}")
  outputs["reward_curve_png"] = str(run_dir / "reward_curve.png")
  (run_dir / "eval.json").write_text(
      json.dumps(evaluate.strip_trajectories(results["after"]), indent=2) + "\n"
  )
  summary = build_summary(run_dir, cfg, results["after"], outputs)
  (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  return summary


def build_summary(run_dir: Path, cfg, result, outputs) -> dict[str, Any]:
  stats_path = run_dir / "train_stats.json"
  stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
  wall = stats.get("wall_time_s")
  checks = {
      "wall_time_le_30_min": wall is not None and wall <= MAX_WALL_TIME_S,
      "distance_ge_20m_in_8_of_10": result["distance_ok"],
      "alternating_gait": result["gait_ok"],
  }
  return {
      "run": run_dir.name,
      "level": cfg.level,
      "seed": cfg.seed,
      "config_hash": cfg.config_hash(),
      "backend": cfg.sim.impl,
      "gpu": stats.get("gpu") or system_info.gpu_info(),
      "versions": stats.get("versions") or system_info.package_versions(),
      "timesteps": stats.get("timesteps"),
      "wall_time_s": wall,
      "compile_s": stats.get("compile_s"),
      "train_s": stats.get("train_s"),
      "eval_s": stats.get("eval_s"),
      "steps_per_s": stats.get("steady_sps"),
      "xla_peak_gpu_memory_gb": stats.get("xla_peak_gpu_memory_gb"),
      "final_distance_m": result["distance_m"],
      "final_mean_speed_mps": result["mean_speed_mps"],
      "eval_passed_episodes": result["passed_episodes"],
      "eval_falls": result["falls"],
      "mean_alternation": result["mean_alternation"],
      "level_score_m": outputs["after"]["distance_m"],
      "checks": checks,
      "pass": all(checks.values()),
      "outputs": outputs,
  }


def main() -> None:
  p = argparse.ArgumentParser(description="Make replays, videos, plots and summary for a run.")
  p.add_argument("--run", required=True, help="run directory, e.g. runs/20261005-120000-sprint-s0")
  p.add_argument("--no-video", action="store_true")
  p.add_argument("--width", type=int, default=1280)
  p.add_argument("--height", type=int, default=720)
  a = p.parse_args()
  trainlib.require_gpu()
  summary = make_replays(Path(a.run), video=not a.no_video, width=a.width, height=a.height)
  print(json.dumps({k: summary[k] for k in (
      "final_distance_m", "eval_passed_episodes", "eval_falls", "mean_alternation", "checks", "pass")},
      indent=2))


if __name__ == "__main__":
  main()
