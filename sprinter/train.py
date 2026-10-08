"""Config-driven PPO training for the Sprint level.

  uv run python -m sprinter.train --config configs/sprint_default.json

Writes runs/<timestamp>/:
  config.json        the validated config (the contract the app server sends)
  metrics.jsonl      one line per eval: the future live-stats stream
  checkpoints/       Brax/orbax checkpoints, one per eval (step number in the name)
  train_stats.json   versions, GPU, backend, compile vs training time, steps/s
  eval.json          10 deterministic evaluation episodes of the final policy
"""

from __future__ import annotations

import os

# These must be set before JAX initializes the GPU backend.
# Grow GPU memory on demand: Windows already holds ~2.6 GB of the 8 GB card.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Full float32 matmuls: TF32 on Ampere GPUs hurts RL stability (Playground README).
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import argparse  # pylint: disable=g-import-not-at-top,wrong-import-position
import collections
import datetime
import functools
import json
from pathlib import Path
import time
from typing import Any, Callable

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax import linen
import jax
from jax import monitoring
from mujoco_playground import wrapper
import numpy as np

from sprinter import config as cfglib
from sprinter import evaluate
from sprinter import system_info
from sprinter.envs import sprint

ACTIVATION_FNS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "relu": linen.relu,
    "tanh": linen.tanh,
    "swish": linen.swish,
    "elu": linen.elu,
}
CONTACTS_PER_WORLD = 16  # Warp contact budget per parallel sim (naconmax / num_envs)
RUNS_DIR = Path("runs")


# -----------------------------------------------------------------------------
# Config -> env / networks / ppo.train arguments


def check_supported(cfg: cfglib.TrainConfig) -> None:
  """Any valid config can be sent, but this worker only trains PPO with Adam so far."""
  if cfg.trainer.type != "ppo":
    raise NotImplementedError(f"this worker only trains ppo so far, not {cfg.trainer.type}")
  if cfg.optimizer.type != "adam":
    raise NotImplementedError(
        f"Brax's PPO always uses Adam, so this worker can't train ppo with {cfg.optimizer.type} yet"
    )


def make_env(cfg: cfglib.TrainConfig, num_envs: int) -> sprint.Sprint:
  env_cfg = sprint.default_config()
  env_cfg.impl = cfg.sim.impl
  env_cfg.ctrl_dt = cfg.sim.ctrl_dt
  env_cfg.sim_dt = cfg.sim.sim_dt
  env_cfg.episode_length = cfg.sim.episode_length
  env_cfg.naconmax = num_envs * CONTACTS_PER_WORLD
  for name, weight in cfg.level.reward_weights.model_dump().items():
    env_cfg.reward_weights[name] = weight
  return sprint.Sprint(env_cfg)


def make_network_factory(brain: cfglib.MLPBrain):
  return functools.partial(
      ppo_networks.make_ppo_networks,
      policy_hidden_layer_sizes=tuple(brain.policy_hidden),
      value_hidden_layer_sizes=tuple(brain.value_hidden),
      activation=ACTIVATION_FNS[brain.activation],
  )


def ppo_kwargs(cfg: cfglib.TrainConfig) -> dict[str, Any]:
  p = cfg.trainer
  return dict(
      num_timesteps=p.num_timesteps,
      num_envs=p.num_envs,
      batch_size=p.batch_size,
      num_minibatches=p.num_minibatches,
      unroll_length=p.unroll_length,
      num_updates_per_batch=p.num_updates_per_batch,
      learning_rate=cfg.optimizer.learning_rate,
      discounting=p.discounting,
      entropy_cost=p.entropy_cost,
      clipping_epsilon=p.clipping_epsilon,
      reward_scaling=p.reward_scaling,
      normalize_observations=p.normalize_observations,
      num_evals=p.num_evals,
      num_eval_envs=p.num_eval_envs,
      episode_length=cfg.sim.episode_length,
      action_repeat=1,
      num_resets_per_eval=0,
      deterministic_eval=True,
      seed=cfg.seed,
      network_factory=make_network_factory(cfg.brain),
      wrap_env_fn=wrapper.wrap_for_brax_training,
  )


# -----------------------------------------------------------------------------
# Timing helpers


class CompileTimer:
  """Sums JAX's own compile-event durations (trace, lower to MLIR, XLA compile)."""

  EVENTS = {
      "/jax/core/compile/jaxpr_trace_duration": "trace_s",
      "/jax/core/compile/jaxpr_to_mlir_module_duration": "lower_s",
      "/jax/core/compile/backend_compile_duration": "backend_compile_s",
  }

  def __init__(self):
    self.totals = collections.defaultdict(float)
    self.counts = collections.Counter()
    self.cache_hits = 0
    self.enabled = False
    monitoring.register_event_duration_secs_listener(self._on_duration)
    monitoring.register_event_listener(self._on_event)

  def _on_duration(self, event, duration, **unused):
    if self.enabled and event in self.EVENTS:
      self.totals[self.EVENTS[event]] += duration
      self.counts[self.EVENTS[event]] += 1

  def _on_event(self, event, **unused):
    if self.enabled and event == "/jax/compilation_cache/cache_hits":
      self.cache_hits += 1

  def reset(self):
    self.totals.clear()
    self.counts.clear()
    self.cache_hits = 0

  def summary(self) -> dict[str, Any]:
    out = {k: round(v, 2) for k, v in self.totals.items()}
    out["total_s"] = round(sum(self.totals.values()), 2)
    out["xla_compiles"] = self.counts["backend_compile_s"]
    out["persistent_cache_hits"] = self.cache_hits
    return out


COMPILE_TIMER = CompileTimer()


def enable_compile_cache(path: str | None) -> None:
  if not path:
    return
  Path(path).mkdir(parents=True, exist_ok=True)
  jax.config.update("jax_compilation_cache_dir", str(Path(path).resolve()))
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


def require_gpu() -> None:
  """Never train silently on the CPU."""
  backend = jax.default_backend()
  if backend != "gpu":
    raise SystemExit(
        f"JAX backend is '{backend}', not 'gpu'. Refusing to train on the CPU. "
        "Check `nvidia-smi` and `python -c 'import jax; print(jax.devices())'`."
    )


# -----------------------------------------------------------------------------
# Training


def train(
    cfg: cfglib.TrainConfig,
    run_dir: Path,
    *,
    save_checkpoints: bool = True,
    final_eval_episodes: int = 10,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
  """Trains one policy. Returns the train stats (also written to run_dir)."""
  check_supported(cfg)
  require_gpu()
  run_dir = run_dir.resolve()  # orbax only accepts absolute checkpoint paths
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "config.json").write_text(
      json.dumps(cfg.model_dump(mode="json"), indent=2) + "\n"
  )
  metrics_path = run_dir / "metrics.jsonl"
  metrics_path.write_text("")

  env = make_env(cfg, cfg.trainer.num_envs)
  eval_env = make_env(cfg, cfg.trainer.num_eval_envs)
  kwargs = ppo_kwargs(cfg)
  dt = env.dt

  t0 = time.monotonic()
  progress_rows: list[dict[str, Any]] = []

  def progress_fn(step: int, metrics: dict[str, Any]) -> None:
    m = {k: float(np.asarray(v)) for k, v in metrics.items() if np.ndim(v) == 0}
    row = {
        "timesteps": int(step),
        "wall_time_s": round(time.monotonic() - t0, 2),
        "eval_reward": m.get("eval/episode_reward"),
        "reward_components": {
            k: m.get(f"eval/episode_reward/{k}") for k in sprint.REWARD_TERMS
        },
        "distance_m": m.get("eval/episode_distance_x"),
        "mean_speed_mps": m.get("eval/episode_speed_per_step"),
        "fall_rate": m.get("eval/episode_fell"),
        "episode_length": m.get("eval/avg_episode_length"),
        "episode_seconds": (m.get("eval/avg_episode_length") or 0.0) * dt,
        "train_sps": m.get("training/sps"),
        "train_walltime_s": m.get("training/walltime"),
        "eval_time_s": m.get("eval/epoch_eval_time"),
    }
    progress_rows.append(row)
    with metrics_path.open("a") as f:
      f.write(json.dumps(row) + "\n")
    log(
        f"[{row['wall_time_s']:7.1f}s] steps {step:>11,}  "
        f"reward {row['eval_reward']:8.1f}  distance {row['distance_m']:6.2f} m  "
        f"fall rate {row['fall_rate']:.2f}"
    )

  ckpt_dir = run_dir / "checkpoints"
  log(f"Training {cfg.level.type} with {cfg.trainer.type} on {jax.devices()[0].device_kind} "
      f"(impl={cfg.sim.impl}, {cfg.trainer.num_envs} envs, "
      f"{cfg.trainer.effective_timesteps:,} steps) -> {run_dir}")
  COMPILE_TIMER.reset()
  COMPILE_TIMER.enabled = True
  make_inference_fn, params, _ = ppo.train(
      environment=env,
      eval_env=eval_env,
      progress_fn=progress_fn,
      save_checkpoint_path=ckpt_dir if save_checkpoints else None,
      **kwargs,
  )
  train_wall = time.monotonic() - t0
  COMPILE_TIMER.enabled = False
  total_steps = progress_rows[-1]["timesteps"]

  # Save the final params on their own as well (Brax also saved them as the
  # last checkpoint).
  if save_checkpoints:
    ckpt_config = ppo_checkpoint.network_config(
        observation_size=env.observation_size,
        action_size=env.action_size,
        normalize_observations=cfg.trainer.normalize_observations,
        network_factory=kwargs["network_factory"],
    )
    ppo_checkpoint.save(run_dir / "final", total_steps, params, ckpt_config)

  timing = split_timing(progress_rows, cfg.trainer.env_steps_per_update)
  mem = jax.devices()[0].memory_stats() or {}
  stats = {
      "run_dir": str(run_dir),
      "config_hash": cfg.config_hash(),
      "level": cfg.level.type,
      "trainer": cfg.trainer.type,
      "seed": cfg.seed,
      "backend": cfg.sim.impl,
      "timesteps": total_steps,
      "wall_time_s": round(train_wall, 1),
      **timing,
      "compile_events": COMPILE_TIMER.summary(),
      "xla_peak_gpu_memory_gb": round(mem.get("peak_bytes_in_use", 0) / 1e9, 3),
      "gpu": system_info.gpu_info(),
      "versions": system_info.package_versions(),
      "env": {
          "obs_size": int(env.observation_size),
          "action_size": int(env.action_size),
          "ctrl_dt": env.dt,
          "sim_dt": env.sim_dt,
          "episode_length": cfg.sim.episode_length,
      },
  }
  (run_dir / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
  log(
      f"Done: {total_steps:,} steps in {train_wall / 60:.1f} min "
      f"(startup+compile {timing['compile_s']:.0f} s, "
      f"{timing['steady_sps']:,.0f} steps/s while training)"
  )

  if final_eval_episodes:
    policy = make_inference_fn(params, deterministic=True)
    result = evaluate.evaluate(env, policy, final_eval_episodes, seed=cfg.seed + 1000)
    (run_dir / "eval.json").write_text(
        json.dumps(evaluate.strip_trajectories(result), indent=2) + "\n"
    )
    log(evaluate.describe(result))
  return stats


def split_timing(rows: list[dict[str, Any]], steps_per_update: int) -> dict[str, float]:
  """Separates startup/compile time from training time, using wall-clock time.

  Every eval period does the same work, but the first one also compiles the
  training epoch, and the step-0 eval also compiles the eval rollout and env
  reset. Compile time = that extra time. (Brax's own per-epoch
  `training/sps` is not used: with JAX 0.9 it reads 5-10x too high because the
  epoch timer stops before the GPU work is done.)
  """
  wall = [r["wall_time_s"] for r in rows]
  steps = [r["timesteps"] for r in rows]
  startup_s = wall[0]  # setup + compile + first eval rollout
  if len(rows) < 2:
    return {"compile_s": round(startup_s, 1), "train_s": 0.0, "eval_s": 0.0,
            "steady_sps": 0.0, "train_only_sps": 0.0}
  period_s = np.diff(wall)  # training + eval + checkpoint, per eval period
  period_steps = np.diff(steps)
  evals = np.array([r["eval_time_s"] for r in rows])
  later = slice(1, None) if len(period_s) > 1 else slice(0, None)
  sec_per_step = float(np.median(period_s[later] / period_steps[later]))
  eval_steady = float(np.median(evals[2:] if len(evals) > 2 else evals[1:]))
  epoch_compile = max(float(period_s[0] - sec_per_step * period_steps[0]), 0.0)
  compile_s = max(startup_s - eval_steady, 0.0) + epoch_compile
  eval_s = eval_steady * len(rows)
  total_s = wall[-1]
  return {
      "compile_s": round(compile_s, 1),
      "first_eval_s": round(startup_s, 1),
      "train_epoch_compile_s": round(epoch_compile, 1),
      "eval_s": round(eval_s, 1),
      "train_s": round(total_s - compile_s - eval_s, 1),
      # Steps per second over a steady eval period (training + its eval)...
      "steady_sps": round(1.0 / sec_per_step),
      # ...and with the eval time taken out.
      "train_only_sps": round(period_steps[-1] / max(sec_per_step * period_steps[-1] - eval_steady, 1e-9)),
      "env_steps_per_update": steps_per_update,
  }


def default_run_dir(cfg: cfglib.TrainConfig) -> Path:
  stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  return RUNS_DIR / f"{stamp}-{cfg.level.type}-s{cfg.seed}"


def main() -> None:
  parser = argparse.ArgumentParser(description="Train the Sprint level from a config file.")
  parser.add_argument("--config", default=str(cfglib.DEFAULT_CONFIG_PATH))
  parser.add_argument("--run-dir", default=None, help="default: runs/<timestamp>-<level>-s<seed>")
  parser.add_argument("--seed", type=int, default=None, help="override the config's seed")
  parser.add_argument(
      "--compile-cache", default=".jax_cache",
      help="persistent XLA compilation cache dir ('' to disable)",
  )
  args = parser.parse_args()

  cfg = cfglib.load_config(args.config)
  if args.seed is not None:
    cfg = cfg.model_copy(update={"seed": args.seed})
  enable_compile_cache(args.compile_cache)
  run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(cfg)
  train(cfg, run_dir)


if __name__ == "__main__":
  main()
