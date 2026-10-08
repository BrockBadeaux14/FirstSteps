"""Find checkpoints in a run directory and load them as policies."""

from __future__ import annotations

from pathlib import Path

from brax.training import checkpoint as brax_checkpoint
from brax.training import types as brax_types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from sprinter import config as cfglib
from sprinter import train as trainlib

EARLY_FRACTION = 0.10  # "before" = the checkpoint nearest 10% of training


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
  """(step, path) of every periodic checkpoint, oldest first."""
  ckpts = [(int(p.name), p) for p in (run_dir / "checkpoints").iterdir() if p.name.isdigit()]
  return sorted(ckpts)


def final_checkpoint(run_dir: Path) -> tuple[int, Path]:
  final_dir = run_dir / "final"
  final = sorted(
      (int(p.name), p) for p in final_dir.iterdir() if p.name.isdigit()
  ) if final_dir.exists() else []
  return final[-1] if final else list_checkpoints(run_dir)[-1]


def early_checkpoint(run_dir: Path, fraction: float = EARLY_FRACTION) -> tuple[int, Path]:
  final_step, _ = final_checkpoint(run_dir)
  return min(list_checkpoints(run_dir), key=lambda c: abs(c[0] - fraction * final_step))


def pick_checkpoint(run_dir: Path, which: str) -> tuple[int, Path]:
  """`which` is "final", "before" (~10% of training) or a step number."""
  if which == "final":
    return final_checkpoint(run_dir)
  if which == "before":
    return early_checkpoint(run_dir)
  ckpts = dict(list_checkpoints(run_dir))
  if not which.isdigit() or int(which) not in ckpts:
    steps = ", ".join(str(s) for s in ckpts)
    raise ValueError(f"no checkpoint {which!r}; use final, before or one of: {steps}")
  return int(which), ckpts[int(which)]


def load_policy(cfg: cfglib.TrainConfig, ckpt: Path, obs_size: int, action_size: int):
  """Deterministic inference function: policy(obs, key) -> (action, {})."""
  params = brax_checkpoint.load(ckpt)
  normalize = (running_statistics.normalize if cfg.trainer.normalize_observations
               else brax_types.identity_observation_preprocessor)
  network = trainlib.make_network_factory(cfg.brain)(
      obs_size, action_size, preprocess_observations_fn=normalize
  )
  return ppo_networks.make_inference_fn(network)(params, deterministic=True)
