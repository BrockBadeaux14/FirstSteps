"""Training config schema: the JSON contract the app server sends to a GPU worker.

The limits mirror the future block editor: hidden-layer widths come from a
fixed preset list, each network has 1-6 hidden layers, and activations come
from a short list. Every hyperparameter has a default, a range and a one-line
description (exported in the JSON schema, so the app can show tooltips).

  python -m sprinter.config --schema              # print the JSON schema
  python -m sprinter.config configs/sprint_default.json   # validate a file
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

WIDTH_PRESETS = (16, 32, 64, 128, 256)
ACTIVATIONS = ("relu", "tanh", "swish", "elu")
MIN_LAYERS, MAX_LAYERS = 1, 6

Width = Literal[16, 32, 64, 128, 256]
Activation = Literal["relu", "tanh", "swish", "elu"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "sprint_default.json"


class _Block(BaseModel):
  model_config = ConfigDict(extra="forbid")


class NetworkConfig(_Block):
  policy_hidden: list[Width] = Field(
      default=[32, 32, 32, 32], min_length=MIN_LAYERS, max_length=MAX_LAYERS,
      description="Policy (actor) hidden layers, from sensors to motors.",
  )
  value_hidden: list[Width] = Field(
      default=[256, 256, 256, 256, 256], min_length=MIN_LAYERS, max_length=MAX_LAYERS,
      description="Value (critic) hidden layers; only used during training.",
  )
  activation: Activation = Field(
      default="swish", description="Nonlinearity after every hidden layer."
  )


class PPOConfig(_Block):
  num_timesteps: int = Field(
      ge=1_000_000, le=500_000_000,
      description="Training length in simulation steps (all parallel sims together).",
  )
  num_envs: int = Field(
      default=2048, ge=64, le=8192, description="Number of parallel simulations."
  )
  batch_size: int = Field(
      default=1024, ge=32, le=8192, description="Rollouts per gradient minibatch."
  )
  num_minibatches: int = Field(
      default=32, ge=1, le=128, description="Minibatches per batch of collected data."
  )
  unroll_length: int = Field(
      default=30, ge=5, le=200, description="Steps each sim runs before an update."
  )
  num_updates_per_batch: int = Field(
      default=16, ge=1, le=64, description="Passes over each batch of data."
  )
  learning_rate: float = Field(
      default=1e-3, gt=0.0, le=1e-2, description="Step size of the optimizer (Adam)."
  )
  discounting: float = Field(
      default=0.995, ge=0.9, le=0.9999, description="How much future reward counts (gamma)."
  )
  entropy_cost: float = Field(
      default=1e-2, ge=0.0, le=0.1, description="Bonus for exploring with random actions."
  )
  clipping_epsilon: float = Field(
      default=0.3, ge=0.05, le=0.5, description="PPO clip range: max policy change per update."
  )
  reward_scaling: float = Field(
      default=10.0, gt=0.0, le=100.0, description="Multiplier applied to rewards before learning."
  )
  normalize_observations: bool = Field(
      default=True, description="Standardize sensor readings with running statistics."
  )
  num_evals: int = Field(
      default=21, ge=2, le=101,
      description="Evaluations (and checkpoints) during training, including step 0.",
  )
  num_eval_envs: int = Field(
      default=128, ge=8, le=1024, description="Episodes per evaluation."
  )

  @model_validator(mode="after")
  def _check_batching(self) -> PPOConfig:
    if (self.batch_size * self.num_minibatches) % self.num_envs:
      raise ValueError(
          "batch_size * num_minibatches must be a multiple of num_envs "
          f"(got {self.batch_size} * {self.num_minibatches} and {self.num_envs})"
      )
    return self

  @property
  def env_steps_per_update(self) -> int:
    return self.batch_size * self.num_minibatches * self.unroll_length

  @property
  def effective_timesteps(self) -> int:
    """Steps Brax actually runs: whole updates per eval period, rounded up."""
    periods = max(self.num_evals - 1, 1)
    per_period = -(-self.num_timesteps // (periods * self.env_steps_per_update))
    return periods * per_period * self.env_steps_per_update


class RewardWeights(_Block):
  forward_velocity: float = Field(
      default=1.0, ge=0.0, le=10.0, description="Reward per m/s of forward torso speed."
  )
  alive: float = Field(
      default=1.0, ge=0.0, le=10.0, description="Reward per step for staying upright."
  )
  upright: float = Field(
      default=0.0, ge=0.0, le=10.0, description="Reward for keeping the torso vertical (cos pitch)."
  )
  control_cost: float = Field(
      default=1e-3, ge=0.0, le=1.0, description="Penalty on squared motor commands (energy)."
  )


class SimConfig(_Block):
  impl: Literal["warp", "jax"] = Field(
      default="warp", description="Physics backend: MuJoCo Warp or MJX (JAX)."
  )
  episode_seconds: float = Field(
      default=10.0, ge=1.0, le=60.0, description="Episode length in simulated seconds."
  )
  ctrl_dt: float = Field(
      default=0.025, ge=0.005, le=0.05, description="Seconds between policy actions."
  )
  sim_dt: float = Field(
      default=0.0025, ge=0.0005, le=0.01, description="Physics timestep in seconds."
  )

  @model_validator(mode="after")
  def _check_timing(self) -> SimConfig:
    n_sub = self.ctrl_dt / self.sim_dt
    if abs(n_sub - round(n_sub)) > 1e-6:
      raise ValueError("ctrl_dt must be a whole multiple of sim_dt")
    steps = self.episode_seconds / self.ctrl_dt
    if abs(steps - round(steps)) > 1e-6:
      raise ValueError("episode_seconds must be a whole multiple of ctrl_dt")
    return self

  @property
  def episode_length(self) -> int:
    return int(round(self.episode_seconds / self.ctrl_dt))


class TrainConfig(_Block):
  level: Literal["sprint"] = Field(default="sprint", description="Which level to train.")
  seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Random seed.")
  network: NetworkConfig = Field(default_factory=NetworkConfig)
  ppo: PPOConfig
  reward_weights: RewardWeights = Field(default_factory=RewardWeights)
  sim: SimConfig = Field(default_factory=SimConfig)

  def canonical_json(self) -> str:
    return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

  def config_hash(self) -> str:
    """Short, stable hash of the validated config (same config -> same hash)."""
    return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]


def load_config(path: str | Path) -> TrainConfig:
  return TrainConfig.model_validate_json(Path(path).read_text())


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("config", nargs="?", default=None, help="config JSON to validate")
  parser.add_argument("--schema", action="store_true", help="print the JSON schema")
  args = parser.parse_args()
  if args.schema:
    print(json.dumps(TrainConfig.model_json_schema(), indent=2))
    return
  cfg = load_config(args.config or DEFAULT_CONFIG_PATH)
  print(json.dumps(cfg.model_dump(mode="json"), indent=2))
  print(f"valid; hash={cfg.config_hash()}; "
        f"effective num_timesteps={cfg.ppo.effective_timesteps:,}")


if __name__ == "__main__":
  main()
