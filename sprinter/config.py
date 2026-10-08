"""Training config schema: the JSON contract between the app, the server and a GPU worker.

Version 2. A config picks one member of each tagged union below (its `type`
field says which) and gives that member's settings:

  level      the task and its reward weights   sprint (the level cards add the others)
  brain      what the runner learns            rhythm, mlp
  trainer    how it learns                     hill_climbing, ga, pso, cma_es, bc, reinforce, ppo
  optimizer  how a gradient moves the weights  sgd, rmsprop, adam (gradient-based trainers only)

plus `seed` and `sim`. The limits mirror the future block editor, and every
setting has a default, a range and a one-line description (exported in the JSON
schema, so the app can show tooltips). The rules that span blocks are in
`TrainConfig._check_combination`. docs/config.md explains the whole contract.

Version 1 configs (milestone 1: Sprint trained with PPO) still load. They are
migrated to version 2 before validation, so a v1 file and its v2 form are the
same config with the same config_hash.

  python -m sprinter.config --schema              # print the JSON schema
  python -m sprinter.config configs/sprint_default.json   # validate a file
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_VERSION = 2

WIDTH_PRESETS = (16, 32, 64, 128, 256)
ACTIVATIONS = ("relu", "tanh", "swish", "elu")
MIN_LAYERS, MAX_LAYERS = 1, 6
DEFAULT_VALUE_HIDDEN = (256, 256, 256, 256, 256)

Width = Literal[16, 32, 64, 128, 256]
Activation = Literal["relu", "tanh", "swish", "elu"]
Layers = Annotated[list[Width], Field(min_length=MIN_LAYERS, max_length=MAX_LAYERS)]

NUM_MOTORS = 10  # actuators in humanoid2d.xml; every level uses the same body

ZERO_ORDER_TRAINERS = ("hill_climbing", "ga", "pso", "cma_es")
GRADIENT_TRAINERS = ("bc", "reinforce", "ppo")
ACTOR_CRITIC_TRAINERS = ("ppo",)  # the trainers with a value network

# Placeholders until the spike (#9) reports.
# Training length cap per trainer, in the trainer's own unit, sized for about
# 10 minutes on the RTX 3070 (docs/game-design.md#the-budget).
STEP_CAPS = {
    "hill_climbing": 178,  # generations: ~178 generations of 2,048 runners fit in 10 min
    "ga": 178,
    "pso": 178,
    "cma_es": 178,
    "bc": 100_000,  # gradient steps
    "reinforce": 58_982_400,  # environment steps
    "ppo": 58_982_400,  # environment steps: the default Sprint config, 10.4 min
}
# Most numbers CMA-ES may tune: its covariance matrix has one entry per pair.
CMA_ES_MAX_PARAMS = 100

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "sprint_default.json"
EXAMPLES_DIR = ROOT / "configs" / "examples"


class _Block(BaseModel):
  model_config = ConfigDict(extra="forbid")


# -----------------------------------------------------------------------------
# level: one member per level, each with its own reward weights


class SprintRewardWeights(_Block):
  """Weights of the Sprint reward's named terms (sprinter/envs/sprint.py)."""

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
  alternation: float = Field(
      default=10.0, ge=0.0, le=20.0,
      description="Reward for landing on alternate feet; penalty for hopping on one foot.",
  )
  foot_slip: float = Field(
      default=0.5, ge=0.0, le=10.0,
      description="Penalty on the squared sliding speed of grounded feet (stops dragging).",
  )
  symmetry: float = Field(
      default=5.0, ge=0.0, le=20.0,
      description="Penalty when one foot spends more time on the ground than the other (limping).",
  )


class SprintLevel(_Block):
  """Sprint: cover as much ground as possible in 10 s."""

  type: Literal["sprint"]
  reward_weights: SprintRewardWeights = Field(
      default_factory=SprintRewardWeights, description="How much each reward term counts."
  )


# One member so far. Each level card adds its own class (a `type` literal and its
# reward-weights block) and makes this
#   Annotated[Union[SprintLevel, CrawlLevel, ...], Field(discriminator="type")]
Level = SprintLevel


# -----------------------------------------------------------------------------
# brain


class RhythmBrain(_Block):
  """Rhythm controller: each motor follows a fixed rhythm. It reads no sensors."""

  type: Literal["rhythm"]
  harmonics: int = Field(
      default=1, ge=1, le=2,
      description="Sine waves per motor. Each one adds an amplitude and a phase per motor.",
  )

  @property
  def num_params(self) -> int:
    """Numbers a trainer tunes: per motor, an offset plus an amplitude and a phase
    per harmonic; and one shared frequency. 31 with one harmonic, 51 with two."""
    return NUM_MOTORS * (1 + 2 * self.harmonics) + 1


class MLPBrain(_Block):
  """Neural network from the body's sensor readings to its motors."""

  type: Literal["mlp"]
  policy_hidden: Layers = Field(
      default=[32, 32, 32, 32],
      description="Policy (actor) hidden layers, from sensors to motors.",
  )
  value_hidden: Optional[Layers] = Field(
      default=None,
      description=(
          "Value (critic) hidden layers, only for actor-critic trainers (PPO), which get "
          "5 layers of 256 when this is left out. Only used during training."
      ),
  )
  activation: Activation = Field(
      default="swish", description="Nonlinearity after every hidden layer."
  )


Brain = Annotated[Union[RhythmBrain, MLPBrain], Field(discriminator="type")]


# -----------------------------------------------------------------------------
# trainer: one member per trainer, each with its own settings

PopulationSize = Annotated[int, Field(
    ge=32, le=4096, description="Candidates scored per generation, each in its own simulation.",
)]
EliteFraction = Annotated[float, Field(
    ge=0.05, le=0.5, description="Share of each generation's best candidates that the next one is built from.",
)]
NumEnvs = Annotated[int, Field(ge=64, le=8192, description="Number of parallel simulations.")]
Discounting = Annotated[float, Field(
    ge=0.9, le=0.9999, description="How much future reward counts (gamma).",
)]
EntropyCost = Annotated[float, Field(
    ge=0.0, le=0.1, description="Bonus for exploring with random actions.",
)]
NormalizeObservations = Annotated[bool, Field(
    description="Standardize sensor readings with running statistics.",
)]
NumEvals = Annotated[int, Field(
    ge=2, le=101, description="Evaluations (and checkpoints) during training, including the start.",
)]
NumEvalEnvs = Annotated[int, Field(ge=8, le=1024, description="Episodes per evaluation.")]


def _training_length(trainer: str, unit: str, minimum: int) -> Any:
  cap = STEP_CAPS[trainer]
  return Field(
      default=cap, ge=minimum, le=cap,
      description=f"Training length in {unit}, up to the {trainer} budget of {cap:,}.",
  )


class _Trainer(_Block):
  unit: ClassVar[str]  # what training_length counts

  @property
  def training_length(self) -> int:
    """How far the run trains, in `unit`: the number the step cap applies to."""
    raise NotImplementedError


class _EvolutionTrainer(_Trainer):
  unit: ClassVar[str] = "generations"

  @property
  def training_length(self) -> int:
    return self.generations


class HillClimbingTrainer(_EvolutionTrainer):
  """Hill climbing: change the numbers a little and keep the change if the score improves."""

  type: Literal["hill_climbing"]
  generations: int = _training_length("hill_climbing", "generations", 1)
  population_size: PopulationSize = 2048
  step_size: float = Field(
      default=0.1, gt=0.0, le=1.0, description="Size of each random change (standard deviation).",
  )


class GATrainer(_EvolutionTrainer):
  """Genetic algorithm: breed the best candidates and mutate their children."""

  type: Literal["ga"]
  generations: int = _training_length("ga", "generations", 1)
  population_size: PopulationSize = 2048
  elite_fraction: EliteFraction = 0.5
  mutation_std: float = Field(
      default=0.1, gt=0.0, le=1.0, description="Size of each child's random mutation (standard deviation).",
  )


class PSOTrainer(_EvolutionTrainer):
  """Particle swarm: each candidate moves toward its own best numbers and the swarm's best."""

  type: Literal["pso"]
  generations: int = _training_length("pso", "generations", 1)
  population_size: PopulationSize = 2048
  inertia: float = Field(
      default=0.75, ge=0.0, le=1.0, description="How much of its last move each particle keeps.",
  )
  cognitive: float = Field(
      default=1.5, ge=0.0, le=4.0, description="Pull toward each particle's own best numbers.",
  )
  social: float = Field(
      default=2.0, ge=0.0, le=4.0, description="Pull toward the best numbers the whole swarm found.",
  )


class CMAESTrainer(_EvolutionTrainer):
  """CMA-ES: learns which numbers should change together (a covariance matrix)."""

  type: Literal["cma_es"]
  generations: int = _training_length("cma_es", "generations", 1)
  population_size: PopulationSize = 2048
  elite_fraction: EliteFraction = 0.5
  initial_std: float = Field(
      default=0.1, gt=0.0, le=1.0,
      description="Size of the first generation's random changes; CMA-ES adapts it after that.",
  )


class BCTrainer(_Trainer):
  """Copy trainer (behavior cloning): learns to imitate a recorded expert."""

  unit: ClassVar[str] = "gradient steps"
  type: Literal["bc"]
  gradient_steps: int = _training_length("bc", "gradient steps", 1_000)
  batch_size: int = Field(
      default=256, ge=16, le=4096, description="Expert examples per gradient step.",
  )
  normalize_observations: NormalizeObservations = True
  num_evals: NumEvals = 21
  num_eval_envs: NumEvalEnvs = 128

  @property
  def training_length(self) -> int:
    return self.gradient_steps


class ReinforceTrainer(_Trainer):
  """REINFORCE: plain policy gradient on whole episodes, with no value network."""

  unit: ClassVar[str] = "environment steps"
  type: Literal["reinforce"]
  num_timesteps: int = _training_length(
      "reinforce", "simulation steps (all parallel sims together)", 1_000_000
  )
  num_envs: NumEnvs = 2048
  discounting: Discounting = 0.995
  entropy_cost: EntropyCost = 1e-2
  normalize_observations: NormalizeObservations = True
  num_evals: NumEvals = 21
  num_eval_envs: NumEvalEnvs = 128

  @property
  def training_length(self) -> int:
    return self.num_timesteps


class PPOTrainer(_Trainer):
  """PPO: policy gradient with a value network and a limit on each update's change."""

  unit: ClassVar[str] = "environment steps"
  type: Literal["ppo"]
  num_timesteps: int = _training_length(
      "ppo", "simulation steps (all parallel sims together)", 1_000_000
  )
  num_envs: NumEnvs = 2048
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
  discounting: Discounting = 0.995
  entropy_cost: EntropyCost = 1e-2
  clipping_epsilon: float = Field(
      default=0.3, ge=0.05, le=0.5, description="PPO clip range: max policy change per update."
  )
  reward_scaling: float = Field(
      default=10.0, gt=0.0, le=100.0, description="Multiplier applied to rewards before learning."
  )
  normalize_observations: NormalizeObservations = True
  num_evals: NumEvals = 21
  num_eval_envs: NumEvalEnvs = 128

  @model_validator(mode="after")
  def _check_batching(self) -> PPOTrainer:
    if (self.batch_size * self.num_minibatches) % self.num_envs:
      raise ValueError(
          "batch_size * num_minibatches must be a multiple of num_envs "
          f"(got {self.batch_size} * {self.num_minibatches} and {self.num_envs})"
      )
    cap = STEP_CAPS["ppo"]
    if self.effective_timesteps > cap:
      raise ValueError(
          f"num_timesteps rounds up to {self.effective_timesteps:,} steps (whole updates of "
          f"{self.env_steps_per_update:,} in each of {max(self.num_evals - 1, 1)} eval periods), "
          f"over the ppo budget of {cap:,}"
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

  @property
  def training_length(self) -> int:
    return self.effective_timesteps


Trainer = Annotated[
    Union[
        HillClimbingTrainer, GATrainer, PSOTrainer, CMAESTrainer,
        BCTrainer, ReinforceTrainer, PPOTrainer,
    ],
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# optimizer: only for gradient-based trainers


class SGDOptimizer(_Block):
  """SGD: each weight moves against its gradient, scaled by the learning rate."""

  type: Literal["sgd"]
  learning_rate: float = Field(
      default=1e-2, gt=0.0, le=1.0, description="Step size: how far each update moves the weights."
  )


class RMSpropOptimizer(_Block):
  """RMSprop: divides each weight's step by the recent size of its gradient."""

  type: Literal["rmsprop"]
  learning_rate: float = Field(
      default=1e-3, gt=0.0, le=1e-2, description="Step size: how far each update moves the weights."
  )


class AdamOptimizer(_Block):
  """Adam: RMSprop plus momentum (a running average of the gradient's direction)."""

  type: Literal["adam"]
  learning_rate: float = Field(
      default=1e-3, gt=0.0, le=1e-2, description="Step size: how far each update moves the weights."
  )


Optimizer = Annotated[
    Union[SGDOptimizer, RMSpropOptimizer, AdamOptimizer], Field(discriminator="type")
]


# -----------------------------------------------------------------------------
# sim and the whole config


class SimConfig(_Block):
  """Physics backend and timing."""

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
  """One training run: what the app builds and a GPU worker runs."""

  config_version: Literal[2] = Field(description="Version of this config format.")
  seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Random seed.")
  level: Level = Field(description="Which level to train, with its reward weights.")
  brain: Brain = Field(description="What the runner learns: a rhythm controller or a neural network.")
  trainer: Trainer = Field(description="How the brain learns, with that trainer's settings.")
  optimizer: Optional[Optimizer] = Field(
      default=None,
      description=(
          "How a gradient moves the weights. Only for gradient-based trainers "
          "(bc, reinforce, ppo), which get Adam when this is left out."
      ),
  )
  sim: SimConfig = Field(default_factory=SimConfig, description="Physics backend and timing.")

  @model_validator(mode="before")
  @classmethod
  def _migrate(cls, data: Any) -> Any:
    if isinstance(data, dict) and data.get("config_version", 1) == 1:
      return migrate_v1(data)
    return data

  @model_validator(mode="after")
  def _check_combination(self) -> TrainConfig:
    t = self.trainer.type
    gradient = t in GRADIENT_TRAINERS
    # Zero-order trainers search a rhythm controller's few numbers; gradient-based
    # trainers train a neural network (docs/game-design.md).
    brain = "mlp" if gradient else "rhythm"
    if self.brain.type != brain:
      kind = "gradient-based" if gradient else "zero-order (it uses no gradients)"
      raise ValueError(f"{t} is {kind}, so it trains the {brain} brain, not {self.brain.type}")
    if gradient and self.optimizer is None:
      self.optimizer = AdamOptimizer(type="adam")
    if not gradient and self.optimizer is not None:
      raise ValueError(f"{t} uses no gradients, so it takes no optimizer")
    if isinstance(self.brain, MLPBrain):
      if t in ACTOR_CRITIC_TRAINERS and self.brain.value_hidden is None:
        self.brain = self.brain.model_copy(update={"value_hidden": list(DEFAULT_VALUE_HIDDEN)})
      if t not in ACTOR_CRITIC_TRAINERS and self.brain.value_hidden is not None:
        raise ValueError(f"value_hidden is only for actor-critic trainers; {t} has no value network")
    if t == "cma_es" and self.brain.num_params > CMA_ES_MAX_PARAMS:
      raise ValueError(
          f"cma_es tunes at most {CMA_ES_MAX_PARAMS} numbers; "
          f"this rhythm brain has {self.brain.num_params}"
      )
    return self

  def canonical_json(self) -> str:
    # Left-out optional blocks and null are the same config, so nulls are dropped.
    data = self.model_dump(mode="json", exclude_none=True)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))

  def config_hash(self) -> str:
    """Short, stable hash of the validated config (same config -> same hash)."""
    return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]


def migrate_v1(data: dict[str, Any]) -> dict[str, Any]:
  """Version 1 config (milestone 1: Sprint trained with PPO) -> version 2.

  `network` becomes the mlp brain and `ppo` the ppo trainer. Its learning rate
  moves to an adam optimizer (Brax's PPO always used Adam), and `reward_weights`
  move into the level. Fields the v1 config left out stay out, so they get the
  same defaults as before. Unknown fields are kept, so validation still rejects
  them.
  """
  v1 = dict(data)
  v1.pop("config_version", None)
  if not isinstance(v1.get("ppo"), dict):
    raise ValueError(
        "config_version is missing, so this was read as a version 1 config, "
        "which needs a ppo block"
    )
  trainer = {"type": "ppo", **v1.pop("ppo")}
  optimizer = {"type": "adam"}
  if "learning_rate" in trainer:
    optimizer["learning_rate"] = trainer.pop("learning_rate")
  level = {"type": v1.pop("level", "sprint")}
  if "reward_weights" in v1:
    level["reward_weights"] = v1.pop("reward_weights")
  network = v1.pop("network", {})
  brain = {"type": "mlp", **network} if isinstance(network, dict) else network
  return {
      "config_version": CONFIG_VERSION, "level": level, "brain": brain,
      "trainer": trainer, "optimizer": optimizer,
      **v1,  # seed, sim and any unknown fields, as they were
  }


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
  t = cfg.trainer
  print(json.dumps(cfg.model_dump(mode="json"), indent=2))
  print(f"valid; hash={cfg.config_hash()}; {t.type} trains for {t.training_length:,} {t.unit} "
        f"(budget {STEP_CAPS[t.type]:,})")


if __name__ == "__main__":
  main()
