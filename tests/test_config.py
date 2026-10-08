"""Checks on the training-config schema (the contract between the app, the server and a worker)."""

import copy
import itertools
import json
from pathlib import Path
import sys

from flax import linen
import mujoco
from pydantic import ValidationError
import pytest

from sprinter import config as cfglib
from sprinter import train

DEFAULT = json.loads(cfglib.DEFAULT_CONFIG_PATH.read_text())
V1_DEFAULT = json.loads((Path(__file__).parent / "data" / "sprint_default_v1.json").read_text())
EXAMPLES = sorted(cfglib.EXAMPLES_DIR.glob("*.json"))
MODEL_XML = Path(cfglib.__file__).parent / "assets" / "humanoid2d.xml"

TRAINERS = cfglib.ZERO_ORDER_TRAINERS + cfglib.GRADIENT_TRAINERS
OPTIMIZERS = ("sgd", "rmsprop", "adam")
BRAIN_FOR = {t: "rhythm" for t in cfglib.ZERO_ORDER_TRAINERS} | {
    t: "mlp" for t in cfglib.GRADIENT_TRAINERS
}
LENGTH_FIELD = {
    "hill_climbing": "generations", "ga": "generations", "pso": "generations",
    "cma_es": "generations", "bc": "gradient_steps", "reinforce": "num_timesteps",
    "ppo": "num_timesteps",
}


def _with(path: str, value, base=DEFAULT):
  cfg = copy.deepcopy(base)
  node = cfg
  *parents, leaf = path.split(".")
  for p in parents:
    node = node[p]
  node[leaf] = value
  return cfg


def _minimal(trainer: str, /, **blocks):
  """The shortest valid config for a trainer: every setting at its default."""
  cfg = {
      "config_version": 2,
      "level": {"type": "sprint"},
      "brain": {"type": BRAIN_FOR[trainer]},
      "trainer": {"type": trainer},
  }
  cfg.update(blocks)
  return cfg


def _validate(data) -> cfglib.TrainConfig:
  return cfglib.TrainConfig.model_validate(data)


# -----------------------------------------------------------------------------
# The default config and the examples


def test_default_config_is_valid():
  cfg = _validate(DEFAULT)
  assert cfg.config_version == 2
  assert cfg.level.type == "sprint"
  assert cfg.sim.episode_length == 400
  assert cfg.brain.type == "mlp"
  assert cfg.brain.policy_hidden == [32, 32, 32, 32]
  assert cfg.brain.value_hidden == [256] * 5
  assert cfg.trainer.type == "ppo"
  assert cfg.optimizer.type == "adam" and cfg.optimizer.learning_rate == 1e-3


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_examples_are_valid(path):
  cfg = cfglib.load_config(path)
  assert path.name == f"{cfg.level.type}_{cfg.trainer.type}.json"


def test_examples_cover_every_union_member():
  cfgs = [cfglib.load_config(p) for p in [cfglib.DEFAULT_CONFIG_PATH, *EXAMPLES]]
  assert {c.trainer.type for c in cfgs} == set(TRAINERS)
  assert {c.brain.type for c in cfgs} == {"rhythm", "mlp"}
  assert {c.optimizer.type for c in cfgs if c.optimizer} == set(OPTIMIZERS)


def test_trainer_families_match_the_union():
  schema = cfglib.TrainConfig.model_json_schema()
  assert set(schema["properties"]["trainer"]["discriminator"]["mapping"]) == set(TRAINERS)
  assert not set(cfglib.ZERO_ORDER_TRAINERS) & set(cfglib.GRADIENT_TRAINERS)
  assert set(cfglib.ACTOR_CRITIC_TRAINERS) <= set(cfglib.GRADIENT_TRAINERS)
  assert set(cfglib.STEP_CAPS) == set(TRAINERS)


# -----------------------------------------------------------------------------
# level


def test_level_reward_weights_default_when_left_out():
  cfg = _validate(_with("level", {"type": "sprint"}))
  assert cfg.level.reward_weights == cfglib.SprintRewardWeights()


@pytest.mark.parametrize("level", ["sprint", {"type": "climb"}, {}, {"reward_weights": {}}])
def test_rejects_unknown_or_untagged_levels(level):
  with pytest.raises(ValidationError):
    _validate(_with("level", level))


# -----------------------------------------------------------------------------
# brain


@pytest.mark.parametrize("harmonics,params", [(1, 31), (2, 51)])
def test_rhythm_brain_size(harmonics, params):
  cfg = _validate(_minimal("pso", brain={"type": "rhythm", "harmonics": harmonics}))
  assert cfg.brain.num_params == params


@pytest.mark.parametrize("harmonics", [0, 3])
def test_rejects_rhythm_harmonics_out_of_range(harmonics):
  with pytest.raises(ValidationError):
    _validate(_minimal("pso", brain={"type": "rhythm", "harmonics": harmonics}))


def test_rhythm_brain_drives_every_motor():
  assert mujoco.MjModel.from_xml_path(str(MODEL_XML)).nu == cfglib.NUM_MOTORS


@pytest.mark.parametrize("brain", [{"type": "linear"}, {"policy_hidden": [32]},
                                   {"type": "rhythm", "activation": "tanh"}])
def test_rejects_unknown_or_untagged_brains(brain):
  with pytest.raises(ValidationError):
    _validate(_with("brain", brain, base=_minimal("pso")))


@pytest.mark.parametrize("widths", [[48], [32, 100], [512], [0]])
def test_rejects_widths_outside_presets(widths):
  with pytest.raises(ValidationError):
    _validate(_with("brain.policy_hidden", widths))


@pytest.mark.parametrize("n", [0, 7])
def test_rejects_too_few_or_too_many_layers(n):
  with pytest.raises(ValidationError):
    _validate(_with("brain.value_hidden", [64] * n))


@pytest.mark.parametrize("n", [1, 6])
def test_accepts_one_to_six_layers(n):
  _validate(_with("brain.policy_hidden", [16] * n))


@pytest.mark.parametrize("act", ["relu", "tanh", "swish", "elu"])
def test_accepts_preset_activations(act):
  _validate(_with("brain.activation", act))


@pytest.mark.parametrize("act", ["gelu", "sigmoid", "", "Swish"])
def test_rejects_other_activations(act):
  with pytest.raises(ValidationError):
    _validate(_with("brain.activation", act))


def test_value_network_defaults_for_actor_critic_trainers():
  cfg = _validate(_minimal("ppo"))
  assert cfg.brain.value_hidden == list(cfglib.DEFAULT_VALUE_HIDDEN)


@pytest.mark.parametrize("trainer", ["bc", "reinforce"])
def test_no_value_network_without_a_critic(trainer):
  assert _validate(_minimal(trainer)).brain.value_hidden is None
  brain = {"type": "mlp", "value_hidden": [64, 64]}
  with pytest.raises(ValidationError, match="value_hidden is only for actor-critic"):
    _validate(_minimal(trainer, brain=brain))


# -----------------------------------------------------------------------------
# trainer


@pytest.mark.parametrize("trainer", TRAINERS)
def test_every_trainer_validates_with_defaults(trainer):
  cfg = _validate(_minimal(trainer))
  assert cfg.trainer.type == trainer
  assert cfg.trainer.training_length == cfglib.STEP_CAPS[trainer]
  assert (cfg.optimizer is None) == (trainer in cfglib.ZERO_ORDER_TRAINERS)


@pytest.mark.parametrize("trainer", ["trpo", "sac", "es", "PPO", ""])
def test_rejects_unknown_trainers(trainer):
  with pytest.raises(ValidationError):
    _validate(_with("trainer", {"type": trainer}))


def test_rejects_untagged_trainer():
  trainer = {k: v for k, v in DEFAULT["trainer"].items() if k != "type"}
  with pytest.raises(ValidationError):
    _validate(_with("trainer", trainer))


@pytest.mark.parametrize(
    "trainer,field,value",
    [
        ("hill_climbing", "step_size", 0.0),
        ("hill_climbing", "population_size", 8),
        ("ga", "generations", 0),
        ("ga", "elite_fraction", 0.9),
        ("ga", "mutation_std", -0.1),
        ("pso", "inertia", 1.5),
        ("pso", "social", 5.0),
        ("cma_es", "initial_std", 0.0),
        ("cma_es", "population_size", 10_000),
        ("bc", "batch_size", 8),
        ("bc", "gradient_steps", 10),
        ("reinforce", "num_envs", 32),
        ("reinforce", "entropy_cost", 0.5),
        ("ppo", "unroll_length", 2),
    ],
)
def test_rejects_trainer_settings_out_of_range(trainer, field, value):
  with pytest.raises(ValidationError):
    _validate(_minimal(trainer, trainer={"type": trainer, field: value}))


@pytest.mark.parametrize(
    "trainer,field,value",
    [
        ("reinforce", "clipping_epsilon", 0.3),
        ("ppo", "population_size", 2048),
        ("hill_climbing", "mutation_std", 0.1),
        ("bc", "num_timesteps", 1_000_000),
        ("ppo", "learning_rate", 1e-3),  # moved to the optimizer in v2
    ],
)
def test_rejects_another_trainers_settings(trainer, field, value):
  with pytest.raises(ValidationError):
    _validate(_minimal(trainer, trainer={"type": trainer, field: value}))


# -----------------------------------------------------------------------------
# the budget: training length capped per trainer


@pytest.mark.parametrize("trainer", TRAINERS)
def test_training_length_is_capped_per_trainer(trainer):
  cap = cfglib.STEP_CAPS[trainer]
  field = LENGTH_FIELD[trainer]
  ok = _validate(_minimal(trainer, trainer={"type": trainer, field: cap}))
  assert ok.trainer.training_length == cap
  with pytest.raises(ValidationError):
    _validate(_minimal(trainer, trainer={"type": trainer, field: cap + 1}))


def test_ppo_cap_applies_to_the_rounded_up_length():
  # 31-step unrolls: three whole updates per eval period overshoot the cap.
  with pytest.raises(ValidationError, match="rounds up"):
    _validate(_with("trainer.unroll_length", 31))
  shorter = _with("trainer.num_timesteps", 40_000_000, base=_with("trainer.unroll_length", 31))
  assert _validate(shorter).trainer.effective_timesteps <= cfglib.STEP_CAPS["ppo"]


def test_effective_timesteps_rounds_up_to_whole_updates():
  p = _validate(DEFAULT).trainer
  assert p.effective_timesteps >= p.num_timesteps
  assert p.effective_timesteps % (p.env_steps_per_update * (p.num_evals - 1)) == 0


# -----------------------------------------------------------------------------
# optimizer


@pytest.mark.parametrize(
    "trainer,optimizer", list(itertools.product(cfglib.GRADIENT_TRAINERS, OPTIMIZERS))
)
def test_every_optimizer_works_with_every_gradient_trainer(trainer, optimizer):
  cfg = _validate(_minimal(trainer, optimizer={"type": optimizer}))
  assert cfg.optimizer.type == optimizer


@pytest.mark.parametrize("trainer", cfglib.GRADIENT_TRAINERS)
def test_gradient_trainers_default_to_adam(trainer):
  assert _validate(_minimal(trainer)).optimizer == cfglib.AdamOptimizer(type="adam")


@pytest.mark.parametrize("trainer", cfglib.ZERO_ORDER_TRAINERS)
def test_zero_order_trainers_take_no_optimizer(trainer):
  with pytest.raises(ValidationError, match="takes no optimizer"):
    _validate(_minimal(trainer, optimizer={"type": "adam"}))
  assert _validate(_minimal(trainer, optimizer=None)).optimizer is None


@pytest.mark.parametrize(
    "optimizer,lr,ok",
    [
        ("sgd", 0.5, True), ("sgd", 1.5, False), ("sgd", 0.0, False),
        ("rmsprop", 5e-3, True), ("rmsprop", 0.05, False),
        ("adam", 1e-2, True), ("adam", 0.05, False), ("adam", -1e-3, False),
    ],
)
def test_learning_rate_range_per_optimizer(optimizer, lr, ok):
  data = _with("optimizer", {"type": optimizer, "learning_rate": lr})
  if ok:
    assert _validate(data).optimizer.learning_rate == lr
  else:
    with pytest.raises(ValidationError):
      _validate(data)


@pytest.mark.parametrize("optimizer", [{"type": "adagrad"}, {"learning_rate": 1e-3},
                                       {"type": "adam", "momentum": 0.9}])
def test_rejects_unknown_or_untagged_optimizers(optimizer):
  with pytest.raises(ValidationError):
    _validate(_with("optimizer", optimizer))


# -----------------------------------------------------------------------------
# cross-field rules: trainer <-> brain


@pytest.mark.parametrize("trainer", cfglib.ZERO_ORDER_TRAINERS)
def test_zero_order_trainers_use_the_rhythm_brain(trainer):
  with pytest.raises(ValidationError, match="zero-order"):
    _validate(_minimal(trainer, brain={"type": "mlp"}))


@pytest.mark.parametrize("trainer", cfglib.GRADIENT_TRAINERS)
def test_gradient_trainers_use_the_mlp_brain(trainer):
  with pytest.raises(ValidationError, match="gradient-based"):
    _validate(_minimal(trainer, brain={"type": "rhythm"}))


def test_cma_es_parameter_cap(monkeypatch):
  big = {"type": "rhythm", "harmonics": 2}  # 51 numbers
  _validate(_minimal("cma_es", brain=big))  # under today's placeholder cap
  monkeypatch.setattr(cfglib, "CMA_ES_MAX_PARAMS", 40)
  _validate(_minimal("cma_es"))  # 31 numbers
  with pytest.raises(ValidationError, match="cma_es tunes at most 40 numbers"):
    _validate(_minimal("cma_es", brain=big))
  _validate(_minimal("pso", brain=big))  # the cap is CMA-ES's alone


# -----------------------------------------------------------------------------
# other blocks and fields


@pytest.mark.parametrize(
    "path,value",
    [
        ("trainer.discounting", 1.5),
        ("trainer.num_envs", 1000),  # batch_size * num_minibatches not a multiple
        ("trainer.num_timesteps", 10),
        ("trainer.clipping_epsilon", 0.9),
        ("level.reward_weights.alive", -1.0),
        ("sim.impl", "cpu"),
        ("sim.sim_dt", 0.003),  # ctrl_dt is not a whole multiple
        ("seed", -1),
        ("config_version", 3),
    ],
)
def test_rejects_out_of_range_values(path, value):
  with pytest.raises(ValidationError):
    _validate(_with(path, value))


@pytest.mark.parametrize("path", ["trainer.lr", "level.reward_weights.jump", "network", "ppo"])
def test_rejects_unknown_fields(path):
  with pytest.raises(ValidationError):
    _validate(_with(path, 1.0))


# -----------------------------------------------------------------------------
# config_version and the v1 -> v2 migration


def test_v1_config_migrates_to_the_same_config():
  old, new = _validate(V1_DEFAULT), _validate(DEFAULT)
  assert old == new
  assert old.config_hash() == new.config_hash()


def test_v1_config_with_explicit_version_1_migrates():
  assert _validate({"config_version": 1, **V1_DEFAULT}) == _validate(DEFAULT)


def test_v1_left_out_fields_keep_their_defaults():
  cfg = _validate({"ppo": {"num_timesteps": 58_982_400}})
  assert cfg == _validate(_minimal("ppo"))
  assert cfg.brain.value_hidden == [256] * 5 and cfg.optimizer.learning_rate == 1e-3


def test_v1_learning_rate_moves_to_adam():
  cfg = _validate(_with("ppo.learning_rate", 5e-4, base=V1_DEFAULT))
  assert cfg.optimizer == cfglib.AdamOptimizer(type="adam", learning_rate=5e-4)


def test_migration_leaves_the_input_alone():
  data = copy.deepcopy(V1_DEFAULT)
  cfglib.migrate_v1(data)
  assert data == V1_DEFAULT


@pytest.mark.parametrize(
    "path,value",
    [
        ("ppo.learning_rate", 0.5),
        ("ppo.num_timesteps", 500_000_000),  # v1's limit; v2 caps PPO at the budget
        ("ppo.lr", 1e-3),
        ("level", "climb"),
        ("network.activation", "gelu"),
        ("reward_weights.jump", 1.0),
        ("foo", 1),
    ],
)
def test_v1_configs_are_still_checked(path, value):
  with pytest.raises(ValidationError):
    _validate(_with(path, value, base=V1_DEFAULT))


def test_v2_config_without_a_version_is_rejected():
  data = {k: v for k, v in DEFAULT.items() if k != "config_version"}
  with pytest.raises(ValidationError, match="config_version is missing"):
    _validate(data)


# -----------------------------------------------------------------------------
# config_hash


def test_hash_is_stable_and_changes_with_config():
  a = _validate(DEFAULT)
  b = _validate(copy.deepcopy(DEFAULT))
  c = _validate(_with("optimizer.learning_rate", 5e-4))
  d = _validate(_with("optimizer", {"type": "rmsprop"}))
  assert a.config_hash() == b.config_hash()
  assert len({a.config_hash(), c.config_hash(), d.config_hash()}) == 3


def test_hash_is_the_same_with_defaults_left_out():
  assert _validate(_minimal("ppo")).config_hash() == _validate(DEFAULT).config_hash()
  hc = _validate(_minimal("hill_climbing")).config_hash()
  assert _validate(_minimal("hill_climbing", optimizer=None)).config_hash() == hc


def test_hash_ignores_key_order():
  shuffled = dict(reversed(list(DEFAULT.items())))
  shuffled["trainer"] = dict(reversed(list(DEFAULT["trainer"].items())))
  assert _validate(shuffled).config_hash() == _validate(DEFAULT).config_hash()


# -----------------------------------------------------------------------------
# The JSON schema (ranges, defaults and tooltips for the app)


def test_schema_exports_ranges_and_tooltips():
  schema = cfglib.TrainConfig.model_json_schema()
  defs = schema["$defs"]
  lr = defs["AdamOptimizer"]["properties"]["learning_rate"]
  assert lr["default"] == 1e-3 and lr["exclusiveMinimum"] == 0.0 and lr["maximum"] == 1e-2
  assert lr["description"]
  steps = defs["PPOTrainer"]["properties"]["num_timesteps"]
  assert steps["maximum"] == steps["default"] == cfglib.STEP_CAPS["ppo"]
  assert defs["CMAESTrainer"]["properties"]["generations"]["maximum"] == cfglib.STEP_CAPS["cma_es"]


def test_schema_describes_every_block_and_setting():
  schema = cfglib.TrainConfig.model_json_schema()
  defs = schema["$defs"]
  for name, block in [("TrainConfig", schema), *defs.items()]:
    assert block.get("description"), name
    for field, prop in block["properties"].items():
      if field != "type":
        # A field that only repeats its block's docstring has it on the block.
        ref = defs.get(prop.get("$ref", "").rsplit("/", 1)[-1], {})
        assert prop.get("description") or ref.get("description"), f"{name}.{field}"


def test_schema_and_validate_cli(monkeypatch, capsys):
  monkeypatch.setattr(sys, "argv", ["config", "--schema"])
  cfglib.main()
  schema = json.loads(capsys.readouterr().out)
  assert set(schema["properties"]) == {
      "config_version", "seed", "level", "brain", "trainer", "optimizer", "sim"
  }
  monkeypatch.setattr(sys, "argv", ["config", str(EXAMPLES[0])])
  cfglib.main()
  assert "valid; hash=" in capsys.readouterr().out


# -----------------------------------------------------------------------------
# What the worker does with a config


def test_brain_maps_to_brax_factory():
  brain = {"type": "mlp", "policy_hidden": [64, 128], "value_hidden": [256, 16, 32],
           "activation": "elu"}
  cfg = _validate(_with("brain", brain))
  factory = train.make_network_factory(cfg.brain)
  assert factory.keywords["policy_hidden_layer_sizes"] == (64, 128)
  assert factory.keywords["value_hidden_layer_sizes"] == (256, 16, 32)
  assert factory.keywords["activation"] is linen.elu
  nets = factory(28, 10)
  assert nets.policy_network is not None


def test_ppo_trainer_maps_to_brax_train_kwargs():
  cfg = _validate(DEFAULT)
  kw = train.ppo_kwargs(cfg)
  for k in ("num_timesteps", "num_envs", "batch_size", "num_minibatches", "unroll_length",
            "num_updates_per_batch", "discounting", "entropy_cost", "reward_scaling",
            "normalize_observations"):
    assert kw[k] == getattr(cfg.trainer, k), k
  assert kw["learning_rate"] == cfg.optimizer.learning_rate
  assert kw["episode_length"] == 400 and kw["seed"] == cfg.seed


def test_worker_refuses_what_it_cannot_train_yet():
  train.check_supported(_validate(DEFAULT))
  for path in EXAMPLES:
    with pytest.raises(NotImplementedError):
      train.check_supported(cfglib.load_config(path))
  with pytest.raises(NotImplementedError, match="Adam"):
    train.check_supported(_validate(_with("optimizer", {"type": "sgd"})))
