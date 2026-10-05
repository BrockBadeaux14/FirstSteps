"""Checks on the training-config schema (the app server's contract)."""

import copy
import json

from flax import linen
from pydantic import ValidationError
import pytest

from sprinter import config as cfglib
from sprinter import train

DEFAULT = json.loads(cfglib.DEFAULT_CONFIG_PATH.read_text())


def _with(path: str, value):
  cfg = copy.deepcopy(DEFAULT)
  node = cfg
  *parents, leaf = path.split(".")
  for p in parents:
    node = node[p]
  node[leaf] = value
  return cfg


def test_default_config_is_valid():
  cfg = cfglib.TrainConfig.model_validate(DEFAULT)
  assert cfg.level == "sprint"
  assert cfg.sim.episode_length == 400
  assert cfg.network.policy_hidden == [32, 32, 32, 32]
  assert cfg.network.value_hidden == [256] * 5


def test_hash_is_stable_and_changes_with_config():
  a = cfglib.TrainConfig.model_validate(DEFAULT)
  b = cfglib.TrainConfig.model_validate(copy.deepcopy(DEFAULT))
  c = cfglib.TrainConfig.model_validate(_with("ppo.learning_rate", 5e-4))
  assert a.config_hash() == b.config_hash()
  assert a.config_hash() != c.config_hash()


@pytest.mark.parametrize("widths", [[48], [32, 100], [512], [0]])
def test_rejects_widths_outside_presets(widths):
  with pytest.raises(ValidationError):
    cfglib.TrainConfig.model_validate(_with("network.policy_hidden", widths))


@pytest.mark.parametrize("n", [0, 7])
def test_rejects_too_few_or_too_many_layers(n):
  with pytest.raises(ValidationError):
    cfglib.TrainConfig.model_validate(_with("network.value_hidden", [64] * n))


@pytest.mark.parametrize("n", [1, 6])
def test_accepts_one_to_six_layers(n):
  cfglib.TrainConfig.model_validate(_with("network.policy_hidden", [16] * n))


@pytest.mark.parametrize("act", ["relu", "tanh", "swish", "elu"])
def test_accepts_preset_activations(act):
  cfglib.TrainConfig.model_validate(_with("network.activation", act))


@pytest.mark.parametrize("act", ["gelu", "sigmoid", "", "Swish"])
def test_rejects_other_activations(act):
  with pytest.raises(ValidationError):
    cfglib.TrainConfig.model_validate(_with("network.activation", act))


@pytest.mark.parametrize(
    "path,value",
    [
        ("ppo.learning_rate", -1e-3),
        ("ppo.learning_rate", 0.5),
        ("ppo.discounting", 1.5),
        ("ppo.num_envs", 1000),  # batch_size * num_minibatches not a multiple
        ("ppo.num_timesteps", 10),
        ("ppo.clipping_epsilon", 0.9),
        ("reward_weights.alive", -1.0),
        ("sim.impl", "cpu"),
        ("sim.sim_dt", 0.003),  # ctrl_dt is not a whole multiple
        ("level", "climb"),
    ],
)
def test_rejects_out_of_range_values(path, value):
  with pytest.raises(ValidationError):
    cfglib.TrainConfig.model_validate(_with(path, value))


def test_rejects_unknown_fields():
  with pytest.raises(ValidationError):
    cfglib.TrainConfig.model_validate(_with("ppo.lr", 1e-3))
  with pytest.raises(ValidationError):
    cfglib.TrainConfig.model_validate(_with("reward_weights.jump", 1.0))


def test_schema_exports_ranges_and_tooltips():
  schema = cfglib.TrainConfig.model_json_schema()
  lr = schema["$defs"]["PPOConfig"]["properties"]["learning_rate"]
  assert lr["default"] == 1e-3 and lr["exclusiveMinimum"] == 0.0 and lr["maximum"] == 1e-2
  assert lr["description"]


def test_network_maps_to_brax_factory():
  cfg = cfglib.TrainConfig.model_validate(
      _with("network", {"policy_hidden": [64, 128], "value_hidden": [256, 16, 32], "activation": "elu"})
  )
  factory = train.make_network_factory(cfg.network)
  assert factory.keywords["policy_hidden_layer_sizes"] == (64, 128)
  assert factory.keywords["value_hidden_layer_sizes"] == (256, 16, 32)
  assert factory.keywords["activation"] is linen.elu
  nets = factory(28, 10)
  assert nets.policy_network is not None


def test_ppo_section_maps_to_brax_train_kwargs():
  cfg = cfglib.TrainConfig.model_validate(DEFAULT)
  kw = train.ppo_kwargs(cfg)
  for k in ("num_timesteps", "num_envs", "batch_size", "num_minibatches", "unroll_length",
            "num_updates_per_batch", "learning_rate", "discounting", "entropy_cost",
            "reward_scaling", "normalize_observations"):
    assert kw[k] == getattr(cfg.ppo, k), k
  assert kw["episode_length"] == 400 and kw["seed"] == cfg.seed


def test_effective_timesteps_rounds_up_to_whole_updates():
  cfg = cfglib.TrainConfig.model_validate(DEFAULT)
  p = cfg.ppo
  assert p.effective_timesteps >= p.num_timesteps
  assert p.effective_timesteps % (p.env_steps_per_update * (p.num_evals - 1)) == 0
