"""Checks on checkpoint loading and the live player (no window needed)."""

from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
import jax.numpy as jp
import numpy as np
import pytest

from sprinter import config as cfglib
from sprinter import play
from sprinter import policy as policylib
from sprinter import train as trainlib

STEPS = (100, 600, 1000)  # periodic checkpoints in the fake run; 1000 is final


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
  """A run folder with untrained (randomly initialized) network weights."""
  run = tmp_path_factory.mktemp("run")
  cfg = cfglib.load_config(cfglib.DEFAULT_CONFIG_PATH)
  (run / "config.json").write_text(cfg.model_dump_json())
  env = trainlib.make_env(cfg, num_envs=1)
  obs_size, act_size = int(env.observation_size), env.action_size
  factory = trainlib.make_network_factory(cfg.network)
  nets = factory(obs_size, act_size, preprocess_observations_fn=running_statistics.normalize)
  k_policy, k_value = jax.random.split(jax.random.PRNGKey(0))
  params = (
      running_statistics.init_state(specs.Array((obs_size,), jp.dtype("float32"))),
      nets.policy_network.init(k_policy),
      nets.value_network.init(k_value),
  )
  ckpt_config = ppo_checkpoint.network_config(
      observation_size=obs_size, action_size=act_size,
      normalize_observations=True, network_factory=factory,
  )
  for step in STEPS:
    ppo_checkpoint.save(run / "checkpoints", step, params, ckpt_config)
  ppo_checkpoint.save(run / "final", STEPS[-1], params, ckpt_config)
  return run


def test_pick_checkpoint(run_dir):
  assert policylib.pick_checkpoint(run_dir, "final")[0] == 1000
  assert policylib.pick_checkpoint(run_dir, "before")[0] == 100  # nearest 10% of 1000
  assert policylib.pick_checkpoint(run_dir, "600")[0] == 600
  with pytest.raises(ValueError, match="100, 600, 1000"):
    policylib.pick_checkpoint(run_dir, "123")


def test_live_policy_steps_and_restarts(run_dir):
  live = play.LivePolicy(run_dir, "final", seed=0)
  assert live.episode == 1 and live.t == 0
  for _ in range(5):
    qpos, qvel = live.step()
  assert qpos.shape == (live.env.mj_model.nq,) and qvel.shape == (live.env.mj_model.nv,)
  assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
  assert live.seconds == pytest.approx(5 * live.dt)
  assert not live.finished
  live.reset()
  assert live.episode == 2 and live.t == 0 and live.distance == 0.0


def test_display_model_is_a_lighter_copy(run_dir):
  live = play.LivePolicy(run_dir, "final")
  original = live.env.mj_model
  light = play.display_model(original, shadows=False)
  assert light is not original
  assert not light.light_castshadow.any() and not light.mat_reflectance.any()
  assert original.light_castshadow.any() and original.mat_reflectance.any()  # untouched
  fancy = play.display_model(original, shadows=True)
  assert np.array_equal(fancy.light_castshadow, original.light_castshadow)


def test_live_policy_finishes_after_episode_or_fall(run_dir):
  live = play.LivePolicy(run_dir, "before", seed=1)
  steps = 0
  while not live.finished:
    live.step()
    steps += 1
  # Untrained weights fall over quickly; then the player keeps going for 1 s.
  if live.fell_at is not None:
    assert steps == live.fell_at + round(play.AFTER_FALL_S / live.dt)
  else:
    assert steps == live.episode_steps
  assert "episode 1" in live.summary()
