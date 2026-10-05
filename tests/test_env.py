"""Checks on the Sprint environment."""

import functools

import jax
import jax.numpy as jp
from mujoco_playground import wrapper
import numpy as np
import pytest

from sprinter import evaluate
from sprinter.envs import sprint

OBS_SIZE = 28


@functools.lru_cache(maxsize=None)
def make_env(impl: str = "warp", **weights) -> sprint.Sprint:
  cfg = sprint.default_config()
  cfg.impl = impl
  cfg.naconmax = 64 * 16
  for k, v in weights.items():
    cfg.reward_weights[k] = v
  return sprint.Sprint(cfg)


@functools.lru_cache(maxsize=None)
def jitted(env):
  return jax.jit(env.reset), jax.jit(env.step)


def test_reset_starts_standing_on_both_feet():
  env = make_env()
  reset, _ = jitted(env)
  state = reset(jax.random.PRNGKey(0))
  assert state.obs.shape == (OBS_SIZE,) and env.observation_size == OBS_SIZE
  assert np.isfinite(np.asarray(state.obs)).all()
  assert 0.88 < float(state.data.xpos[env._torso_id, 2]) < 0.95  # pylint: disable=protected-access
  soles = np.asarray(env._sole_heights(state.data))  # pylint: disable=protected-access
  assert abs(soles.min() - 0.0005) < 1e-4  # lower foot re-seated on the floor
  assert soles.max() < 0.02  # the other one (joint noise) is close to it
  assert float(state.obs[-2:].max()) == 1.0
  assert set(state.metrics) == {f"reward/{k}" for k in sprint.REWARD_TERMS} | {
      "distance_x", "speed_per_step", "fell"
  }
  assert float(state.done) == 0.0


def test_reset_is_random_but_reproducible():
  env = make_env()
  reset, _ = jitted(env)
  a = reset(jax.random.PRNGKey(1)).data.qpos
  b = reset(jax.random.PRNGKey(1)).data.qpos
  c = reset(jax.random.PRNGKey(2)).data.qpos
  assert np.allclose(a, b)
  assert not np.allclose(a, c)


def test_reward_is_weighted_sum_of_named_terms():
  env = make_env()
  reset, step = jitted(env)
  state = reset(jax.random.PRNGKey(0))
  action = jp.full((env.action_size,), 0.5)
  state = step(state, action)
  parts = {k: float(state.metrics[f"reward/{k}"]) for k in sprint.REWARD_TERMS}
  assert np.isclose(float(state.reward), sum(parts.values()), atol=1e-5)
  assert np.isclose(parts["alive"], 1.0)
  assert np.isclose(parts["control_cost"], -1e-3 * 10 * 0.25)
  assert parts["upright"] == 0.0  # weight 0 in the defaults
  dx = float(state.metrics["distance_x"])
  assert np.isclose(parts["forward_velocity"], dx / env.dt, atol=1e-4)


def test_reward_weights_come_from_config():
  env = make_env(alive=2.5, upright=1.0, control_cost=0.0)
  reset, step = jitted(env)
  state = step(reset(jax.random.PRNGKey(0)), jp.ones(env.action_size))
  assert np.isclose(float(state.metrics["reward/alive"]), 2.5)
  assert float(state.metrics["reward/upright"]) > 0.9
  assert float(state.metrics["reward/control_cost"]) == 0.0


@pytest.mark.parametrize(
    "joint,value", [("rootz", -0.45), ("rooty", 1.3), ("rooty", -1.3)]
)
def test_terminates_when_fallen(joint, value):
  env = make_env()
  reset, step = jitted(env)
  state = reset(jax.random.PRNGKey(0))
  idx = env.mj_model.joint(joint).qposadr[0]
  state = state.replace(data=state.data.replace(qpos=state.data.qpos.at[idx].add(value)))
  state = step(state, jp.zeros(env.action_size))
  assert float(state.done) == 1.0
  assert float(state.metrics["fell"]) == 1.0
  assert float(state.metrics["reward/alive"]) == 0.0


def test_brax_training_wrapper_runs_batched_episodes():
  env = make_env()
  wrapped = wrapper.wrap_for_brax_training(env, episode_length=20)
  keys = jax.random.split(jax.random.PRNGKey(0), 8)
  state = jax.jit(wrapped.reset)(keys)
  step = jax.jit(wrapped.step)
  for i in range(25):
    action = jax.random.uniform(jax.random.PRNGKey(i), (8, env.action_size), minval=-1, maxval=1)
    state = step(state, action)
  assert state.obs.shape == (8, OBS_SIZE)
  assert np.isfinite(np.asarray(state.obs)).all()
  assert np.asarray(state.info["steps"]).max() <= 20


@pytest.mark.parametrize("impl", ["warp", "jax"])
def test_both_backends_agree_on_first_steps(impl):
  """Same start and actions: warp and jax should give nearly the same motion."""
  env = make_env(impl)
  reset, step = jitted(env)
  state = reset(jax.random.PRNGKey(3))
  for i in range(10):
    state = step(state, 0.3 * jp.sin(jp.arange(10.0) + i))
  qpos = np.asarray(state.data.qpos)
  ref_env = make_env("jax" if impl == "warp" else "warp")
  ref_reset, ref_step = jitted(ref_env)
  ref = ref_reset(jax.random.PRNGKey(3))
  for i in range(10):
    ref = ref_step(ref, 0.3 * jp.sin(jp.arange(10.0) + i))
  assert np.abs(qpos - np.asarray(ref.data.qpos)).max() < 0.05


def test_evaluate_reports_distance_and_falls():
  env = make_env()

  def stand_still(obs, key):
    del key
    return jp.zeros(obs.shape[:-1] + (env.action_size,)), {}

  result = evaluate.evaluate(env, stand_still, num_episodes=3, seed=0)
  assert result["num_episodes"] == 3
  # A limp humanoid collapses: every episode should end in a fall, short of 20 m.
  assert result["falls"] == 3
  assert result["passed_episodes"] == 0 and not result["distance_ok"]
  for e in result["episodes"]:
    assert e["seconds"] < 10.0 and abs(e["distance_m"]) < 2.0


def test_alternation_term_rewards_running_and_penalizes_hopping():
  env = make_env()
  t = np.arange(200)
  patterns = {
      "run": np.stack([(t % 20) < 6, ((t + 10) % 20) < 6], axis=1),
      "two_foot_hop": np.stack([(t % 20) < 6, (t % 20) < 6], axis=1),
      "one_leg_hop": np.stack([(t % 20) < 6, np.zeros_like(t, dtype=bool)], axis=1),
      "tap_dance": np.stack([(t % 4) < 2, ((t + 2) % 4) < 2], axis=1),  # 0.05 s in the air
  }
  totals = {}
  for name, pattern in patterns.items():
    last, air, was = jp.array(-1, jp.int32), jp.zeros(2), jp.array([True, True])
    total = 0.0
    for down in pattern:
      down = jp.array(down)
      value, last, air = env._alternation(was, down, air, last)  # pylint: disable=protected-access
      was = down
      total += float(value)
    totals[name] = total
  assert totals["run"] >= 15  # ~19 alternating landings at full bonus
  assert totals["two_foot_hop"] <= -9
  assert totals["one_leg_hop"] <= -8
  assert totals["tap_dance"] == 0.0  # steps too short to count


def test_gait_metrics_tell_running_from_hopping():
  dt = 0.025
  t = np.arange(200)
  run = np.stack([(t % 20) < 8, ((t + 10) % 20) < 8], axis=1).astype(float)
  hop = np.stack([(t % 20) < 8, (t % 20) < 8], axis=1).astype(float)
  one_leg = np.stack([(t % 20) < 8, np.zeros_like(t)], axis=1).astype(float)
  x = np.zeros((200, 2, 2))
  z = np.zeros((200, 2, 2))
  knee = np.full((200, 2), 0.5)
  assert evaluate.gait_metrics(run, x, z, knee, dt)["alternation"] > 0.95
  assert evaluate.gait_metrics(hop, x, z, knee, dt)["alternation"] < 0.1
  assert evaluate.gait_metrics(one_leg, x, z, knee, dt)["alternation"] < 0.1
  # A foot dragged along at 2 m/s while "down" shows up as slip.
  drag = np.stack([np.ones(200), np.zeros(200)], axis=1)
  x_drag = np.repeat((np.arange(200) * 2.0 * dt)[:, None, None], 2, axis=1).repeat(2, axis=2)
  assert abs(evaluate.gait_metrics(drag, x_drag, z, knee, dt)["foot_slip_mps"] - 2.0) < 1e-6


def test_foot_slip_penalizes_a_dragged_foot_only():
  env = make_env(foot_slip=1.0)
  reset, _ = jitted(env)
  s0 = reset(jax.random.PRNGKey(0)).data
  moved = s0.replace(site_xpos=s0.site_xpos.at[:, 0].add(0.05))  # every site 5 cm forward
  down = jp.array([True, False])
  slip = float(env._foot_slip(s0, moved, down))  # pylint: disable=protected-access
  assert np.isclose(slip, 0.05 / env.dt)  # only the grounded (right) foot counts
  assert float(env._foot_slip(s0, s0, jp.array([True, True]))) == 0.0  # pylint: disable=protected-access
