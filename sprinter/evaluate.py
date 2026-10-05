"""Deterministic evaluation episodes, gait analysis and pass/fail checks."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jp
import numpy as np

from sprinter.envs import sprint

# Definition of done for the Sprint level.
MIN_DISTANCE_M = 20.0  # in one 10 s episode
MIN_PASSING_EPISODES = 8  # out of 10
# Gait checks (an automated stand-in for watching the video).
MIN_ALTERNATION = 0.8  # share of consecutive touchdowns made by the other foot
MIN_TOUCHDOWNS_PER_FOOT = 3
MIN_AIR_STEPS = 2  # a foot must be off the floor this many control steps to count a touchdown


def rollout(env: sprint.Sprint, policy: Callable, num_episodes: int, seed: int) -> dict[str, np.ndarray]:
  """Runs num_episodes full-length episodes in parallel (no auto-reset).

  Returns time-major numpy arrays with T+1 frames (frame 0 = reset state):
    poses     (T+1, N, nbody-1, 3)  [x, z, angle] of every body except world
    qpos      (T+1, N, nq)          for rendering with MuJoCo
    contacts  (T+1, N, 2)           foot contact flags (right, left)
    site_x    (T+1, N, 2, 2)        heel/toe site x of each foot
    site_z    (T+1, N, 2, 2)        heel/toe site z of each foot
    knee_z    (T+1, N, 2)           knee (shin frame) heights
    done      (T, N)                termination flag after each step
  Steps after an episode terminates keep simulating and should be ignored.
  """
  T = env._config.episode_length  # pylint: disable=protected-access
  m = env.mj_model
  foot_sites = np.array([[m.site(f"{s}_heel").id, m.site(f"{s}_toe").id] for s in sprint.FEET])
  shins = np.array([m.body(f"{s}_shin").id for s in sprint.FEET])

  def frame(state):
    d = state.data
    xmat = d.xmat[:, 1:]
    # Rotation about +y by theta has xmat[0,2] = sin(theta), xmat[0,0] = cos(theta).
    # Exported angle is counter-clockwise positive with +x right, +z up: -theta.
    angle = -jp.arctan2(xmat[..., 0, 2], xmat[..., 0, 0])
    poses = jp.concatenate([d.xpos[:, 1:, 0:1], d.xpos[:, 1:, 2:3], angle[..., None]], axis=-1)
    return {
        "poses": poses,
        "qpos": d.qpos,
        "contacts": jax.vmap(env.foot_contacts)(d),
        "site_x": d.site_xpos[:, foot_sites, 0],
        "site_z": d.site_xpos[:, foot_sites, 2],
        "knee_z": d.xpos[:, shins, 2],
    }

  @jax.jit
  def run(keys, act_key):
    state = jax.vmap(env.reset)(keys)

    def body(carry, _):
      s, k = carry
      k, sub = jax.random.split(k)
      action, _ = policy(s.obs, sub)
      s = jax.vmap(env.step)(s, action)
      out = frame(s)
      out["done"] = s.done
      return (s, k), out

    _, frames = jax.lax.scan(body, (state, act_key), None, length=T)
    return frame(state), frames

  keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
  first, frames = run(keys, jax.random.PRNGKey(seed + 1))
  out = {k: np.concatenate([np.asarray(first[k])[None], np.asarray(frames[k])]) for k in first}
  out["done"] = np.asarray(frames["done"])
  return out


def episode_lengths(done: np.ndarray) -> np.ndarray:
  """Steps taken until (and including) the first termination, per episode."""
  T = done.shape[0]
  hit = done > 0.5
  return np.where(hit.any(axis=0), hit.argmax(axis=0) + 1, T)


def gait_metrics(
    contacts: np.ndarray, site_x: np.ndarray, site_z: np.ndarray, knee_z: np.ndarray, dt: float
) -> dict[str, Any]:
  """Gait statistics for one episode. Arrays cover only the frames while alive.

  contacts (T, 2); site_x, site_z (T, 2 feet, heel/toe); knee_z (T, 2).
  """
  c = contacts > 0.5  # (T, 2)
  events = []  # (frame, foot)
  for f in range(2):
    air = 0
    for t in range(len(c)):
      if c[t, f]:
        if t > 0 and air >= MIN_AIR_STEPS:
          events.append((t, f))
        air = 0
      else:
        air += 1
  events.sort()
  # Both feet landing within one step of each other is a two-footed hop (foot 2).
  merged = []
  for t, f in events:
    if merged and merged[-1][1] not in (f, 2) and t - merged[-1][0] <= 1:
      merged[-1] = (merged[-1][0], 2)
    else:
      merged.append((t, f))
  pairs = list(zip(merged, merged[1:]))
  alternating = sum(1 for (_, f0), (_, f1) in pairs if f0 != f1 and 2 not in (f0, f1))
  n = [sum(1 for _, f in events if f == foot) for foot in range(2)]
  # Same measure as the foot_slip reward term: speed of the grounded foot's lowest point.
  low = np.argmin(site_z[1:], axis=-1)  # (T-1, 2)
  dx = np.take_along_axis(np.diff(site_x, axis=0), low[..., None], axis=-1)[..., 0]
  slip = [np.abs(dx[c[1:, f], f]).mean() / dt for f in range(2) if c[1:, f].any()]
  duration = max(len(c) - 1, 1) * dt
  return {
      "touchdowns": {"right": n[0], "left": n[1]},
      "alternation": round(alternating / len(pairs), 3) if pairs else 0.0,
      "cadence_steps_per_s": round(len(events) / duration, 2),
      "flight_fraction": round(float(np.mean(~c[:, 0] & ~c[:, 1])), 3),
      "double_support_fraction": round(float(np.mean(c[:, 0] & c[:, 1])), 3),
      "duty_factor": {"right": round(float(c[:, 0].mean()), 3), "left": round(float(c[:, 1].mean()), 3)},
      "foot_slip_mps": round(float(np.mean(slip)), 3) if slip else 0.0,
      "min_knee_height_m": round(float(knee_z.min()), 3),
  }


def evaluate(env: sprint.Sprint, policy: Callable, num_episodes: int = 10, seed: int = 0) -> dict[str, Any]:
  traj = rollout(env, policy, num_episodes, seed)
  dt = env.dt
  lengths = episode_lengths(traj["done"])
  T = traj["done"].shape[0]
  episodes = []
  for i in range(num_episodes):
    L = int(lengths[i])
    fell = bool(traj["done"][L - 1, i] > 0.5)
    x = traj["poses"][:, i, 0, 0]  # torso x
    distance = float(x[L] - x[0])
    alive = slice(0, L + 1)
    gait = gait_metrics(
        traj["contacts"][alive, i], traj["site_x"][alive, i], traj["site_z"][alive, i],
        traj["knee_z"][alive, i], dt,
    )
    episodes.append({
        "episode": i,
        "distance_m": round(distance, 2),
        "fell": fell,
        "seconds": round(L * dt, 3),
        "mean_speed_mps": round(distance / (L * dt), 2),
        "passed": (not fell) and distance >= MIN_DISTANCE_M,
        "gait": gait,
    })
  passed = sum(e["passed"] for e in episodes)
  alternation = float(np.mean([e["gait"]["alternation"] for e in episodes]))
  gait_ok = all(
      e["gait"]["alternation"] >= MIN_ALTERNATION
      and min(e["gait"]["touchdowns"].values()) >= MIN_TOUCHDOWNS_PER_FOOT
      for e in episodes if not e["fell"]
  ) and any(not e["fell"] for e in episodes)
  distances = [e["distance_m"] for e in episodes]
  return {
      "num_episodes": num_episodes,
      "episode_seconds": T * dt,
      "distance_m": {
          "mean": round(float(np.mean(distances)), 2),
          "min": round(float(np.min(distances)), 2),
          "max": round(float(np.max(distances)), 2),
      },
      "mean_speed_mps": round(float(np.mean([e["mean_speed_mps"] for e in episodes])), 2),
      "falls": sum(e["fell"] for e in episodes),
      "passed_episodes": passed,
      "distance_ok": passed >= MIN_PASSING_EPISODES,
      "mean_alternation": round(alternation, 3),
      "gait_ok": gait_ok,
      "criteria": {
          "min_distance_m": MIN_DISTANCE_M,
          "min_passing_episodes": MIN_PASSING_EPISODES,
          "min_alternation": MIN_ALTERNATION,
          "min_touchdowns_per_foot": MIN_TOUCHDOWNS_PER_FOOT,
      },
      "episodes": episodes,
      "_trajectories": traj,
  }


def strip_trajectories(result: dict[str, Any]) -> dict[str, Any]:
  return {k: v for k, v in result.items() if not k.startswith("_")}


def describe(result: dict[str, Any]) -> str:
  d = result["distance_m"]
  return (
      f"Eval ({result['num_episodes']} episodes, {result['episode_seconds']:.0f} s): "
      f"distance mean {d['mean']:.1f} m (min {d['min']:.1f}, max {d['max']:.1f}), "
      f"falls {result['falls']}, passed {result['passed_episodes']}/{result['num_episodes']} "
      f"(need {MIN_PASSING_EPISODES}), alternation {result['mean_alternation']:.2f}, "
      f"gait {'OK' if result['gait_ok'] else 'CHECK'}"
  )
