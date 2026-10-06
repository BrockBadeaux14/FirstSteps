"""Watch a trained runner live in MuJoCo's interactive viewer.

  uv run python -m sprinter.play --run runs/final-seed1
  uv run python -m sprinter.play --run runs/final-seed1 --checkpoint before --speed 0.5

The policy and physics run exactly as in training and evaluation (the Sprint env
on the GPU, same backend). Each control step is drawn in a viewer window in real
time. Episodes restart on their own after 10 s, or 1 s after a fall.

In the window: Space pauses, Enter restarts the episode. Left-drag orbits,
right-drag pans and scroll zooms; the camera keeps following the runner.
"""

from __future__ import annotations

import os

# Set before JAX starts. (No MUJOCO_GL here: the viewer opens its own window.)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import argparse  # pylint: disable=g-import-not-at-top,wrong-import-position
import copy
from pathlib import Path
import threading
import time

import jax
import mujoco
import mujoco.viewer
import numpy as np

from sprinter import config as cfglib
from sprinter import policy as policylib
from sprinter import train as trainlib

AFTER_FALL_S = 1.0  # keep showing the body this long after a fall
KEY_SPACE, KEY_ENTER = 32, 257  # GLFW key codes


class LivePolicy:
  """A trained policy driving the Sprint env, one control step at a time."""

  def __init__(self, run_dir: str | Path, checkpoint: str = "final", seed: int = 0):
    run_dir = Path(run_dir).resolve()
    self.cfg = cfglib.load_config(run_dir / "config.json")
    self.env = trainlib.make_env(self.cfg, num_envs=1)
    self.checkpoint_step, path = policylib.pick_checkpoint(run_dir, checkpoint)
    policy = policylib.load_policy(
        self.cfg, path, int(self.env.observation_size), self.env.action_size
    )
    self.dt = self.env.dt
    self.episode_steps = self.cfg.sim.episode_length
    self._reset = jax.jit(self.env.reset)
    self._step = jax.jit(lambda state, key: self.env.step(state, policy(state.obs, key)[0]))
    self._key = jax.random.PRNGKey(seed)
    self.episode = 0
    self.reset()

  def _new_key(self) -> jax.Array:
    self._key, key = jax.random.split(self._key)
    return key

  def reset(self) -> None:
    self.state = self._reset(self._new_key())
    self._next = None  # the following step, already queued on the GPU
    self.episode += 1
    self.t = 0  # control steps taken
    self.fell_at = None  # control step of the fall, if any
    self.x0 = float(self.state.data.qpos[0])
    self.distance = 0.0  # frozen at the fall, like the level score

  def step(self) -> tuple[np.ndarray, np.ndarray]:
    """Advances one control step; returns (qpos, qvel) for drawing."""
    if self._next is None:
      self._next = self._step(self.state, self._new_key())
    self.state = self._next
    # Queue the step after this one now (JAX dispatch is asynchronous), so the
    # GPU computes it while the caller draws this frame.
    self._next = self._step(self.state, self._new_key())
    self.t += 1
    qpos = np.asarray(self.state.data.qpos)  # waits for this step only
    if self.fell_at is None:
      self.distance = float(qpos[0]) - self.x0
      if float(self.state.done) > 0.5:
        self.fell_at = self.t
    return qpos, np.asarray(self.state.data.qvel)

  @property
  def seconds(self) -> float:
    return self.t * self.dt

  @property
  def finished(self) -> bool:
    if self.fell_at is not None:
      return self.t >= self.fell_at + round(AFTER_FALL_S / self.dt)
    return self.t >= self.episode_steps

  def summary(self) -> str:
    if self.fell_at is not None:
      return (f"episode {self.episode}: fell at {self.fell_at * self.dt:.1f} s "
              f"after {self.distance:.1f} m")
    return (f"episode {self.episode}: {self.distance:.1f} m in {self.seconds:.0f} s "
            f"({self.distance / self.seconds:.1f} m/s)")


def _overlay(live: LivePolicy, paused: bool):
  if live.fell_at is not None:
    status = "fell"
  else:
    status = "paused" if paused else "running"
  speed = live.distance / max(live.fell_at or live.t, 1) / live.dt
  left = "Episode\nTime\nDistance\nMean speed\nCheckpoint\n"
  right = (f"{live.episode} ({status})\n{live.seconds:4.1f} s\n{live.distance:5.1f} m\n"
           f"{speed:4.1f} m/s\n{live.checkpoint_step / 1e6:.0f}M steps\n")
  return [
      (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right),
      (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
       "Space pause   Enter restart   drag to orbit, scroll to zoom", ""),
  ]


def display_model(model: mujoco.MjModel, shadows: bool) -> mujoco.MjModel:
  """Copy of the model for drawing only (physics runs in the env).

  Without a GPU OpenGL driver (WSLg falls back to llvmpipe, software rendering
  on the CPU), shadows and the floor reflection can stall a frame for ~15 ms,
  enough to miss the 25 ms real-time budget. They are off unless asked for.
  """
  m = copy.deepcopy(model)
  if not shadows:
    m.light_castshadow[:] = 0
    m.mat_reflectance[:] = 0
  return m


def watch(live: LivePolicy, speed: float = 1.0, episodes: int = 0, shadows: bool = False) -> None:
  """Opens the viewer and plays episodes until the window closes."""
  m = display_model(live.env.mj_model, shadows)
  d = mujoco.MjData(m)
  keys = {"paused": False, "restart": False}

  def on_key(key: int) -> None:
    if key == KEY_SPACE:
      keys["paused"] = not keys["paused"]
    elif key == KEY_ENTER:
      keys["restart"] = True

  # Compile the step before the window opens, so playback starts smoothly.
  live.step()
  live.episode -= 1
  live.reset()

  viewer = mujoco.viewer.launch_passive(
      m, d, key_callback=on_key, show_left_ui=False, show_right_ui=False
  )
  try:
    with viewer.lock():
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = m.body("torso").id
      viewer.cam.distance = 5.0
      viewer.cam.azimuth = 90.0  # look along +y: the runner moves left to right
      viewer.cam.elevation = -12.0
    next_frame = time.perf_counter()
    played = 0.0  # wall-clock seconds of playback this episode (pauses excluded)
    while viewer.is_running():
      if live.finished:
        rate = live.seconds / played if played else 0.0
        print(f"{live.summary()}  [played at {rate:.2f}x real time]", flush=True)
        if episodes and live.episode >= episodes:
          break
        live.reset()
        played = 0.0
      elif keys["restart"]:
        keys["restart"] = False
        live.reset()
        played = 0.0
      if keys["paused"]:
        viewer.set_texts(_overlay(live, paused=True))
        viewer.sync()
        time.sleep(0.05)
        next_frame = time.perf_counter()
        continue

      frame_start = time.perf_counter()
      qpos, qvel = live.step()
      with viewer.lock():
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        d.time = live.seconds
        mujoco.mj_forward(m, d)
      viewer.set_texts(_overlay(live, paused=False))
      viewer.sync()

      # Real-time pacing; if the GPU falls behind, carry on rather than catch up.
      next_frame += live.dt / speed
      delay = next_frame - time.perf_counter()
      if delay > 0:
        time.sleep(delay)
      else:
        next_frame = time.perf_counter()
      played += time.perf_counter() - frame_start
  finally:
    viewer.close()
    _wait_for_viewer_thread()


def _wait_for_viewer_thread(timeout: float = 10.0) -> None:
  """Lets the viewer's render thread destroy its window before Python exits.

  Otherwise glfw.terminate() (an exit hook) can pull the window out from under
  a frame still being drawn: X11 then reports GLXBadDrawable and the process hangs.
  """
  for thread in threading.enumerate():
    if thread is not threading.current_thread() and "_launch_internal" in thread.name:
      thread.join(timeout)


def main() -> None:
  p = argparse.ArgumentParser(description="Watch a trained runner live in the MuJoCo viewer.")
  p.add_argument("--run", required=True, help="run directory, e.g. runs/final-seed1")
  p.add_argument("--checkpoint", default="final",
                 help='"final" (default), "before" (~10%% of training) or a step number')
  p.add_argument("--speed", type=float, default=1.0, help="playback speed, 0.5 = slow motion")
  p.add_argument("--episodes", type=int, default=0, help="stop after N episodes (0 = until closed)")
  p.add_argument("--seed", type=int, default=0, help="random seed for the start poses")
  p.add_argument("--shadows", action="store_true",
                 help="draw shadows and floor reflections (slow with software OpenGL)")
  a = p.parse_args()

  if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    raise SystemExit("No display found. Run this from a desktop session "
                     "(on Windows, a WSL2 Ubuntu shell provides one through WSLg).")
  if a.speed <= 0:
    raise SystemExit("--speed must be positive")
  trainlib.require_gpu()
  live = LivePolicy(a.run, a.checkpoint, a.seed)
  print(f"{Path(a.run).name}: checkpoint at {live.checkpoint_step:,} steps. "
        "Close the window (or Ctrl+C) to quit.", flush=True)
  try:
    watch(live, speed=a.speed, episodes=a.episodes, shadows=a.shadows)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
