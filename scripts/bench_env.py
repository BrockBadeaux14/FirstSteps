"""Time env.reset and env.step for a batch of envs (no learning).

  uv run python scripts/bench_env.py --env WalkerRun --impl warp --num_envs 2048
  uv run python scripts/bench_env.py --env Sprint --impl jax --num_envs 2048
"""

import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jp
from mujoco_playground import registry
from mujoco_playground import wrapper


def make_env(name: str, impl: str, num_envs: int, quiet: bool):
  from sprinter.envs import sprint  # pylint: disable=g-import-not-at-top
  if name == "Sprint":
    cfg = sprint.default_config()
    cfg.impl = impl
    cfg.naconmax = num_envs * 16
    return sprint.Sprint(cfg), cfg.episode_length
  cfg = registry.get_default_config(name)
  env = registry.load(name, config_overrides={"impl": impl})
  if quiet:  # same warning filter the Sprint env applies
    env._mjx_model = sprint.quiet_warp_overflow(env.mjx_model)  # pylint: disable=protected-access
  return env, cfg.episode_length


def timed(fn, *args, n=5):
  out = fn(*args)
  jax.block_until_ready(out)
  t = time.perf_counter()
  for _ in range(n):
    out = fn(*args)
  jax.block_until_ready(out)
  return out, (time.perf_counter() - t) / n


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--env", default="WalkerRun")
  p.add_argument("--impl", default="warp")
  p.add_argument("--num_envs", type=int, default=2048)
  p.add_argument("--unroll", type=int, default=30)
  p.add_argument("--keep_overflow_warnings", action="store_true",
                 help="Playground envs only: keep Warp's iteration warnings")
  a = p.parse_args()

  env, episode_length = make_env(a.env, a.impl, a.num_envs,
                                 quiet=not a.keep_overflow_warnings)
  env = wrapper.wrap_for_brax_training(env, episode_length=episode_length)
  keys = jax.random.split(jax.random.PRNGKey(0), a.num_envs)

  t = time.perf_counter()
  reset = jax.jit(env.reset)
  state = jax.block_until_ready(reset(keys))
  print(f"reset: first call (compile) {time.perf_counter() - t:.1f}s")
  state, dt = timed(reset, keys)
  print(f"reset: {dt * 1e3:.1f} ms per call for {a.num_envs} envs")

  def unroll(state, key):
    def body(s, k):
      act = jax.random.uniform(k, (a.num_envs, env.action_size), minval=-1, maxval=1)
      return env.step(s, act), None
    return jax.lax.scan(body, state, jax.random.split(key, a.unroll))[0]

  unroll = jax.jit(unroll)
  t = time.perf_counter()
  state = jax.block_until_ready(unroll(state, jax.random.PRNGKey(1)))
  print(f"step: first call (compile) {time.perf_counter() - t:.1f}s")
  state, dt = timed(unroll, state, jax.random.PRNGKey(2))
  sps = a.num_envs * a.unroll / dt
  print(f"step: {dt / a.unroll * 1e3:.2f} ms per step -> {sps:,.0f} env steps/s "
        f"({a.env}, impl={a.impl}, {a.num_envs} envs)")
  mem = jax.devices()[0].memory_stats() or {}
  print(f"XLA peak memory: {mem.get('peak_bytes_in_use', 0) / 1e9:.2f} GB")


if __name__ == "__main__":
  main()
