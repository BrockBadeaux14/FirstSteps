"""Step 2 smoke test: MuJoCo Playground's own PPO trainer, with timing logs added.

All command-line flags are passed straight through to Playground's
`train-jax-ppo` (learning/train_jax_ppo.py). This wrapper only adds:
  * one `PROGRESS {...}` JSON line per eval with every metric Brax reports
    (including training/sps), plus seconds since ppo.train() started;
  * JAX compile-event totals (trace / lower / backend compile) and the XLA
    allocator's peak GPU memory, printed at exit;
  * unless SMOKE_KEEP_WARP_WARNINGS=1, the Warp iteration-warning filter from
    Playground main (see sprinter.envs.sprint.quiet_warp_overflow).

Example:
  uv run python scripts/smoke_walker.py --env_name WalkerRun --impl warp \
      --num_timesteps 19660800 --num_evals 3 --logdir runs/smoke
"""

import atexit
import collections
import json
import os
import time

import jax
from jax import monitoring
from mujoco import mjx
import numpy as np

# Importing the trainer sets XLA_PYTHON_CLIENT_PREALLOCATE=false and MUJOCO_GL=egl
# before the GPU backend initializes.
import learning.train_jax_ppo as trainer  # pylint: disable=g-bad-import-order
from sprinter.envs.sprint import quiet_warp_overflow

if os.environ.get("SMOKE_KEEP_WARP_WARNINGS") != "1":
  _orig_put_model = mjx.put_model
  mjx.put_model = lambda *a, **kw: quiet_warp_overflow(_orig_put_model(*a, **kw))

_T_START = time.monotonic()
_compile_secs = collections.defaultdict(float)
_compile_counts = collections.Counter()


def _on_duration(event, duration, **unused_kwargs):
  if event.startswith(("/jax/core/compile", "/jax/compilation_cache")):
    _compile_secs[event] += duration
    _compile_counts[event] += 1


monitoring.register_event_duration_secs_listener(_on_duration)

_orig_train = trainer.ppo.train


def _train(*args, progress_fn=None, **kwargs):
  t_train = time.monotonic()

  def progress(step, metrics):
    row = {"step": int(step), "t_since_train_start": round(time.monotonic() - t_train, 2)}
    for k, v in metrics.items():
      if np.ndim(v) == 0:
        row[k] = float(v)
    print("PROGRESS " + json.dumps(row), flush=True)
    if progress_fn is not None:
      progress_fn(step, metrics)

  return _orig_train(*args, progress_fn=progress, **kwargs)


trainer.ppo.train = _train


@atexit.register
def _report():
  stats = {
      "total_wall_s": round(time.monotonic() - _T_START, 1),
      "compile_secs": {k: round(v, 2) for k, v in _compile_secs.items()},
      "compile_counts": dict(_compile_counts),
  }
  try:
    mem = jax.devices()[0].memory_stats() or {}
    stats["xla_peak_bytes_in_use_gb"] = round(mem.get("peak_bytes_in_use", 0) / 1e9, 3)
  except Exception as e:  # pylint: disable=broad-except
    stats["xla_memory_error"] = repr(e)
  print("SMOKE_STATS " + json.dumps(stats), flush=True)


if __name__ == "__main__":
  trainer.run()
