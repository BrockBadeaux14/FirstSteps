"""Package versions and GPU details for run summaries."""

from __future__ import annotations

from importlib import metadata
import subprocess
from typing import Any

import jax

PACKAGES = (
    "jax", "jaxlib", "jax-cuda12-plugin", "mujoco", "mujoco-mjx", "warp-lang",
    "brax", "playground", "flax", "optax", "orbax-checkpoint", "numpy", "pydantic",
)


def package_versions() -> dict[str, str]:
  out = {}
  for name in PACKAGES:
    try:
      out[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
      out[name] = "not installed"
  return out


def gpu_info() -> dict[str, Any]:
  dev = jax.devices()[0]
  info: dict[str, Any] = {"jax_backend": jax.default_backend(), "device": dev.device_kind}
  try:
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=True,
    ).stdout.strip().splitlines()[0]
    name, driver, mem = (s.strip() for s in q.split(","))
    info.update(name=name, driver=driver, memory_total=mem)
  except (OSError, subprocess.SubprocessError, IndexError, ValueError):
    pass
  return info
