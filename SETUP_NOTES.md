# Setup notes

Everything that was installed for this project, in order, with exact versions.
Date: 2026-10-05.

## Step 0: Windows host

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| GPU | NVIDIA GeForce RTX 3070, 8 GB (WDDM) |
| Windows NVIDIA driver | 610.74 (CUDA UMD 13.3) |
| VRAM already used by the Windows desktop at idle | ~2.6-2.7 GB of 8 GB |
| WSL distros | `Ubuntu` (default, WSL2), `Ubuntu-22.04` (WSL2), `docker-desktop` |

Commands run (normal PowerShell, no admin):

```powershell
nvidia-smi
wsl --status
wsl -l -v
```

Nothing was installed on the Windows side. No NVIDIA driver was installed inside WSL:
the Windows driver exposes the GPU through `/usr/lib/wsl/lib` (`libcuda.so.1`, `nvidia-smi`).

## Step 1: WSL environment

| Item | Value |
|---|---|
| Distro | `Ubuntu` = Ubuntu 24.04.4 LTS |
| Kernel | 6.6.87.2-microsoft-standard-WSL2 |
| `nvidia-smi` inside WSL | driver 610.53 (WSL user-mode), KMD 610.74, CUDA 13.3, RTX 3070 visible |
| `libcuda` | `/usr/lib/wsl/lib/libcuda.so.1` (found by `ldconfig -p`) |
| RAM / CPUs visible to WSL | 30 GB / 16 |

### System packages

Already present: `git 1:2.43.0-1ubuntu7.3`, `libegl1 1.7.0-1build1`, `libgl1 1.7.0-1build1`, `curl`.

`sudo` inside WSL asks for a password, so the missing packages were installed by
starting the distro as root from Windows (`wsl -u root`). That needs no admin
PowerShell and is limited to the WSL distro.

```bash
# run from Windows: wsl -d Ubuntu -u root --exec bash -lc '...'
apt-get update
apt-get install -y build-essential ffmpeg libosmesa6
```

Installed: `build-essential 12.10ubuntu1`, `ffmpeg 7:6.1.1-3ubuntu5`,
`libosmesa6 25.1.7-1ubuntu2~24.04.2`.

### uv and Python

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh   # ~/.local/bin is already on PATH
uv python install 3.12                                                    # uv-managed CPython 3.12.15
```

uv 0.12.23.

### Project

The project lives in the Linux filesystem at `~/projects/rl-humanoid` (not under `/mnt/c`).
It is a clone of the FirstSteps repo (`git clone /mnt/c/Users/brock/Documents/GitHub/FirstSteps`),
with `origin` pointing at `https://github.com/BrockBadeaux14/FirstSteps.git` and a
`windows` remote pointing at the Windows checkout.

```bash
cd ~/projects/rl-humanoid
uv venv --python 3.12 --managed-python
uv add "jax[cuda12]"
uv add playground
uv add mediapy matplotlib pydantic pytest
# then, after the smoke test failed on JAX 0.11.2 (see below):
uv add "jax[cuda12]>=0.9.2,<0.10" "mujoco-mjx[warp]>=3.14.0"
```

Final versions (pinned in `uv.lock`):

| Package | Version |
|---|---|
| jax | 0.9.2 |
| jaxlib | 0.9.2 |
| jax-cuda12-plugin / jax-cuda12-pjrt | 0.9.2 |
| CUDA wheels | cublas 12.9.2.10, cudnn 9.27.0.42, nvcc/nvrtc 12.9.86, cuda-runtime 12.9.79 |
| mujoco | 3.14.0 |
| mujoco-mjx | 3.14.0 |
| warp-lang | 1.17.0 |
| brax | 0.14.2 |
| playground (MuJoCo Playground) | 0.2.0 |
| flax | 0.12.6 |
| optax | 0.2.8 |
| orbax-checkpoint | 0.12.6 |
| numpy | 2.5.3 |
| mediapy | 1.2.7 |
| matplotlib | 3.11.2 |
| pydantic | 2.13.5 |
| pytest | 9.1.1 |

Why these pins:

* **JAX 0.9.2, not the latest 0.11.2.** Brax 0.14.2 (the newest release, and what
  Playground 0.2.0 requires) calls `jax.device_put_replicated`, which JAX removed in
  0.10.0 (April 2026). With JAX 0.11.2, `ppo.train` crashes with
  `AttributeError: jax.device_put_replicated is deprecated`. Brax's GitHub main has the
  fix but it is unreleased. 0.9.2 is the last JAX release that works with Brax 0.14.2.
  Playground's own `uv.lock` pins jax 0.6.2; 0.9.2 is the newest version that still works.
* **warp-lang 1.17.0, not 1.18.0.** `mujoco-mjx 3.14.0[warp]` pins 1.17.0, and
  Playground main imports 1.17.0. 1.18.0 was released on 2026-10-05 and is untested with MJX.

### Checkpoint 1: JAX on the GPU

```bash
uv run python -c "import jax; print(jax.devices())"
# [CudaDevice(id=0)]
```

A jitted 4096x4096 float32 matmul runs on `cuda:0` (NVIDIA GeForce RTX 3070):
7.6 ms per call, about 18 TFLOP/s (TF32, JAX's default matmul precision on Ampere).
(Measured with JAX 0.11.2 before the downgrade; 0.9.2 sees the same `CudaDevice(id=0)`.)

GPU memory: by default JAX preallocates 75% of the 8 GB (6 GB). With ~2.6 GB already
used by Windows, that reservation fails and XLA logs `CUDA_ERROR_OUT_OF_MEMORY` at
startup, then continues with a smaller pool. Everything here therefore runs with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (memory grows on demand). Playground's trainer
sets the same flag. Peak use stays around 1 GB in the XLA allocator and about 2 GB in
total including Warp and the CUDA context, so no OOM was hit and
`XLA_PYTHON_CLIENT_MEM_FRACTION` was not needed.

## Step 2: smoke test with Playground's planar walker

Playground 0.2.0's trainer is `train-jax-ppo` (`learning/train_jax_ppo.py`). Its
`--impl` flag defaults to `jax`. WalkerRun's PPO config has `num_evals=10` and
`num_resets_per_eval=10`, so Brax rounds the step budget up to whole updates
(983,040 env steps each). `--num_timesteps 20000000` alone would run 88M steps, so
the smoke test used `--num_evals 3 --num_timesteps 19660800`, which is exactly 20 updates.

`scripts/smoke_walker.py` runs that trainer unchanged, adding only timing logs
(every Brax metric per eval, JAX compile events, peak memory). `scripts/with_gpu_mem.sh`
samples `nvidia-smi` during the run.

```bash
PYTHONUNBUFFERED=1 bash scripts/with_gpu_mem.sh uv run python scripts/smoke_walker.py \
    --env_name WalkerRun --impl warp --num_timesteps 19660800 --num_evals 3 --logdir runs/smoke
# and the same with --impl jax
```

### Warp slowdown and fix

The first Warp run averaged only **38k steps/s**. MJX 3.14's Warp backend prints
"solver/linesearch iterations limit reached" from the GPU every time the solver hits
its (deliberately small) iteration limit, which floods the log and slows every step.
Playground main disables these two warnings in `mjx_env.put_model` (commit `fb246cd`,
September 2026), but the 0.2.0 release does not. This project applies the same
mask (`sprinter.envs.sprint.quiet_warp_overflow`). The smoke script applies it to the
walker too, unless `SMOKE_KEEP_WARP_WARNINGS=1` is set.
Raw env throughput (`scripts/bench_env.py`, 2048 envs, random actions):

| Env | Warp, warnings on | Warp, warnings off | JAX |
|---|---|---|---|
| WalkerRun | 46k steps/s | 289k steps/s | 98k steps/s |
| Sprint (humanoid2d) | - | 243k steps/s | 48k steps/s |

### Checkpoint 2 results (WalkerRun, 19.66M steps, 2048 envs)

| | Warp | JAX |
|---|---|---|
| Time to first eval (Playground's "Time to JIT compile") | 13.4 s | 63.6 s |
| Estimated total startup + compile (adds the training-epoch compile) | ~24 s | ~81 s |
| JAX compile events, whole process (trace + lower + XLA) | 32.7 s | 129 s |
| Steady training throughput, PPO updates included | ~140k steps/s | ~76k steps/s |
| Total wall time | 211 s | 473 s |
| Peak GPU memory, XLA allocator | 1.00 GB | 0.93 GB |
| Peak GPU memory, nvidia-smi delta over the idle baseline | 1.97 GB | 2.05 GB |
| Eval reward at 19.66M steps | 162.7 | 153.6 |
| Rollout MP4 (`MUJOCO_GL=egl`) | rendered | rendered |

Both backends work on WSL2. EGL rendering worked on the first try, so OSMesa was not needed.
Brax's per-epoch `training/sps` metric reads 0.8-1.3M steps/s, which is wrong. With JAX 0.9 the
epoch timer stops before the GPU work finishes. The numbers above come from
wall-clock time between evals. For reference, the unpatched Warp run (warnings on)
took 629 s and reached eval reward 154.9.

## Steps 3-6: no further installs

Nothing else was installed after Step 2. Environment variables the code sets for itself
(only if not already set):

| Variable | Value | Why |
|---|---|---|
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Windows holds ~2.6 GB of VRAM; grow on demand instead of reserving 75% |
| `JAX_DEFAULT_MATMUL_PRECISION` | `highest` | full FP32 matmuls; Playground's README warns that TF32 on Ampere hurts RL stability |
| `MUJOCO_GL` | `egl` (replay only) | headless rendering; `osmesa` is installed as a fallback |

Other findings that affect the setup:

* **Persistent XLA compile cache and Warp.** `sprinter.train` enables a persistent cache
  (`.jax_cache`), but with the Warp backend it barely helps. Warp's JAX bridge
  (`warp/_src/jax/ffi.py`) stamps every traced FFI call with an incrementing `call_id`, so
  the compiled program differs on every trace. It got 0-3 cache hits per run. See the recompile
  test in the README.
* **Harmless warnings:** `jaxopt` prints a deprecation warning on import (pulled in by Brax),
  and Warp prints one `Module ... load on device` line per kernel (cached in
  `~/.cache/warp/1.17.0` after the first run).
* **WSL2 GPU sharing:** the Windows desktop's VRAM use (2.2-2.9 GB here) varies with open apps.
  Training peaks at about 1.8 GB on top of that, so 8 GB cards have room for one run at a time
  plus the desktop.
