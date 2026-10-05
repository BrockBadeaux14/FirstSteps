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
```

Resolved versions (also pinned in `uv.lock`):

| Package | Version |
|---|---|
| jax | 0.11.2 |
| jaxlib | 0.11.2 |
| jax-cuda12-plugin / jax-cuda12-pjrt | 0.11.2 |
| CUDA wheels | cublas 12.9.2.10, cudnn 9.27.0.42, nvcc/nvrtc 12.9.86, cuda-runtime 12.9.79 |
| mujoco | 3.14.0 |
| mujoco-mjx | 3.14.0 |
| warp-lang | 1.18.0 |
| brax | 0.14.2 |
| playground (MuJoCo Playground) | 0.2.0 |
| flax | 0.12.10 |
| optax | 0.2.8 |
| orbax-checkpoint | 0.12.6 |
| numpy | 2.5.3 |
| mediapy | 1.2.7 |
| matplotlib | 3.11.2 |
| pydantic | 2.13.5 |
| pytest | 9.1.1 |

### Checkpoint 1: JAX on the GPU

```bash
uv run python -c "import jax; print(jax.devices())"
# [CudaDevice(id=0)]
```

A jitted 4096x4096 float32 matmul runs on `cuda:0` (NVIDIA GeForce RTX 3070):
7.6 ms per call, about 18 TFLOP/s (TF32, JAX's default matmul precision on Ampere).

GPU memory: by default JAX preallocates 75% of the 8 GB (6 GB). With ~2.6 GB already
used by Windows, that reservation fails and XLA logs `CUDA_ERROR_OUT_OF_MEMORY` at
startup, then continues with a smaller pool. JAX's memory share is capped for training
(see the GPU memory notes below).
