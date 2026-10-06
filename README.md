# FirstSteps: a 2D humanoid that learns to sprint

Milestone 1 of a reinforcement-learning game. A planar humanoid ("Sprinter") learns to run
from one JSON config, on one GPU, in about 10 minutes. Training uses a MuJoCo Playground
environment and Brax PPO on JAX, with the MuJoCo Warp physics backend.

The config file is the contract the future app server will send to a GPU worker. Training
streams one line per evaluation to `metrics.jsonl` (the future live-stats feed). Each run
ends with `replay.json` files that a phone can draw without any physics.

## Setup

Linux (or WSL2 Ubuntu 24.04) with an NVIDIA GPU and driver; no CUDA toolkit needed. See
[SETUP_NOTES.md](SETUP_NOTES.md) for the exact versions and the WSL details.

```bash
sudo apt install build-essential git ffmpeg libegl1 libgl1 libosmesa6   # once
curl -LsSf https://astral.sh/uv/install.sh | sh                        # once, if uv is missing
uv sync                                                                  # Python 3.12 + locked deps
```

Check the GPU: `uv run python -c "import jax; print(jax.devices())"` must print a `CudaDevice`.
Training refuses to run on the CPU.

## Train

```bash
uv run python -m sprinter.train --config configs/sprint_default.json
```

This writes `runs/<timestamp>-sprint-s<seed>/`:

| File | What it is |
|---|---|
| `config.json` | the validated config |
| `metrics.jsonl` | one line per eval: timesteps, wall time, eval reward, each reward component, distance, speed, fall rate |
| `checkpoints/<step>/` | Brax (orbax) checkpoint after every eval period |
| `final/<step>/` | the final params |
| `train_stats.json` | versions, GPU, backend, compile time vs training time, steps/s, peak GPU memory |
| `eval.json` | 10 deterministic 10 s episodes of the final policy, with gait statistics |

Options: `--seed N` overrides the config's seed, `--run-dir DIR` picks the output folder, and
`--compile-cache DIR` sets the persistent XLA compilation cache (default `.jax_cache`, `""` to
disable). The cache cuts startup for repeat configs on the `jax` physics backend but barely
helps on `warp` (see the recompile test below). It stays small either way (a few MB).

## Replay

```bash
uv run python -m sprinter.replay --run runs/<id>
```

Adds to the run folder:

| File | What it is |
|---|---|
| `replay.json`, `replay.mp4` | the final policy, one 10 s episode (the median of the 10 eval episodes) |
| `before/replay.json`, `before/replay.mp4` | the checkpoint nearest 10% of training |
| `before_after.mp4` | the two side by side |
| `reward_curve.png` | reward, reward components, distance and falls over training |
| `summary.json` | versions, backend, GPU, compile time, steps/s, wall time, distance, pass/fail |

## Watch live

```bash
uv run python -m sprinter.play --run runs/final-seed1
```

Opens MuJoCo's viewer and plays the trained runner in real time. The policy and physics run
exactly as in training and evaluation: the Sprint env on the GPU, with the same backend. A
new episode starts after 10 s, or 1 s after a fall. The overlay shows the episode, time,
distance, mean speed and checkpoint. Each finished episode is also printed with its playback
rate (about 1.0x on the RTX 3070).

* `--checkpoint before` plays the ~10% checkpoint; `--checkpoint <step>` plays any saved step.
  The default is `final`.
* `--speed 0.5` gives slow motion. `--episodes N` stops after N episodes. `--seed` changes
  the start poses.
* In the window: Space pauses, Enter restarts the episode, left-drag orbits, scroll zooms. The
  camera keeps following the runner.
* It needs a display. In WSL2, WSLg provides one, but its OpenGL is software (llvmpipe), so
  shadows and floor reflections are off by default to hold real time. `--shadows` turns them on.

### replay.json format

Built so the phone needs no physics and no kinematics:

* `skeleton`: written once, one entry per body in a fixed order: `name`, `parent`, `offset`
  (body origin in the parent frame, `[x, z]`), `side` (left/right/center), `depth`, `color`,
  and `geom` (`capsule` with `from`/`to`/`radius`, or `sphere` with `center`/`radius`,
  in the body frame).
* `frames`: one per control step (40 per second): `t` and `poses`, a list with each body's
  world pose `[x, z, angle]` in skeleton order, rounded to 3 decimals.
* `meta`: level, `dt`, units and axes (meters, radians, x forward, z up), the angle convention
  (counter-clockwise positive with +x right and +z up), `draw_order`, distance, `fell`,
  checkpoint step, seed and a hash of the config.

To draw a body: a point `(u, v)` in its frame is at
`(x + u cos a - v sin a, z + u sin a + v cos a)`. A 10 s episode is about 127 KB.

## Tests

```bash
uv run pytest
```

* `tests/test_model.py`: the model loads; human proportions; body parts only collide with
  the floor; a passive drop falls realistically (free-fall check, no energy gain, comes to
  rest); the standing pose rests on the ground and can be held; 1,000 random-action steps give
  no NaNs on both the `warp` and `jax` backends.
* `tests/test_env.py`: reset/step, the reward is the weighted sum of its named terms, weights
  come from the config, termination, the Brax training wrapper, warp and jax agree, gait terms.
* `tests/test_config.py`: the schema accepts the presets and rejects everything else.
* `tests/test_play.py`: checkpoint selection and the live player's stepping and restarts,
  using a run with untrained weights (no window needed).

## Config

`configs/sprint_default.json` is validated by the pydantic schema in `sprinter/config.py`.
The limits mirror the future block editor:

* `network`: `policy_hidden` and `value_hidden`, 1-6 layers each, widths from
  {16, 32, 64, 128, 256}; `activation` from {relu, tanh, swish, elu}. Mapped to Brax's
  `make_ppo_networks` through `network_factory`.
* `ppo`: passed to `brax.training.agents.ppo.train` (timesteps, parallel envs, batch size,
  minibatches, unroll length, updates per batch, learning rate, discount, entropy cost, clip
  range, reward scaling, observation normalization, number of evals).
* `reward_weights`: one weight per named reward term (below).
* `sim`: physics backend (`warp` or `jax`), episode length, control and physics timesteps.

`uv run python -m sprinter.config --schema` prints the JSON schema (ranges, defaults and a
one-line description per field, ready for tooltips). Brax rounds `num_timesteps` up to whole
PPO updates (983,040 steps each with the default batch settings), so the default is an exact
multiple: 58,982,400 = 60 updates.

## The model and the level

`sprinter/assets/humanoid2d.xml`: 1.75 m, 70.1 kg, segment lengths and masses from Winter's
anthropometric tables. Planar root (`rootx`, `rootz` slides and `rooty` hinge, like the
dm_control walker); 10 motors (hips, knees, ankles, shoulders, elbows) with `ctrlrange`
[-1, 1]; knees and elbows bend one way only. Left limbs are blue and right limbs orange.
Only the feet, shins, hands, head and torso touch the floor, and body parts never touch each
other. Hands are separate bodies with grip sites, and feet have heel and toe sites. The
scene is a track from -10 m to 305 m with a line across it every 10 m, a post every 10 m up to
150 m, and a camera that follows the runner.

`sprinter/envs/sprint.py` (`Sprint`, a Playground `MjxEnv` modeled on the dm_control walker):

* 10 s episodes: 400 control steps at `ctrl_dt` 0.025 s, physics at 0.0025 s.
* Reset: standing keyframe plus small joint-angle and velocity noise, feet re-seated on the floor.
* Observations (28): joint angles and velocities, torso height, torso pitch as sin/cos, root
  linear and angular velocity, foot contact flags.
* Termination: hips below 0.65 m, torso pitch beyond 1 rad, or NaN.
* Reward: a weighted sum of named terms, each logged separately:

| Term | Per step | Default weight |
|---|---|---|
| `forward_velocity` | torso x velocity, clipped at 12 m/s | 1.0 |
| `alive` | 1 while upright | 1.0 |
| `upright` | cos(torso pitch) | 0.0 |
| `control_cost` | -sum of squared motor commands | 0.001 |
| `alternation` | on each landing: +1 (scaled by air time, full at 0.25 s) for the other foot than last time, -1 for the same foot or both feet; 0 otherwise | 10.0 |
| `foot_slip` | -squared speed (m²/s²) of each grounded foot's lowest point | 0.5 |
| `symmetry` | -\|right - left\| of each foot's 2 s running-average ground contact | 5.0 |

The first four are the brief's starting terms (Gymnasium Walker2d weights). The last three
were added to fix the reward hacks below. Foot contact is sole height under 5 mm, and a
landing needs 0.1 s of air time first.

* Metrics: `distance_x` (m), mean speed, `fell`. The level score is the distance covered in 10 s.

## Reward hacks found

Each one was spotted in the eval videos and the gait statistics in `eval.json` (landings
per foot, alternation, share of time each foot is on the ground, foot slip), then fixed in the
reward. Every one of these runners passed the 20 m bar, so distance alone would not have caught
any of them. They make good in-game lessons.

1. **One-leg skipping with a high kick** (the brief's starting weights: forward 1.0, alive 1.0,
   control 0.001). 59.6 m in 10 s without falling, but the right leg did all the work: 37
   right-foot landings to 16 left, with the right foot on the ground 48% of the time and the left
   19%. The left leg was thrown forward in a high kick and came down only every other stride.
   Fix: `alternation`, which pays for landing on the other foot and charges for landing on the
   same foot twice or on both feet. A landing needs 0.1 s of air time first, so tap-dancing
   earns nothing.
2. **Toe-drag leap** (after fix 1, at alternation weight 5). Landings alternated (0.97), but the
   runner made long split leaps while dragging its right toe along the track behind it, like a
   third leg for balance. The right foot "touched" the floor 44% of the time and grounded feet
   slid at 1.2 m/s. Fix: `foot_slip`, a penalty on how fast a grounded foot's lowest point moves.
   A planted or rolling foot pivots about that point, but a dragged toe slides. A linear penalty
   (weight 1) cured it for two of three seeds; seed 2 still dragged (slip 1.75 m/s), so the
   penalty is now squared (weight 0.5). A ~6 m/s drag then costs three times the forward reward,
   while the small slips of a landing stay cheap.
3. **Limp** (after fix 2). Seed 2 then learned to vault over a planted right leg and dab the
   left foot down briefly: perfect alternation, no sliding, but the right foot was on the
   ground 37% of the time and the left 10%. Fix: `symmetry`, minus the gap between the two
   feet's running-average ground contact.
4. **Stutter step** (after fix 3). To even out ground time cheaply, the runner sometimes put
   the left foot down twice in a row (alternation fell to 0.78-0.91 per episode). Fix: double
   the `alternation` weight (5 -> 10), so a stutter costs more than the imbalance it hides.

Hacks that did not show up: diving forward at the start (falling ends the episode and loses
the alive bonus), knee-sliding (prevented by the hip-height termination), and glitching through
contacts (no NaNs or energy spikes in any run).

Still imperfect: the arms are not rewarded, so each seed holds them in its own odd pose (one
seed keeps a hand on its head). Seed 2's run is crouched, with knees bent.

## Results (RTX 3070, WSL2)

`configs/sprint_default.json`, Warp backend, 58,982,400 steps, three seeds. Each was trained
with one command from a fresh shell, with no compile cache (cold compile):
`uv run python -m sprinter.train --config configs/sprint_default.json --seed N --run-dir runs/final-seedN --compile-cache ""`.
The table comes from
`uv run python scripts/seed_report.py runs/final-seed0 runs/final-seed1 runs/final-seed2`.

| Seed | Wall time (incl. compile) | Compile | Steps/s | Distance in 10 s (mean, min-max of 10) | Falls | Passed | Alternation | Stance imbalance | Foot slip | Peak GPU mem (XLA) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 10.4 min | 20 s | 98,765 | 58.2 m (57.8-58.6) | 0/10 | 10/10 | 1.00 | 0.02 | 0.46 m/s | 1.20 GB |
| 1 | 10.4 min | 20 s | 98,699 | 67.9 m (67.0-68.3) | 0/10 | 10/10 | 1.00 | 0.02 | 0.48 m/s | 1.20 GB |
| 2 | 10.5 min | 19 s | 98,108 | 47.0 m (46.8-47.5) | 0/10 | 10/10 | 1.00 | 0.06 | 0.81 m/s | 1.20 GB |

* All three meet the bar (>= 20 m in 10 s without falling in >= 8 of 10 episodes) and the
  15-minute stretch goal. Average speeds are 4.7-6.8 m/s. The training evals first show
  >= 20 m with no falls at 23.6M-29.5M steps (4.4-5.4 min into the run), and distance is
  still climbing when training ends.
* Steps/s counts steady training including the periodic evals (about 105k without them).
  Compile is the extra time of the first eval and first training period over steady
  state (JAX's own compile events add up to about 28 s, part of which overlaps other work).
* Total GPU memory per nvidia-smi is about 1.8 GB on top of what Windows uses (XLA pool, Warp
  and the CUDA context). JAX runs with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
* Gait statistics come from `eval.json`. Alternation is the share of landings made by the
  other foot. Stance imbalance is \|right - left\| share of time on the ground. Foot slip is the
  mean sliding speed of grounded feet.

## Recompile test

`uv run python -m sprinter.recompile_test --impl warp` (and `--impl jax`) runs short
trainings (2 PPO updates) where only the learning rate changes, and sums JAX's own compile events
(trace + lower + XLA compile). Results are in `runs/recompile_test_<impl>.json`.

| Case | Warp: compile (XLA programs) | JAX physics: compile (XLA programs) |
|---|---|---|
| First training in a fresh process | 21.9 s (99) | 105.6 s (99) |
| Same config again, same process | 17.4 s (6) | 99.6 s (7) |
| Only `learning_rate` changed, same process | 17.6 s (6) | 99.0 s (7) |
| `learning_rate` changed, persistent XLA cache on | 18.2 s (0 cache hits) | 48.6 s (4 cache hits) |
| New process, warm persistent cache, other seed | 12.6 s (3 cache hits) | 36.9 s (5 cache hits) |

What this means for the server:

* **Yes, it recompiles.** Brax's `ppo.train` bakes the learning rate (and every other PPO
  hyperparameter) into the compiled program as constants and builds new closures on every call.
  Changing only the learning rate costs a full recompile of the 6-7 large programs (training
  epoch, eval rollout, resets). Re-running an identical config in the same process costs the same.
* **With Warp it is cheap.** About 18-22 s per job (3% of a 10-minute job). A warm worker
  process saves only the ~90 small compiles (about 4 s).
* **The persistent cache cannot fix it for Warp.** Warp's JAX bridge stamps every traced FFI
  call with a new `call_id`, so the program text changes on every trace. With the pure-JAX physics
  backend the cache works across processes (106 s -> 37 s), but that backend simulates 2-5x
  slower, which costs far more than it saves.
* Reusing one compiled program across hyperparameter changes would need a custom PPO loop that
  takes them as runtime inputs (for example `optax.inject_hyperparams`), one cached program per
  network shape, and a Warp binding with stable call ids. At ~20 s per job that is not worth it yet.

## Open issues

* **Version pins.** JAX is held at 0.9.x because Brax 0.14.2 (the latest release) uses
  `jax.device_put_replicated`, which JAX 0.10 removed. Lift the pin when Brax ships a release
  with the fix that is already on its main branch.
* **Playground 0.2.0 vs MJX 3.14.** The Warp iteration-warning flood (6x slower steps) is
  patched here (`quiet_warp_overflow`), as on Playground main. Drop the patch when Playground
  makes a new release.
* **Brax's `training/sps` metric is wrong** under JAX 0.9 (0.8-1.3M steps/s reported). Timing
  here uses wall-clock time between evals.
* **Arms do nothing useful.** No reward term involves them, so each seed holds them in its own
  odd pose. A posture or arm-swing term (a future reward block) would make the runners look more
  human. Gait style also varies by seed, from bounding leaps to a crouched quick-step.
* **Late-training dips.** Seed 2 dropped from 50.6 m to 44.2 m at 56M steps before recovering to
  47.0 m (seeds 0 and 1 rose steadily). The learning rate is a constant 1e-3; a decaying
  schedule may help.
* **Contact sensing** uses sole height (under 5 mm). That is fine on flat ground. The climbing level
  will need real contact forces (touch sensors or Warp contact data).
* **Training length granularity.** Brax runs whole PPO updates (983,040 steps each) and the
  same number per eval period, so the app's "training length" block should offer multiples of
  `(num_evals - 1) x 983,040` steps. The schema reports `effective_timesteps`.

What I would change about the defaults: `num_timesteps` could go to about 78.6M (+33%, still
under 15 minutes), since distance is still rising at the end of training. FP32 matmuls
(`JAX_DEFAULT_MATMUL_PRECISION=highest`) cost about 8% versus TF32 (80.7k vs 86.9k steps/s on a
short run); they are kept for stability as Playground recommends for Ampere GPUs.

## Repo layout

```
configs/sprint_default.json   default config (the app's contract)
sprinter/
  assets/humanoid2d.xml       the humanoid and the scene
  envs/sprint.py              Sprint environment (MuJoCo Playground MjxEnv)
  config.py                   pydantic schema and presets
  train.py                    config -> Brax PPO, metrics.jsonl, checkpoints, timing
  evaluate.py                 deterministic eval episodes, gait statistics, pass/fail
  replay.py                   replay.json / mp4 / before-after / summary.json
  play.py                     watch a trained runner live in the MuJoCo viewer
  policy.py                   find checkpoints in a run and load them as policies
  plots.py                    reward_curve.png
  recompile_test.py           does a hyperparameter-only change recompile?
  system_info.py              versions and GPU info
scripts/                      Step 2 smoke test, env benchmark, GPU memory sampler
tests/
runs/                         outputs (gitignored)
```
