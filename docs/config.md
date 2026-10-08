# Training config, version 2

The training config is the contract between the app, the server and the GPU worker. The app's
block editor builds one, the server checks it and sends it to a worker, and the worker trains
it. The schema is `sprinter/config.py`. This page explains its shape and the rules that span
blocks. For each setting's range, default and one-line description (the app's tooltips), print
the JSON schema:

```bash
uv run python -m sprinter.config --schema
```

Validate a file with `uv run python -m sprinter.config <file>`. The levels, brains and trainers
come from [game-design.md](game-design.md).

## Shape

```json
{
  "config_version": 2,
  "seed": 0,
  "level": {"type": "sprint", "reward_weights": {"forward_velocity": 1.0, "alive": 1.0}},
  "brain": {"type": "mlp", "policy_hidden": [32, 32, 32, 32], "activation": "swish"},
  "trainer": {"type": "ppo", "num_timesteps": 58982400, "num_envs": 2048},
  "optimizer": {"type": "adam", "learning_rate": 0.001},
  "sim": {"impl": "warp", "episode_seconds": 10}
}
```

| Block | What it sets | Members (`type`) |
|---|---|---|
| `config_version` | Always `2` | |
| `seed` | Random seed | |
| `level` | The task and its reward weights | `sprint` |
| `brain` | What the runner learns | `rhythm`, `mlp` |
| `trainer` | How it learns, with that trainer's settings | `hill_climbing`, `ga`, `pso`, `cma_es`, `bc`, `reinforce`, `ppo` |
| `optimizer` | How a gradient moves the weights | `sgd`, `rmsprop`, `adam` |
| `sim` | Physics backend and timing | |

* `level`, `brain`, `trainer` and `optimizer` are tagged unions: `type` picks the member, and
  the block's other fields are that member's settings. A setting from another member is
  rejected, like any unknown field.
* A setting left out gets its default. `config_version`, `level`, `brain`, `trainer` and each
  block's `type` are required. `optimizer` and `brain.value_hidden` may also be `null`, which
  is the same as leaving them out.

## Levels

| `type` | Level | Settings |
|---|---|---|
| `sprint` | Sprint | `reward_weights`: one weight per named reward term in `sprinter/envs/sprint.py` |

Each level card adds its level with its own reward-weights block. See
[Adding a level](#adding-a-level).

## Brains

| `type` | Brain | Settings |
|---|---|---|
| `rhythm` | Rhythm controller: each motor follows a fixed rhythm and no sensors are read | `harmonics` (1-2): sine waves per motor |
| `mlp` | Neural network from the sensor readings to the motors | `policy_hidden`, `value_hidden`, `activation` |

* The rhythm brain has, per motor, an offset plus an amplitude and a phase per harmonic, and
  one shared frequency: 31 numbers with one harmonic (the spike's prototype) and 51 with two.
* `policy_hidden` and `value_hidden` take 1-6 layers with widths from {16, 32, 64, 128, 256}.
  `activation` is one of relu, tanh, swish and elu.
* `value_hidden` is the critic. Only actor-critic trainers (PPO) have one, and they get
  5 layers of 256 when it is left out. Any other trainer rejects it.

## Trainers

| `type` | Trainer | Family | Brain | Training length | Cap | Other settings |
|---|---|---|---|---|---|---|
| `hill_climbing` | Hill climbing | zero-order | `rhythm` | `generations` | 178 | `population_size`, `step_size` |
| `ga` | Genetic algorithm | zero-order | `rhythm` | `generations` | 178 | `population_size`, `elite_fraction`, `mutation_std` |
| `pso` | Particle swarm | zero-order | `rhythm` | `generations` | 178 | `population_size`, `inertia`, `cognitive`, `social` |
| `cma_es` | CMA-ES | zero-order | `rhythm` | `generations` | 178 | `population_size`, `elite_fraction`, `initial_std` |
| `bc` | Copy trainer (behavior cloning) | gradient | `mlp` | `gradient_steps` | 100,000 | `batch_size`, `normalize_observations`, `num_evals`, `num_eval_envs` |
| `reinforce` | REINFORCE | gradient | `mlp` | `num_timesteps` | 58,982,400 | `num_envs`, `discounting`, `entropy_cost`, `normalize_observations`, `num_evals`, `num_eval_envs` |
| `ppo` | PPO | gradient, actor-critic | `mlp` | `num_timesteps` | 58,982,400 | milestone 1's PPO settings, without `learning_rate` |

* Training length is in each trainer's own unit, as [The budget](game-design.md#the-budget)
  sets: generations, gradient steps or environment steps (all parallel simulations together).
  It defaults to the cap, a full budget, and a shorter run is allowed.
* The cap replaces milestone 1's 500 million limit on `num_timesteps`. The JSON schema exports
  each cap as its field's `maximum`.
* Brax runs PPO in whole updates and rounds `num_timesteps` up to fill them, so PPO's cap
  applies to the steps it actually runs (`effective_timesteps`). With the default batch
  settings an update is 983,040 steps.
* TRPO ([#15]) and SAC ([#16]) join the union when their cards are built.

## Optimizers

| `type` | Optimizer | `learning_rate` default | Range |
|---|---|---|---|
| `sgd` | SGD | 0.01 | (0, 1] |
| `rmsprop` | RMSprop | 0.001 | (0, 0.01] |
| `adam` | Adam | 0.001 | (0, 0.01] |

Only gradient-based trainers have an optimizer. When it is left out they get Adam at its
default learning rate, which is what milestone 1 trained with.

## Rules across blocks

1. **Trainer and brain.** Zero-order trainers (`hill_climbing`, `ga`, `pso`, `cma_es`) use the
   `rhythm` brain. Gradient-based trainers (`bc`, `reinforce`, `ppo`) use the `mlp` brain.
2. **Trainer and optimizer.** Zero-order trainers take no optimizer. Gradient-based trainers
   take any of the three.
3. **Value network.** `brain.value_hidden` is only allowed with an actor-critic trainer (`ppo`).
4. **CMA-ES parameter cap.** CMA-ES tunes at most 100 numbers. Its covariance matrix has one
   entry per pair of numbers ([why](game-design.md#cma-es-is-never-paired-with-a-neural-network)).
5. **The budget.** A trainer's training length is at most its cap (the table above).

Inside a block: PPO's `batch_size × num_minibatches` must be a multiple of `num_envs`, and in
`sim`, `ctrl_dt` must be a whole multiple of `sim_dt` and `episode_seconds` of `ctrl_dt`.

The schema does not tie trainers to levels. Any combination that passes these rules is a valid
config. Which trainers a player has on each level is up to the lineup and the
[unlock rules](game-design.md#unlock-rules).

## Placeholders

These values come from the spike ([#9]). Until it reports they are placeholders in
`sprinter/config.py`.

| Value | Placeholder | Where the real value comes from |
|---|---|---|
| Step cap per trainer | `STEP_CAPS`: 178 generations, 100,000 gradient steps, 58,982,400 environment steps | What each trainer reaches in 10 minutes in #9. Then they are stored with the reference scores in `configs/unlocks.json` ([#7]). |
| CMA-ES parameter cap | `CMA_ES_MAX_PARAMS`: 100 | #9 |
| Rhythm brain shape | Offset, amplitude and phase per motor, shared frequency, 1-2 harmonics | The #9 prototype; the Jump card ([#11]) for its rhythm / trajectory brain |

The 178 generations are 2,048 runners per generation at the Sprint env's ~243k steps/s
(`scripts/bench_env.py`). PPO's cap is today's default Sprint config, 10.4 minutes. The settings
of the trainers that are not built yet (everything but PPO) are a first proposal. Each
trainer's card can change them.

## What the worker runs

Any valid config can be sent, but `sprinter.train` trains only PPO with Adam so far. Brax's PPO
always uses Adam. Other trainers and optimizers raise `NotImplementedError` until their cards
add them.

## Version, migration and hash

* **`config_version: 2`** is required. A config without it, or with `config_version: 1`, is read
  as version 1 (milestone 1: Sprint trained with PPO) and migrated before validation:

  | Version 1 | Version 2 |
  |---|---|
  | `level: "sprint"` | `level.type: "sprint"` |
  | `reward_weights` | `level.reward_weights` |
  | `network` | `brain`, with `type: "mlp"` |
  | `ppo` | `trainer`, with `type: "ppo"` |
  | `ppo.learning_rate` | `optimizer`, with `type: "adam"` |

  Fields the v1 file left out stay out, so they keep the same defaults. Run folders from
  milestone 1 load unchanged in `sprinter.replay` and `sprinter.play`.
* **`config_hash`** is the first 16 hex digits of the SHA-256 of the validated version 2
  config as JSON, with sorted keys, defaults filled in and nulls dropped. So a config hashes
  the same in its v1 or v2 form, with its defaults left out or written out, and in any key
  order.
* Hashes written by milestone 1 (`train_stats.json`, `replay.json`, `summary.json`) were taken
  over the version 1 form, so they don't match the version 2 hash of the same config.

## Examples

`configs/sprint_default.json` is the PPO example. `configs/examples/sprint_<trainer>.json` has
one for each other trainer, with settings at their defaults: the rhythm brain for the
zero-order trainers (two harmonics for CMA-ES), SGD for the copy trainer and RMSprop for
REINFORCE. All are on Sprint because it is the only level so far. The tests check every file,
and that together they cover every trainer, brain and optimizer.

## Adding a level

A level card ([#10]-[#14]):

1. Adds `<Level>RewardWeights` and `<Level>Level` (with `type: Literal["<level>"]` and a
   `reward_weights` field) to `sprinter/config.py`.
2. Turns `Level` into a union:
   `Annotated[Union[SprintLevel, <Level>Level], Field(discriminator="type")]`. A Sprint config
   keeps its hash.
3. Has `sprinter.train.make_env` build the level's env from `level.type`.
4. Adds `configs/examples/<level>_<trainer>.json` for the level's trainers in the lineup, and
   tests in `tests/test_config.py`.

[#7]: https://github.com/BrockBadeaux14/FirstSteps/issues/7
[#9]: https://github.com/BrockBadeaux14/FirstSteps/issues/9
[#10]: https://github.com/BrockBadeaux14/FirstSteps/issues/10
[#11]: https://github.com/BrockBadeaux14/FirstSteps/issues/11
[#14]: https://github.com/BrockBadeaux14/FirstSteps/issues/14
[#15]: https://github.com/BrockBadeaux14/FirstSteps/issues/15
[#16]: https://github.com/BrockBadeaux14/FirstSteps/issues/16
