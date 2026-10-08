# FirstSteps game design

The game's levels, the idea each one teaches, the brain the runner has on each, and the
trainers a player starts with and unlocks. This file is the source of truth for the lineup and
the trainer progression. It started as [#4]. Change it with a pull request so the design stays
versioned with the code, and link cards here instead of copying the design into them.

The training pipeline from milestone 1 (the Sprint level, trained with PPO) is documented in
the [README](../README.md).

## The body is 2D

The runner is the planar humanoid in `sprinter/assets/humanoid2d.xml`. Its root has three
degrees of freedom: it moves forward and back (`rootx`), up and down (`rootz`), and tips
forward and back (`rooty`, pitch). Ten motors drive the hips, knees, ankles, shoulders and
elbows.

No level can ask for turning, sidestepping, cartwheels or anything else that needs
side-to-side balance. Every level and every lesson has to fit inside these limits.

## Level lineup

| # | Level | New idea it teaches | Brain | Trainers (start → unlock) |
|---|---|---|---|---|
| 1 | [Crawl][#10] | Trial and error, evolution | Rhythm controller (~30 numbers, no sensors) | Hill climbing (tutorial) → GA → PSO |
| 2 | [Jump][#11] | Joints working together | Rhythm / trajectory (~30-60 numbers) | PSO → CMA-ES |
| 3 | [Walk][#12] | Learning by copying, gradients | Neural network | Copy trainer with SGD → RMSprop → Adam |
| 4 | Sprint | Learning from reward | Neural network + sensors | The copycat → REINFORCE → PPO |
| 5 | [Climb Steps][#13] | Curriculum (low steps first) | Neural network | PPO |
| 6 | [Backflip][#14] | Reward design, reference motion | Neural network | PPO (SAC optional) |

* Each level teaches one new idea, and each unlock follows a failure the player has just seen.
* Trainers unlock in the order shown. When a run earns an unlock is set in
  [Unlock rules](#unlock-rules).
* A level starts with the trainer the player finished the last level with, unless the brain
  changes. PSO carries over from Crawl to Jump, and the copy trainer carries over from Walk to
  Sprint, where it is "the copycat". Walk starts a new trainer because the CMA-ES used on Jump
  cannot train a neural network (see [below](#cma-es-is-never-paired-with-a-neural-network)).
* Sprint is the level built in milestone 1. Its score is the distance covered in 10 s. Each
  other level's score, pass bar, start pose, end conditions and lesson text are set in the
  design sheet on its level card.

The brains:

* **Rhythm controller.** Each motor follows a fixed rhythm. The spike's prototype ([#9]) has
  an amplitude, an offset and a phase per joint, plus one shared frequency (31 numbers). It
  reads no sensors, so it plays the same motion whatever the body is doing.
* **Rhythm / trajectory** (Jump). A small motion pattern of about 30-60 numbers. Its exact
  form comes out of the spike and the Jump card.
* **Neural network.** A network from the body's sensor readings (28 on Sprint) to the 10
  motors. The default Sprint policy, 4 hidden layers of 32, has 4,756 weights. The block
  editor allows up to 6 layers of 256, which is 341,524 weights.

## Eras and what moves the player on

The six levels fall into three eras, one per trainer family: evolution (Crawl, Jump),
gradients (Walk) and reinforcement learning (Sprint, Climb Steps, Backflip). The player moves
to the next era only after watching the current one fail.

### Crawl and Jump → Walk: a fixed rhythm can't balance upright

**The failure.** Crawling and jumping work with a fixed rhythm, but walking upright does not.
The rhythm controller has no sensors, so when the body starts to tip it keeps playing the same
motion and falls.

**Why the next era.** Staying upright needs a brain that reads the body's sensors and
reacts: a neural network. Even the default network has thousands of numbers, too many for the
evolution trainers to search. Gradients can tune all of them at once, so Walk introduces them.

### Walk → Sprint: the copycat stumbles

**The failure.** On Walk the copy trainer learns to imitate a recorded expert walk. The copy
holds up while it stays on the expert's path. After one small mistake it stumbles, because it
only ever saw the expert's states and never saw itself off course, so it never learned how to
recover.

**Why the next era.** To recover from its own mistakes, the runner has to learn from its own
experience: try, fail and get rewarded. That is reinforcement learning, starting with
REINFORCE on Sprint.

### After Sprint: what surrounds the trainer

From Sprint on, the player has PPO, and the main path adds no new trainer. Each later level's
new idea is about what surrounds the trainer instead:

* **Climb Steps: curriculum.** Start with low steps and raise them as the runner succeeds.
* **Backflip: reward design and reference motion.** Spinning is easy to reward; landing on
  the feet is not. Research backflips usually come from copying a recorded reference motion
  (DeepMimic) or from human feedback.

By this point the wall the player hits is the task, not the trainer. Backflip makes the
point: for hard skills, the reward design matters more than the trainer. Sprint already hints
at it. The README's [reward hacks](../README.md#reward-hacks-found) (one-leg skipping, toe-drag
leap, limp, stutter step) all came from the reward, not from PPO.

## Trainer decisions

| Trainer | Decision | Why |
|---|---|---|
| Hill climbing | Added, as the Crawl tutorial | The simplest trial and error: change the numbers a little and keep the change if the score improves. It comes before the trainers that use a population. |
| Copy trainer (behavior cloning) | Added, on Walk | Teaches gradients, and SGD, RMSprop and Adam, on a supervised problem before reward enters. Its stumble is the failure that leads to RL. |
| REINFORCE | Added, as the first RL trainer on Sprint | Plain policy gradient. An update that is too big makes learning collapse, which is the failure PPO fixes. |
| TRPO | Stretch ([#15]) | Until it is built, the PPO unlock tells TRPO's idea as story text: limit how much the policy changes per update. |
| SAC | Optional, not on the main path ([#16]) | Its strength is learning from fewer simulation steps, which matters little with 2,048 parallel GPU simulations. |
| Brax ES and ARS | Kept off the main path | They scale evolution to neural networks, which undercuts the lesson that unlocks gradients. |
| CMA-ES | Never paired with a neural-network brain | Its covariance matrix grows with the square of the parameter count (below). |

### CMA-ES is never paired with a neural network

CMA-ES learns a covariance matrix over every number it tunes, so the matrix has one entry per
pair of parameters. That is fine for a rhythm controller and out of reach for a network:

| Brain | Parameters | Covariance entries | Size in float32 |
|---|---|---|---|
| Rhythm controller | ~30 | ~900 | ~4 KB |
| Default Sprint policy (28 inputs → 4 × 32 → 20 outputs) | 4,756 | 22.6 million | 90 MB |
| Largest policy in the editor (28 → 6 × 256 → 20) | 341,524 | 1.2 × 10¹¹ | 470 GB |

The policy has 20 outputs because Brax's PPO policy gives a mean and a spread for each of the
10 motors. The development GPU, an RTX 3070, has 8 GB.

### SGD, RMSprop and Adam are optimizers, not trainers

An optimizer takes a gradient and decides how to change the weights. SGD, RMSprop and Adam run
inside a gradient-based trainer (the copy trainer, REINFORCE, PPO). They are not trainers
themselves. The player meets them on Walk, where the copy trainer starts with SGD and unlocks
RMSprop, then Adam. Evolution trainers (hill climbing, GA, PSO, CMA-ES) use no gradients, so
they have no optimizer.

### Why this order

GA, PSO and CMA-ES, gradient methods and RL suit different problems, so trainers unlock in the
order the levels need them, not the order they were invented.

Older does not always mean worse. Random search with linear policies matched deep RL on MuJoCo
locomotion ([Mania et al., 2018][mania]), and evolution strategies taught a 3D humanoid to
walk ([Salimans et al., 2017][salimans]). So every unlock has to be confirmed by the benchmark
spike ([#9]), which runs each trainer on its level at the same GPU budget. Where an unlock
turns out weak, the lineup is revised in [#8].

## Considered but not in v1

* Get Up
* Hurdles
* Shove modifier

Grip climbing (hands on a wall or a ladder) is a separate stretch card on the board, after
Climb Steps.

## Open questions

These are decided on their own cards, not here.

| Question | Card |
|---|---|
| Jump: long jump, high jump, or both as two scores? | [#11] |
| Each level's score, pass bar, start pose, end conditions and lesson text | Design sheets in [#10], [#11], [#12], [#13], [#14] |
| Can a rhythm controller produce a crawl and a jump with this body? | [#9] |
| Is each unlock a step up a player would notice? Close pairs: GA vs PSO, RMSprop vs Adam | [#9], then [#8] |
| Does CMA-ES on a small or linear policy sprint well enough to undercut the gradient unlock? | [#9], then [#8] |
| The parameter cap for CMA-ES | [#9] |
| Backflip: staged rewards or a reference motion? | [#14] |
| Does TRPO earn a slot between REINFORCE and PPO? | [#15] |
| Where does SAC fit: a side option or a Backflip candidate? | [#16] |

## Unlock rules

On each level, a trainer unlocks the next one when a player's run with it scores at least 80%
of that trainer's reference score on that level:

> **unlock when run score ≥ 0.8 × reference(level, trainer)**

The reference is the best score the team can show the trainer reaching on that level when it
is well tuned, at the same budget a player's run gets. A run at 80% of it has used most of what
the trainer can do, so further gains need the next trainer. That is the wall the player hits
before each unlock.

This section fixes the rule and its terms ([#6]). The reference scores and thresholds are
measured in [#7].

### The score

The unlock reads the level's score, the same score its pass bar uses:

| Level | Score |
|---|---|
| Crawl | Set in the design sheet on [#10] |
| Jump | Set in the design sheet on [#11] |
| Walk | Set in the design sheet on [#12] |
| Sprint | Distance covered in 10 s, in meters |

Climb Steps ([#13]) and Backflip ([#14]) have a single trainer, so they have no unlock.

* The score is always measured in the simulation. It is never a trainer's own number: not the
  copy trainer's imitation loss, an evolution trainer's fitness or the RL reward. So every
  trainer on a level is judged on the same scale.
* Higher is better, and a runner that does nothing scores 0: distance from the start, height
  gained above standing. Otherwise 80% of the reference can be reached without learning
  anything. If Jump scored the torso's peak height instead of the height gained, a runner that
  stood still would already be most of the way there. Each level's design sheet has to define
  its score this way.

### Which result counts

* **One run's final evaluation.** The score is taken from the `eval.json` of a single training
  run: 10 deterministic episodes of the brain the run ends with (for an evolution trainer, the
  best candidate it found). On Sprint it is `distance_m.mean`, and an episode that ends in a
  fall counts the distance covered before the fall. So falls lower the score. `summary.json`'s
  `level_score_m` is a different number, the distance of the one replayed (median) episode,
  and the unlock does not use it.
* Not a checkpoint from partway through training, not the evals logged during training, and not
  a mean over several of the player's runs. Each run is checked on its own when it finishes,
  and a player can make as many runs as they like.
* The run uses the trainer and brain the level gives at that point, with any settings the block
  editor allows. A shorter run than the budget is allowed and is held to the same threshold.
* The unlock reads only the score. Sprint's gait checks are part of its pass bar, not of the
  unlock: a hacked gait that covers the distance has still pushed the trainer that far.
* Unlocks are per player and permanent.

### How a reference score is measured

| | Rule |
|---|---|
| Tuning | As much as the team wants, but only through settings a player can set in the block editor, within the config schema's ranges (network, trainer settings, reward weights). No code changes. |
| Budget | The same as a player's run ([below](#the-budget)). Every run in the tuning sweep gets that budget; the sweep's total cost does not count. |
| Seeds | Tune on any seeds, then rerun the chosen settings on 5 fresh seeds that the tuning never used. |
| Statistic | The mean of the 5 runs' scores, each from its own final `eval.json`: the same number a player's run is judged on. |

* **Only settings a player can reach**, so every reference is a score a player could get. If the
  team needs a setting the editor lacks, the editor gets it or the reference goes without.
* **Fresh seeds.** The best of many settings tried on the same seeds is partly seed luck, so its
  score overstates what those settings do. Rerunning them on new seeds removes that.
* **The mean, not the best seed.** A player's run is one seed. Measured from a mean, the 20%
  margin covers most of the spread between seeds; measured from the best seed, it does not. The
  milestone 1 Sprint runs ([README](../README.md#results-rtx-3070-wsl2); default settings, not
  tuned, 3 seeds) show it: 58.2, 67.9 and 47.0 m. Against the mean, the threshold is
  0.8 × 57.7 = 46.2 m, and all three runs clear it. Against the best seed it is
  0.8 × 67.9 = 54.3 m, and seed 2 misses with the same settings.
* The tuned settings are checked in as config files ([#7]) so every reference can be rerun.
* A reference belongs to one version of its level. A change to the level's environment, score,
  brain, editor ranges or budget means remeasuring the references on that level.

### The budget

A player's run and every reference run get the same budget: a maximum training length per
trainer, sized so that a run takes about 10 minutes on the reference GPU (the RTX 3070 in WSL2,
with the Warp backend). Sprint's default PPO config is this size today: 58,982,400 steps in
10.4-10.5 minutes, compile included.

* **Counted in steps, not minutes**, in each trainer's own unit: environment steps for
  REINFORCE and PPO, generations for the evolution trainers, and gradient steps for the copy
  trainer. The server enforces the step cap, not the clock. A faster or slower server GPU then
  finishes sooner or later but reaches the same score, so the references stay valid when the
  server's GPU changes.
* Settings that make each step slower, such as a bigger network, make a run take longer but
  not train further. The editor's limits bound how much longer.
* The caps are set per trainer from the timings in the spike ([#9]), which runs at the same
  10 minutes, and stored with the references ([below](#where-the-numbers-live)). Today's Sprint
  schema allows `num_timesteps` up to 500 million; config contract v2 lowers that to the cap.
* Without the same budget, 80% of the reference may be out of reach: a reference trained for
  an hour would hold a 10-minute run to a score it cannot get.

### Unlocking a trainer vs passing a level

| | Unlock threshold | Pass bar |
|---|---|---|
| What it opens | The next trainer on the same level | The next level |
| Measured against | The trainer in use: 0.8 × its reference | One absolute bar per level, the same for every trainer |
| Set in | This section and [#7] | The level's design sheet. Sprint: at least 20 m in 10 s without falling in 8 of 10 episodes, plus the gait checks in `sprinter/evaluate.py` |
| What it checks | The score only | The score, plus any other checks the sheet adds |

* Every finished run is checked against both. A run can unlock without passing, which is the
  usual case for a level's early trainers, or pass without unlocking.
* **Passing a level unlocks every trainer on it** that the player has not unlocked yet. The next
  level then always starts with the trainer the lineup lists (PSO on Jump, the copycat on
  Sprint).
* So passing early skips the remaining unlocks. To keep the usual path through every trainer, a
  level's pass bar should sit above the unlock thresholds of its earlier trainers, so a player
  whose scores climb toward the bar meets the next trainer first. Each design sheet checks its
  bar against the table below once [#7] has the numbers.
* **Once the last trainer on a level is unlocked**, nothing more unlocks there. The player keeps
  training toward the pass bar and their own best score. On Climb Steps and Backflip that is the
  whole level. When the last trainer carries over to the next level (PSO from Crawl to Jump), it
  has a separate reference there: references are per level and trainer.

### Unlock table

| Level | Trainer | Unlocks | Reference score | Threshold (0.8 × reference) |
|---|---|---|---|---|
| Crawl | Hill climbing | GA | TBD | TBD |
| Crawl | GA | PSO | TBD | TBD |
| Crawl | PSO | — (last on the level) | TBD | — |
| Jump | PSO | CMA-ES | TBD | TBD |
| Jump | CMA-ES | — (last on the level) | TBD | — |
| Walk | Copy trainer with SGD | RMSprop | TBD | TBD |
| Walk | Copy trainer with RMSprop | Adam | TBD | TBD |
| Walk | Copy trainer with Adam | — (last on the level) | TBD | — |
| Sprint | The copycat | REINFORCE | TBD | TBD |
| Sprint | REINFORCE | PPO | TBD | TBD |
| Sprint | PPO | — (last on the level) | TBD | — |
| Climb Steps | PPO | — (single trainer) | — | — |
| Backflip | PPO | — (single trainer) | — | — |

* The last trainer on each level unlocks nothing, but it gets a reference too, so [#8] can check
  that every unlock is a clear step up: the next trainer's reference well above the current
  one's.
* If TRPO ([#15]) is built, it gets a row between REINFORCE and PPO. SAC ([#16]) stays off the
  main path and gets no row unless it becomes an unlock.
* Lineup changes from [#8] update this table.

### Where the numbers live

`configs/unlocks.json` is the only store of the reference scores and thresholds. [#7] creates it
with the first measured references. The server loads it and sends it to the app, and the app's
unlock engine uses what the server sent; the app keeps no copy of its own. The table above
mirrors the file, and when they disagree, the file is right.

The file holds:

* `fraction`: 0.8, stored once, so changing the rule is one edit.
* One entry per level and trainer: the trainer it unlocks, the reference score, the threshold,
  the 5 seeds and their scores, the step cap it was measured at (the same cap the server
  enforces on player runs), the path to the tuned config, and the commit it was measured at.
* The threshold is stored, not only computed, so the server and the app can never round it
  differently. A schema in `sprinter/` validates the file, and a test checks that each
  threshold is `fraction` × reference, rounded to 2 decimals like the scores in `eval.json`.

[#4]: https://github.com/BrockBadeaux14/FirstSteps/issues/4
[#6]: https://github.com/BrockBadeaux14/FirstSteps/issues/6
[#7]: https://github.com/BrockBadeaux14/FirstSteps/issues/7
[#8]: https://github.com/BrockBadeaux14/FirstSteps/issues/8
[#9]: https://github.com/BrockBadeaux14/FirstSteps/issues/9
[#10]: https://github.com/BrockBadeaux14/FirstSteps/issues/10
[#11]: https://github.com/BrockBadeaux14/FirstSteps/issues/11
[#12]: https://github.com/BrockBadeaux14/FirstSteps/issues/12
[#13]: https://github.com/BrockBadeaux14/FirstSteps/issues/13
[#14]: https://github.com/BrockBadeaux14/FirstSteps/issues/14
[#15]: https://github.com/BrockBadeaux14/FirstSteps/issues/15
[#16]: https://github.com/BrockBadeaux14/FirstSteps/issues/16
[mania]: https://arxiv.org/abs/1803.07055
[salimans]: https://arxiv.org/abs/1703.03864
