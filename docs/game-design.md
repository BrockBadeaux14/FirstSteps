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

_To be written in [#6] (unlock at 80% of a tuned reference score). The reference scores and
thresholds come from [#7]._

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
