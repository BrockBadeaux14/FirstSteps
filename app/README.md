# FirstSteps app

The Flutter app. For now it is the replay player: it plays the `replay.json` files that
`sprinter.replay` writes (format `sprinter-replay/1`, see
[replay.json format](../README.md#replayjson-format)) and shows the runner before and after
training side by side. It needs no server and no physics: the sample replays are bundled.

```bash
flutter run
```

## What it shows

* **Playback** at 40 frames per second, one frame per control step. Play/pause (play at the
  end restarts), a slider that scrubs frame by frame, and speeds of 0.25×, 0.5×, 1× and 2×.
* **The runner.** Each body's capsule or sphere in its `color`, in `meta.draw_order` (left limbs
  behind the torso, right limbs in front). A point (u, v) in a body's frame is drawn at
  `(x + u cos a - v sin a, z + u sin a + v cos a)`. A thin dark outline keeps overlapping
  limbs apart.
* **The track.** The camera follows the torso and keeps the floor still. A line across the
  track every `meta.marker_every_m` (10 m) with its distance, and a short tick every meter.
* **The HUD.** Time, and distance measured like `eval.json` (torso x minus its start). The
  distance stops at the fall, so the last frame shows `meta.distance_m`. A red "Fell at" badge
  appears at `meta.end_s` when `fell` is true; a green "No fall" badge at the end otherwise.
* **Before and after** on one clock. Before is the checkpoint nearest 10% of training, after is
  the final policy. The shorter replay holds its last frame. The panes sit side by side in
  landscape and stack in portrait, where a narrow pane would show too little track.

## Code

| File | What it is |
|---|---|
| `lib/replay/replay.dart` | model and parsing; frame lookup, distance and falls |
| `lib/replay/replay_painter.dart` | follow camera, track painter, runner painter |
| `lib/replay/playback.dart` | the playback clock (a `Ticker`) shared by both panes |
| `lib/replay/replay_view.dart` | one pane: painters, HUD and title |
| `lib/replay/replay_screen.dart` | the before/after screen and the controls |
| `lib/replay/sample_replays.dart` | the bundled replays and the asset loader |

Parsing rejects files whose `format` is not `sprinter-replay/1`, frames whose pose count doesn't
match the skeleton, and `draw_order` names that aren't bodies.

## Sample replays

`assets/replays/final-seed1/` holds `replay.json` (final policy: 68.0 m in 10 s, no fall) and
`before/replay.json` (5.9M steps: falls at 0.575 s) from the WSL2 run `runs/final-seed1`, made
with:

```bash
uv run python -m sprinter.replay --run runs/final-seed1
```

The tests read the same files as fixtures and check this run's numbers, so new sample files
need the tests and goldens updated too.

## Tests

```bash
flutter test
```

* `test/replay_test.dart`: parsing, the files it rejects, frame lookup, distance and falls, and
  the pose transform (including that every child body sits at its skeleton offset from its
  parent, in every frame).
* `test/replay_painter_test.dart`: the camera, draw order and colors, the track lines and ticks.
* `test/playback_test.dart`: the clock: 40 frames a second, pause, speed, the end, scrubbing.
* `test/replay_screen_test.dart`: widget tests: the app loads the bundled replays, layout in
  portrait and landscape, play, pause, the slider, the speed menu, the fall and the end.
* `test/golden_test.dart`: golden images in `test/goldens/`. After an intended change to the
  drawing, regenerate them with `flutter test --update-goldens test/golden_test.dart`.

The goldens were made on Windows. Text and anti-aliasing can differ by a few pixels on macOS
or Linux, so `test/flutter_test_config.dart` lets a golden pass when under 0.5% of its pixels
differ. Tests draw text with Flutter's test font, which shows letters as boxes.
