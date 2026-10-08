// The bundled sample replays, read from disk as test fixtures.
// `flutter test` runs from the package root, so asset keys are file paths.

import 'dart:convert';
import 'dart:io';

import 'package:first_steps/replay/replay.dart';
import 'package:first_steps/replay/sample_replays.dart';

/// Final policy of final-seed1: 68.0 m in 10 s, no fall.
final Replay sampleAfter = Replay.parse(File(sampleAfterAsset).readAsStringSync());

/// Checkpoint nearest 10% of training: falls at 0.575 s.
final Replay sampleBefore =
    Replay.parse(File(sampleBeforeAsset).readAsStringSync());

/// A fresh, editable copy of a sample replay's JSON.
Map<String, Object?> sampleJson(String asset) =>
    (jsonDecode(File(asset).readAsStringSync()) as Map).cast<String, Object?>();
