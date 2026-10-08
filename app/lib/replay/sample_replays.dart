// The replays bundled with the app, so the player works with no server.
// They come from the WSL2 run final-seed1 (see app/README.md).

import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'replay.dart';
import 'replay_screen.dart';

const sampleAfterAsset = 'assets/replays/final-seed1/replay.json';
const sampleBeforeAsset = 'assets/replays/final-seed1/before/replay.json';

Future<Replay> loadReplayAsset(AssetBundle bundle, String key) async {
  // load() rather than loadString(): loadString decodes large files on a
  // background isolate, which this small a file doesn't need.
  final data = await bundle.load(key);
  return Replay.parse(utf8.decode(
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes)));
}

Future<(Replay, Replay)> loadSampleReplays(AssetBundle bundle) async => (
      await loadReplayAsset(bundle, sampleBeforeAsset),
      await loadReplayAsset(bundle, sampleAfterAsset),
    );

/// Loads the bundled replays, then shows the [ReplayScreen].
class SampleReplayScreen extends StatefulWidget {
  const SampleReplayScreen({super.key, this.bundle});

  /// Defaults to [rootBundle].
  final AssetBundle? bundle;

  @override
  State<SampleReplayScreen> createState() => _SampleReplayScreenState();
}

class _SampleReplayScreenState extends State<SampleReplayScreen> {
  late final Future<(Replay, Replay)> _replays =
      loadSampleReplays(widget.bundle ?? rootBundle);

  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      future: _replays,
      builder: (context, snapshot) {
        if (snapshot.data case (final before, final after)) {
          return ReplayScreen(before: before, after: after);
        }
        return Scaffold(
          appBar: AppBar(title: const Text('Replay')),
          body: Center(
            child: snapshot.hasError
                ? Padding(
                    padding: const EdgeInsets.all(24),
                    child: Text('Could not load the replays:\n'
                        '${snapshot.error}'),
                  )
                : const CircularProgressIndicator(),
          ),
        );
      },
    );
  }
}
