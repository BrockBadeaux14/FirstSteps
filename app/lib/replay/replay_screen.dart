// The replay player: the before and after replays side by side on one clock,
// with play/pause, a scrub slider and a speed menu.

import 'dart:math' as math;

import 'package:flutter/material.dart';

import 'playback.dart';
import 'replay.dart';
import 'replay_view.dart';

class ReplayScreen extends StatefulWidget {
  const ReplayScreen({
    super.key,
    required this.before,
    required this.after,
    this.autoplay = true,
  });

  /// The early checkpoint (nearest 10% of training).
  final Replay before;

  /// The final policy.
  final Replay after;
  final bool autoplay;

  @override
  State<ReplayScreen> createState() => _ReplayScreenState();
}

class _ReplayScreenState extends State<ReplayScreen>
    with SingleTickerProviderStateMixin {
  late final PlaybackController _playback = PlaybackController(
    vsync: this,
    duration: math.max(widget.before.duration, widget.after.duration),
  );

  @override
  void initState() {
    super.initState();
    if (widget.autoplay) _playback.play();
  }

  @override
  void dispose() {
    _playback.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final level = widget.after.meta.level;
    final frameCount =
        math.max(widget.before.frames.length, widget.after.frames.length);
    return Scaffold(
      appBar: AppBar(
        title: Text(level.isEmpty
            ? 'Replay'
            : '${level[0].toUpperCase()}${level.substring(1)} replay'),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(8, 0, 8, 4),
          child: Column(
            children: [
              Expanded(
                child: ListenableBuilder(
                  listenable: _playback,
                  builder: (context, _) => BeforeAfterView(
                    before: widget.before,
                    after: widget.after,
                    time: _playback.time,
                  ),
                ),
              ),
              PlaybackControls(controller: _playback, frameCount: frameCount),
            ],
          ),
        ),
      ),
    );
  }
}

/// The two replays side by side when the space is wide, stacked when it is
/// tall: a narrow pane would show too little of the track.
class BeforeAfterView extends StatelessWidget {
  const BeforeAfterView({
    super.key,
    required this.before,
    required this.after,
    required this.time,
  });

  final Replay before;
  final Replay after;
  final double time;

  @override
  Widget build(BuildContext context) {
    final finalStep = after.meta.checkpointStep;
    final beforeStep = before.meta.checkpointStep;
    final share = finalStep > 0 ? (100 * beforeStep / finalStep).round() : null;
    return LayoutBuilder(
      builder: (context, constraints) => Flex(
        direction: constraints.maxWidth > constraints.maxHeight
            ? Axis.horizontal
            : Axis.vertical,
        spacing: 8,
        children: [
          Expanded(
            child: ReplayView(
              key: const ValueKey('before'),
              replay: before,
              time: time,
              title: share == null ? 'Before' : 'Before · $share%',
              subtitle: formatSteps(beforeStep),
            ),
          ),
          Expanded(
            child: ReplayView(
              key: const ValueKey('after'),
              replay: after,
              time: time,
              title: 'After',
              subtitle: formatSteps(finalStep),
            ),
          ),
        ],
      ),
    );
  }
}

String formatSpeed(double speed) => speed == speed.roundToDouble()
    ? '${speed.toInt()}×'
    : '$speed×';

class PlaybackControls extends StatelessWidget {
  const PlaybackControls({
    super.key,
    required this.controller,
    required this.frameCount,
  });

  final PlaybackController controller;

  /// Frames in the longest replay. The slider snaps to them.
  final int frameCount;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: controller,
      builder: (context, _) {
        final playing = controller.isPlaying;
        return Row(
          children: [
            IconButton.filledTonal(
              tooltip: playing ? 'Pause' : 'Play',
              onPressed: controller.togglePlay,
              icon: Icon(
                playing
                    ? Icons.pause
                    : controller.atEnd
                        ? Icons.replay
                        : Icons.play_arrow,
              ),
            ),
            Expanded(
              child: Slider(
                value: controller.time,
                max: controller.duration > 0 ? controller.duration : 1,
                divisions: frameCount > 1 ? frameCount - 1 : null,
                semanticFormatterCallback: formatSeconds,
                onChangeStart: (_) => controller.beginScrub(),
                onChanged: controller.seek,
                onChangeEnd: (_) => controller.endScrub(),
              ),
            ),
            PopupMenuButton<double>(
              tooltip: 'Playback speed',
              initialValue: controller.speed,
              onSelected: (speed) => controller.speed = speed,
              itemBuilder: (context) => [
                for (final speed in PlaybackController.speeds)
                  PopupMenuItem(value: speed, child: Text(formatSpeed(speed))),
              ],
              child: Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                child: Text(
                  formatSpeed(controller.speed),
                  style: Theme.of(context).textTheme.labelLarge,
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}
