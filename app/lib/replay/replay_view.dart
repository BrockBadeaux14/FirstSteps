// One replay pane: the track and the runner, with the time, distance and fall
// status on top.

import 'package:flutter/material.dart';

import 'replay.dart';
import 'replay_painter.dart';

String formatSeconds(double s) => '${(s + 1e-9).toStringAsFixed(2)} s';

/// Meters with one decimal. Tiny backward drift at the start shows as 0.0
/// rather than -0.0.
String formatMeters(double m) =>
    '${(m.abs() < 0.05 ? 0.0 : m).toStringAsFixed(1)} m';

/// Training steps in millions: 58982400 -> "59.0M".
String formatSteps(int steps) =>
    '${(steps / 1e6).toStringAsFixed(1)}M steps';

class ReplayView extends StatelessWidget {
  const ReplayView({
    super.key,
    required this.replay,
    required this.time,
    required this.title,
    this.subtitle,
  });

  final Replay replay;

  /// Playback time in seconds. Past the end of the replay, the last frame
  /// holds.
  final double time;
  final String title;
  final String? subtitle;

  @override
  Widget build(BuildContext context) {
    final frame = replay.frameIndexAt(time);
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Stack(
        children: [
          Positioned.fill(
            child: CustomPaint(
              painter: TrackPainter(replay, frame),
              foregroundPainter: RunnerPainter(replay, frame),
            ),
          ),
          Positioned(
            left: 12,
            top: 10,
            child: ReplayHud(replay: replay, frame: frame),
          ),
          Positioned(
            right: 12,
            top: 10,
            child: _PaneTitle(title: title, subtitle: subtitle),
          ),
        ],
      ),
    );
  }
}

/// Time, distance, and whether the runner fell.
class ReplayHud extends StatelessWidget {
  const ReplayHud({super.key, required this.replay, required this.frame});

  final Replay replay;
  final int frame;

  static const _fellColor = Color(0xFFE5484D);
  static const _noFallColor = Color(0xFF2F9E66);

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    // Tabular figures keep the numbers from jittering as they change.
    const numbers = TextStyle(
      color: Colors.white,
      fontFeatures: [FontFeature.tabularFigures()],
    );
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          formatSeconds(replay.frames[frame].t),
          style: text.titleMedium
              ?.merge(numbers)
              .copyWith(fontWeight: FontWeight.w600),
        ),
        Text(
          formatMeters(replay.distanceAt(frame)),
          style: text.bodyMedium?.merge(numbers),
        ),
        if (replay.hasFallenAt(frame))
          _Badge(
            label: 'Fell at ${formatSeconds(replay.meta.endS)}',
            color: _fellColor,
          )
        else if (replay.hasEndedAt(frame))
          const _Badge(label: 'No fall', color: _noFallColor),
      ],
    );
  }
}

class _Badge extends StatelessWidget {
  const _Badge({required this.label, required this.color});

  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(top: 6),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        label,
        style: Theme.of(context).textTheme.labelMedium?.copyWith(
              color: Colors.white,
              fontWeight: FontWeight.w600,
            ),
      ),
    );
  }
}

class _PaneTitle extends StatelessWidget {
  const _PaneTitle({required this.title, this.subtitle});

  final String title;
  final String? subtitle;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          title,
          style: text.titleSmall?.copyWith(
            color: Colors.white,
            fontWeight: FontWeight.w700,
          ),
        ),
        if (subtitle != null)
          Text(
            subtitle!,
            style: text.bodySmall?.copyWith(color: Colors.white70),
          ),
      ],
    );
  }
}
