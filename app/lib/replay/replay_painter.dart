// Drawing a replay frame: a camera that follows the runner, the track, and the
// runner's bodies as capsules and spheres.

import 'dart:math' as math;

import 'package:flutter/rendering.dart';

import 'replay.dart';

/// Colors of the scene. It stays dark in both app themes so the runner's light
/// torso and its blue (left) and orange (right) limbs stand out.
abstract final class SceneColors {
  static const skyTop = Color(0xFF1B2129);
  static const skyBottom = Color(0xFF2A333E);
  static const track = Color(0xFF3B3633);
  static const trackEdge = Color(0xFF8A817A);
  static const meterTick = Color(0x55FFFFFF);
  static const markerLine = Color(0xDDFFFFFF);
  static const markerLabel = Color(0xEEFFFFFF);
  static const outline = Color(0x99000000);
}

/// Maps world meters (x forward, z up) to screen pixels. The camera follows
/// the runner horizontally; the floor stays at a fixed height on screen.
class ReplayCamera {
  ReplayCamera({required this.size, required this.focusX, this.floorZ = 0.0})
      : scale = math.min(
          size.height / (belowFloorM + aboveFloorM),
          size.width / minWidthM,
        );

  /// Room under the floor for the track.
  static const belowFloorM = 0.45;

  /// Room above the floor for the body (1.75 m tall, arms can go higher).
  static const aboveFloorM = 2.15;

  /// The narrowest stretch of track the view shows.
  static const minWidthM = 2.0;

  final Size size;

  /// World x shown at the center of the view.
  final double focusX;
  final double floorZ;

  /// Pixels per meter.
  final double scale;

  double get floorY => size.height - belowFloorM * scale;

  Offset toScreen(Offset world) => Offset(
        size.width / 2 + (world.dx - focusX) * scale,
        floorY - (world.dy - floorZ) * scale,
      );

  /// World x at the left and right edges of the view.
  (double, double) get visibleX => (
        focusX - size.width / 2 / scale,
        focusX + size.width / 2 / scale,
      );

  /// A camera centered on the runner's torso at [frame].
  factory ReplayCamera.follow(Replay replay, int frame, Size size) =>
      ReplayCamera(
        size: size,
        focusX: replay.frames[frame].pose(replay.rootIndex).x,
        floorZ: replay.meta.floorZ,
      );
}

/// The background: sky, the track, a tick every meter, and a line across the
/// track every `meta.markerEveryM` meters with its distance.
class TrackPainter extends CustomPainter {
  TrackPainter(this.replay, this.frame);

  final Replay replay;
  final int frame;

  @override
  void paint(Canvas canvas, Size size) {
    final camera = ReplayCamera.follow(replay, frame, size);
    final floorY = camera.floorY;

    canvas.drawRect(
      Offset.zero & size,
      Paint()
        ..shader = const LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [SceneColors.skyTop, SceneColors.skyBottom],
        ).createShader(Offset.zero & size),
    );
    canvas.drawRect(
      Rect.fromLTRB(0, floorY, size.width, size.height),
      Paint()..color = SceneColors.track,
    );
    canvas.drawLine(
      Offset(0, floorY),
      Offset(size.width, floorY),
      Paint()
        ..color = SceneColors.trackEdge
        ..strokeWidth = 2,
    );

    final every = replay.meta.markerEveryM;
    final tickLength = 0.08 * camera.scale;
    final tickPaint = Paint()
      ..color = SceneColors.meterTick
      ..strokeWidth = 1;
    final markerPaint = Paint()
      ..color = SceneColors.markerLine
      ..strokeWidth = 2;
    final (left, right) = camera.visibleX;
    // One meter of margin so a label that starts just off screen still shows.
    for (var m = left.floor() - 1; m <= right.ceil(); m++) {
      final x = camera.toScreen(Offset(m.toDouble(), replay.meta.floorZ)).dx;
      if (m >= 0 && every > 0 && m % every == 0) {
        canvas.drawLine(
            Offset(x, floorY), Offset(x, size.height), markerPaint);
        _label(canvas, '$m m', Offset(x + 6, floorY + 0.12 * camera.scale));
      } else {
        canvas.drawLine(
            Offset(x, floorY), Offset(x, floorY + tickLength), tickPaint);
      }
    }
  }

  void _label(Canvas canvas, String text, Offset at) {
    final painter = TextPainter(
      text: TextSpan(
        text: text,
        style: const TextStyle(
          color: SceneColors.markerLabel,
          fontSize: 12,
          fontWeight: FontWeight.w600,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    painter.paint(canvas, at);
    painter.dispose();
  }

  @override
  bool shouldRepaint(TrackPainter oldDelegate) =>
      oldDelegate.replay != replay || oldDelegate.frame != frame;
}

/// The runner: each body's capsule or sphere in its color, in
/// `meta.drawOrder` (back to front). Each shape gets a thin dark outline so
/// overlapping limbs of the same color stay apart.
class RunnerPainter extends CustomPainter {
  RunnerPainter(this.replay, this.frame);

  static const outlineWidth = 1.5;

  final Replay replay;
  final int frame;

  @override
  void paint(Canvas canvas, Size size) {
    final camera = ReplayCamera.follow(replay, frame, size);
    final poses = replay.frames[frame];

    for (final i in replay.drawOrder) {
      final body = replay.skeleton[i];
      final pose = poses.pose(i);
      final r = body.geom.radius * camera.scale;
      final outline = Paint()
        ..color = SceneColors.outline
        ..strokeCap = StrokeCap.round;
      final fill = Paint()
        ..color = body.color
        ..strokeCap = StrokeCap.round;
      switch (body.geom) {
        // A capsule is a round-capped stroke as wide as its diameter.
        case CapsuleGeom(:final from, :final to):
          final a = camera.toScreen(pose.toWorld(from));
          final b = camera.toScreen(pose.toWorld(to));
          canvas.drawLine(a, b, outline..strokeWidth = 2 * (r + outlineWidth));
          canvas.drawLine(a, b, fill..strokeWidth = 2 * r);
        case SphereGeom(:final center):
          final c = camera.toScreen(pose.toWorld(center));
          canvas.drawCircle(c, r + outlineWidth, outline);
          canvas.drawCircle(c, r, fill);
      }
    }
  }

  @override
  bool shouldRepaint(RunnerPainter oldDelegate) =>
      oldDelegate.replay != replay || oldDelegate.frame != frame;
}
