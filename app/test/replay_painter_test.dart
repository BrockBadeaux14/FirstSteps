import 'dart:typed_data';
import 'dart:ui';

import 'package:first_steps/replay/replay.dart';
import 'package:first_steps/replay/replay_painter.dart';
import 'package:flutter_test/flutter_test.dart';

import 'fixtures.dart';

const size = Size(400, 300);

/// Paint stores numbers and colors as 32-bit floats.
double f32(double v) => (Float32List(1)..[0] = v)[0];

bool sameColor(Color a, Color b) => a.toARGB32() == b.toARGB32();

void main() {
  group('camera', () {
    test('centers the focus and keeps the floor at a fixed height', () {
      final camera = ReplayCamera(size: size, focusX: 12.5);
      expect(camera.toScreen(const Offset(12.5, 0)),
          Offset(200, camera.floorY));
      expect(camera.floorY,
          closeTo(size.height - ReplayCamera.belowFloorM * camera.scale, 1e-9));
      // One meter up and forward.
      expect(camera.toScreen(const Offset(13.5, 1)),
          Offset(200 + camera.scale, camera.floorY - camera.scale));
    });

    test('fits the body height in the view', () {
      for (final s in const [Size(390, 320), Size(420, 250), Size(150, 600)]) {
        final camera = ReplayCamera(size: s, focusX: 0);
        final head = camera.toScreen(const Offset(0, 1.75));
        expect(head.dy, greaterThan(0), reason: '$s');
        expect(camera.visibleX.$2 - camera.visibleX.$1,
            greaterThanOrEqualTo(ReplayCamera.minWidthM - 1e-9),
            reason: '$s');
      }
    });

    test('follows the torso', () {
      final frame = sampleAfter.frameIndexAt(5.0);
      final camera = ReplayCamera.follow(sampleAfter, frame, size);
      expect(camera.focusX, sampleAfter.frames[frame].pose(0).x);
      expect(camera.focusX, greaterThan(20));
    });
  });

  group('runner painter', () {
    test('draws every body in draw order, in its color', () {
      final replay = sampleAfter;
      const frame = 120;
      final camera = ReplayCamera.follow(replay, frame, size);
      var pattern = paints;
      for (final i in replay.drawOrder) {
        final body = replay.skeleton[i];
        final pose = replay.frames[frame].pose(i);
        final r = body.geom.radius * camera.scale;
        switch (body.geom) {
          case CapsuleGeom(:final from, :final to):
            final a = camera.toScreen(pose.toWorld(from));
            final b = camera.toScreen(pose.toWorld(to));
            pattern = pattern
              ..line(p1: a, p2: b, color: SceneColors.outline)
              ..line(p1: a, p2: b, color: body.color, strokeWidth: f32(2 * r));
          case SphereGeom(:final center):
            final c = camera.toScreen(pose.toWorld(center));
            pattern = pattern
              ..circle(
                  x: c.dx,
                  y: c.dy,
                  radius: r + RunnerPainter.outlineWidth,
                  color: SceneColors.outline)
              ..circle(x: c.dx, y: c.dy, radius: r, color: body.color);
        }
      }
      expect(
        (Canvas canvas) => RunnerPainter(replay, frame).paint(canvas, size),
        pattern,
      );
    });

    test('draws the left limbs first and the right arm last', () {
      final colors = <Color>[];
      expect(
        (Canvas canvas) =>
            RunnerPainter(sampleAfter, 0).paint(canvas, size),
        paints
          ..everything((method, args) {
            if (method == #drawLine) colors.add((args[2] as Paint).color);
            if (method == #drawCircle) colors.add((args[2] as Paint).color);
            return true;
          }),
      );
      final fills =
          colors.where((c) => !sameColor(c, SceneColors.outline)).toList();
      expect(fills.first, isSameColorAs(const Color(0xFF4285F5)));
      expect(fills.last, isSameColorAs(const Color(0xFFF57829)));
    });
  });

  group('track painter', () {
    bool isLineAt(List<dynamic> args, double x, Color color) {
      final p1 = args[0] as Offset;
      final paint = args[2] as Paint;
      return (p1.dx - x).abs() < 1e-6 && sameColor(paint.color, color);
    }

    test('draws a line across the track at 10 m with its label', () {
      // The frame where the runner is closest to the 10 m line.
      final replay = sampleAfter;
      var frame = 0;
      while (replay.frames[frame].pose(0).x < 10) {
        frame++;
      }
      final camera = ReplayCamera.follow(replay, frame, size);
      final x10 = camera.toScreen(const Offset(10, 0)).dx;
      expect(
        (Canvas canvas) => TrackPainter(replay, frame).paint(canvas, size),
        paints
          ..something((method, args) =>
              method == #drawLine &&
              isLineAt(args, x10, SceneColors.markerLine) &&
              (args[1] as Offset).dy == size.height)
          ..paragraph(),
      );
    });

    test('draws a start line at 0 m', () {
      final camera = ReplayCamera.follow(sampleAfter, 0, size);
      final x0 = camera.toScreen(Offset.zero).dx;
      expect(
        (Canvas canvas) => TrackPainter(sampleAfter, 0).paint(canvas, size),
        paints
          ..something((method, args) =>
              method == #drawLine && isLineAt(args, x0, SceneColors.markerLine)),
      );
    });

    test('draws a short tick at every other meter', () {
      final camera = ReplayCamera.follow(sampleAfter, 0, size);
      var markers = 0;
      final ticks = <double>[];
      expect(
        (Canvas canvas) => TrackPainter(sampleAfter, 0).paint(canvas, size),
        paints
          ..everything((method, args) {
            if (method != #drawLine) return true;
            final color = (args[2] as Paint).color;
            if (sameColor(color, SceneColors.markerLine)) markers++;
            if (sameColor(color, SceneColors.meterTick)) {
              ticks.add((args[0] as Offset).dx);
            }
            return true;
          }),
      );
      expect(markers, 1);
      final x1 = camera.toScreen(const Offset(1, 0)).dx;
      expect(ticks.any((x) => (x - x1).abs() < 1e-6), isTrue);
      final gaps = [for (var i = 1; i < ticks.length; i++) ticks[i] - ticks[i - 1]];
      for (final gap in gaps) {
        // One meter apart, or two across the start line.
        expect(gap, anyOf(closeTo(camera.scale, 1e-6), closeTo(2 * camera.scale, 1e-6)));
      }
    });
  });
}
