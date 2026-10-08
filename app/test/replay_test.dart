import 'dart:convert';
import 'dart:math' as math;
import 'dart:ui';

import 'package:first_steps/replay/replay.dart';
import 'package:first_steps/replay/sample_replays.dart';
import 'package:flutter_test/flutter_test.dart';

import 'fixtures.dart';

Matcher closeToOffset(Offset expected, double tolerance) => predicate<Offset>(
      (o) => (o - expected).distance <= tolerance,
      'within $tolerance of $expected',
    );

void main() {
  group('parsing', () {
    test('reads the meta of the final replay', () {
      final meta = sampleAfter.meta;
      expect(meta.level, 'sprint');
      expect(meta.dt, 0.025);
      expect(meta.numFrames, 401);
      expect(meta.distanceM, 68.0);
      expect(meta.fell, isFalse);
      expect(meta.endS, 10.0);
      expect(meta.floorZ, 0.0);
      expect(meta.markerEveryM, 10);
      expect(meta.checkpointStep, 58982400);
      expect(meta.seed, 1);
    });

    test('reads the meta of the before replay', () {
      final meta = sampleBefore.meta;
      expect(meta.fell, isTrue);
      expect(meta.endS, 0.575);
      expect(meta.checkpointStep, 5898240);
      // Frames run 1 s past the fall: 23 steps + 40 + the reset frame.
      expect(sampleBefore.frames, hasLength(64));
      expect(sampleBefore.endFrame, 23);
    });

    test('reads the skeleton', () {
      final skeleton = sampleAfter.skeleton;
      expect(skeleton, hasLength(14));
      expect(skeleton.first.name, 'torso');
      expect(skeleton.first.parent, isNull);
      expect(sampleAfter.rootIndex, 0);

      final head = skeleton.firstWhere((b) => b.name == 'head');
      expect(head.parent, 'torso');
      expect(head.offset, const Offset(0, 0.71));
      expect(head.geom, isA<SphereGeom>());
      expect(head.geom.radius, 0.11);

      final shin = skeleton.firstWhere((b) => b.name == 'left_shin');
      expect(shin.side, BodySide.left);
      expect(shin.depth, 0.09);
      expect(shin.color, const Color(0xFF4285F5));
      final geom = shin.geom as CapsuleGeom;
      expect(geom.from, const Offset(0, -0.38));
      expect(geom.to, Offset.zero);
      expect(geom.radius, 0.05);

      expect(
        skeleton.firstWhere((b) => b.name == 'right_thigh').color,
        const Color(0xFFF57829),
      );
    });

    test('maps draw_order to skeleton indices, back to front', () {
      final names = [
        for (final i in sampleAfter.drawOrder) sampleAfter.skeleton[i].name,
      ];
      expect(names, sampleAfter.meta.drawOrder);
      expect(names.first, 'left_upper_arm');
      expect(names.indexOf('torso'), greaterThan(names.indexOf('left_foot')));
      expect(names.indexOf('right_thigh'), greaterThan(names.indexOf('torso')));
    });

    test('reads every frame with a pose per body', () {
      expect(sampleAfter.frames, hasLength(401));
      for (final frame in sampleAfter.frames) {
        expect(frame.bodyCount, 14);
      }
      expect(sampleAfter.frames[1].t, 0.025);
      final torso = sampleAfter.frames.first.pose(0);
      expect([torso.x, torso.z, torso.angle], [0.0, 0.929, 0.0]);
      expect(sampleAfter.duration, closeTo(10.0, 1e-9));
    });

    void expectRejected(void Function(Map<String, Object?> json) edit,
        String message) {
      final json = sampleJson(sampleBeforeAsset);
      edit(json);
      expect(
        () => Replay.parse(jsonEncode(json)),
        throwsA(isA<FormatException>().having(
            (e) => e.message, 'message', contains(message))),
      );
    }

    Map<String, Object?> meta(Map<String, Object?> json) =>
        (json['meta'] as Map).cast<String, Object?>();
    List<Object?> frames(Map<String, Object?> json) => json['frames'] as List;

    test('rejects another format version', () {
      expectRejected(
        (json) => meta(json)['format'] = 'sprinter-replay/2',
        'Unsupported replay format',
      );
    });

    test('rejects a frame whose pose count does not match the skeleton', () {
      expectRejected(
        (json) => ((frames(json)[3] as Map)['poses'] as List).removeLast(),
        'frames[3] has 13 poses for 14 bodies',
      );
    });

    test('rejects a draw_order name that is not a body', () {
      expectRejected(
        (json) => (meta(json)['draw_order'] as List)[0] = 'tail',
        'unknown body "tail"',
      );
    });

    test('rejects a draw_order that leaves a body out', () {
      expectRejected(
        (json) => (meta(json)['draw_order'] as List).removeLast(),
        'every body exactly once',
      );
    });

    test('rejects a frame count that does not match meta', () {
      expectRejected(
        (json) => frames(json).removeLast(),
        'meta.num_frames is 64 but the file has 63 frames',
      );
    });

    test('rejects an unknown geom type', () {
      expectRejected(
        (json) => (((json['skeleton'] as List)[1] as Map)['geom']
            as Map)['type'] = 'box',
        'unknown type "box"',
      );
    });

    test('rejects a bad color', () {
      expectRejected(
        (json) => ((json['skeleton'] as List)[0] as Map)['color'] = 'beige',
        'must be "#rrggbb"',
      );
    });

    test('rejects text that is not JSON', () {
      expect(() => Replay.parse('{"meta": '), throwsFormatException);
    });
  });

  group('frame lookup', () {
    test('one frame per control step, 40 per second', () {
      expect(sampleAfter.frameIndexAt(0), 0);
      expect(sampleAfter.frameIndexAt(0.0249), 0);
      expect(sampleAfter.frameIndexAt(0.025), 1);
      expect(sampleAfter.frameIndexAt(1.0), 40);
      expect(sampleAfter.frameIndexAt(5.0), 200);
      // Time summed from many small steps lands on the right frame.
      var t = 0.0;
      for (var i = 0; i < 120; i++) {
        t += 0.025;
      }
      expect(sampleAfter.frameIndexAt(t), 120);
    });

    test('clamps to the first and last frames', () {
      expect(sampleAfter.frameIndexAt(-1), 0);
      expect(sampleAfter.frameIndexAt(10.0), 400);
      expect(sampleAfter.frameIndexAt(99), 400);
      expect(sampleBefore.frameIndexAt(5.0), 63);
    });
  });

  group('distance and falls', () {
    test('distance matches meta at the last frame', () {
      for (final replay in [sampleAfter, sampleBefore]) {
        expect(
          replay.distanceAt(replay.frames.length - 1),
          closeTo(replay.meta.distanceM, 0.01),
        );
      }
      expect(sampleAfter.distanceAt(0), 0);
    });

    test('distance stops at the fall', () {
      final end = sampleBefore.endFrame;
      expect(sampleBefore.distanceAt(end + 10), sampleBefore.distanceAt(end));
      expect(sampleBefore.distanceAt(63), sampleBefore.distanceAt(end));
    });

    test('the before runner has fallen from meta.end_s on', () {
      expect(sampleBefore.hasFallenAt(sampleBefore.endFrame - 1), isFalse);
      expect(sampleBefore.hasFallenAt(sampleBefore.endFrame), isTrue);
      expect(sampleBefore.hasFallenAt(63), isTrue);
    });

    test('the final runner never falls', () {
      expect(sampleAfter.hasFallenAt(400), isFalse);
      expect(sampleAfter.hasEndedAt(399), isFalse);
      expect(sampleAfter.hasEndedAt(400), isTrue);
    });
  });

  group('pose transform', () {
    test('translates the body-frame point', () {
      expect(const Pose(1, 2, 0).toWorld(const Offset(0.5, -0.25)),
          const Offset(1.5, 1.75));
    });

    test('rotates counter-clockwise with +x right and +z up', () {
      const quarter = Pose(0, 0, math.pi / 2);
      expect(quarter.toWorld(const Offset(1, 0)),
          closeToOffset(const Offset(0, 1), 1e-12));
      expect(quarter.toWorld(const Offset(0, 1)),
          closeToOffset(const Offset(-1, 0), 1e-12));
    });

    test('rotates and translates: (x + u cos a - v sin a, z + u sin a + v cos a)',
        () {
      const a = 0.3;
      const pose = Pose(2, 1, a);
      const u = 0.4, v = -0.2;
      expect(
        pose.toWorld(const Offset(u, v)),
        closeToOffset(
          Offset(2 + u * math.cos(a) - v * math.sin(a),
              1 + u * math.sin(a) + v * math.cos(a)),
          1e-12,
        ),
      );
    });

    test('puts each child body at its offset in the parent frame', () {
      // Checks the transform against the physics: every child's origin is
      // its skeleton offset placed by the parent's pose, in every frame.
      // Poses are rounded to 3 decimals, hence the tolerance.
      final index = {
        for (final (i, b) in sampleAfter.skeleton.indexed) b.name: i,
      };
      for (final frame in sampleAfter.frames) {
        for (final (i, body) in sampleAfter.skeleton.indexed) {
          if (body.parent == null) continue;
          final parent = frame.pose(index[body.parent]!);
          final child = frame.pose(i);
          expect(
            parent.toWorld(body.offset),
            closeToOffset(Offset(child.x, child.z), 2e-3),
            reason: '${body.name} at t = ${frame.t}',
          );
        }
      }
    });
  });
}
