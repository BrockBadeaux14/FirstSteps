// The replay.json format (sprinter-replay/1) written by sprinter/replay.py.
//
// The file carries everything needed to draw the runner: a skeleton (written
// once) and every body's 2D world pose per control step. The app needs no
// physics and no kinematics. See the README section "replay.json format".

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Color, Offset;

const String replayFormat = 'sprinter-replay/1';

/// A body's world pose: origin (x, z) in meters and angle in radians,
/// counter-clockwise positive with +x right and +z up.
class Pose {
  const Pose(this.x, this.z, this.angle);

  final double x;
  final double z;
  final double angle;

  /// World position of the point (u, v) given in this body's frame:
  /// (x + u cos a - v sin a, z + u sin a + v cos a).
  Offset toWorld(Offset local) {
    final c = math.cos(angle);
    final s = math.sin(angle);
    return Offset(
      x + local.dx * c - local.dy * s,
      z + local.dx * s + local.dy * c,
    );
  }
}

enum BodySide { left, right, center }

/// A body's shape in its own frame, in meters ([x, z]).
sealed class BodyGeom {
  const BodyGeom(this.radius);

  final double radius;
}

final class CapsuleGeom extends BodyGeom {
  const CapsuleGeom(this.from, this.to, super.radius);

  final Offset from;
  final Offset to;
}

final class SphereGeom extends BodyGeom {
  const SphereGeom(this.center, super.radius);

  final Offset center;
}

class Body {
  const Body({
    required this.name,
    required this.parent,
    required this.offset,
    required this.side,
    required this.depth,
    required this.color,
    required this.geom,
  });

  final String name;

  /// Name of the parent body, or null for the root (the torso).
  final String? parent;

  /// Body origin in the parent's frame.
  final Offset offset;
  final BodySide side;

  /// Sideways offset of the limb (+ is the runner's left, away from the camera).
  final double depth;
  final Color color;
  final BodyGeom geom;
}

class ReplayMeta {
  const ReplayMeta({
    required this.level,
    required this.dt,
    required this.numFrames,
    required this.drawOrder,
    required this.floorZ,
    required this.markerEveryM,
    required this.distanceM,
    required this.fell,
    required this.endS,
    required this.meanSpeedMps,
    required this.episodeSeconds,
    required this.checkpointStep,
    required this.seed,
    required this.configHash,
  });

  final String level;

  /// Seconds per frame (one control step): 0.025, so 40 frames per second.
  final double dt;
  final int numFrames;

  /// Body names, back to front as seen from the camera.
  final List<String> drawOrder;
  final double floorZ;
  final int markerEveryM;

  /// Torso x when the episode ended minus torso x at the start.
  final double distanceM;
  final bool fell;

  /// When the episode ended: the episode length, or the moment of the fall.
  /// A replay that fell keeps going for a second after this.
  final double endS;
  final double meanSpeedMps;
  final double episodeSeconds;
  final int checkpointStep;
  final int seed;
  final String configHash;
}

/// One frame per control step. Poses are stored flat, three numbers per body
/// in skeleton order.
class ReplayFrame {
  ReplayFrame(this.t, this._poses);

  final double t;
  final Float64List _poses;

  int get bodyCount => _poses.length ~/ 3;

  Pose pose(int body) =>
      Pose(_poses[3 * body], _poses[3 * body + 1], _poses[3 * body + 2]);
}

class Replay {
  Replay._({
    required this.meta,
    required this.skeleton,
    required this.frames,
    required this.drawOrder,
    required this.rootIndex,
    required this.endFrame,
  });

  final ReplayMeta meta;
  final List<Body> skeleton;
  final List<ReplayFrame> frames;

  /// Skeleton indices in `meta.drawOrder`.
  final List<int> drawOrder;

  /// The body without a parent (the torso). Distance is measured on it.
  final int rootIndex;

  /// Frame at `meta.endS`.
  final int endFrame;

  /// Time of the last frame.
  double get duration => (frames.length - 1) * meta.dt;

  /// The frame shown at playback time [t]: one frame per `dt`, clamped to the
  /// first and last frames.
  int frameIndexAt(double t) {
    // The epsilon keeps t = k * dt on frame k despite rounding.
    final i = (t / meta.dt + 1e-6).floor();
    return i.clamp(0, frames.length - 1);
  }

  /// Distance run by [frame], measured like eval.json: torso x minus torso x
  /// at the start. It stops at the fall, so the last frame gives
  /// `meta.distanceM`.
  double distanceAt(int frame) {
    final end = math.min(frame, endFrame);
    return frames[end].pose(rootIndex).x - frames[0].pose(rootIndex).x;
  }

  /// Whether the runner has fallen by [frame].
  bool hasFallenAt(int frame) => meta.fell && frame >= endFrame;

  /// Whether [frame] is at or past the end of the episode (10 s or the fall).
  bool hasEndedAt(int frame) => frame >= endFrame;

  static Replay parse(String source) {
    final Object? json;
    try {
      json = jsonDecode(source);
    } on FormatException catch (e) {
      throw FormatException('replay.json is not valid JSON: ${e.message}');
    }
    return Replay.fromJson(_map(json, 'replay'));
  }

  factory Replay.fromJson(Map<String, Object?> json) {
    final metaJson = _map(json['meta'], 'meta');
    final format = metaJson['format'];
    if (format != replayFormat) {
      throw FormatException(
          'Unsupported replay format "$format", expected "$replayFormat"');
    }
    final meta = ReplayMeta(
      level: _string(metaJson['level'], 'meta.level'),
      dt: _double(metaJson['dt'], 'meta.dt'),
      numFrames: _int(metaJson['num_frames'], 'meta.num_frames'),
      drawOrder: _list(metaJson['draw_order'], 'meta.draw_order')
          .map((e) => _string(e, 'meta.draw_order[]'))
          .toList(),
      floorZ: _double(metaJson['floor_z'], 'meta.floor_z'),
      markerEveryM: _int(metaJson['marker_every_m'], 'meta.marker_every_m'),
      distanceM: _double(metaJson['distance_m'], 'meta.distance_m'),
      fell: _bool(metaJson['fell'], 'meta.fell'),
      endS: _double(metaJson['end_s'], 'meta.end_s'),
      meanSpeedMps: _double(metaJson['mean_speed_mps'], 'meta.mean_speed_mps'),
      episodeSeconds:
          _double(metaJson['episode_seconds'], 'meta.episode_seconds'),
      checkpointStep: _int(metaJson['checkpoint_step'], 'meta.checkpoint_step'),
      seed: _int(metaJson['seed'], 'meta.seed'),
      configHash: _string(metaJson['config_hash'], 'meta.config_hash'),
    );
    if (meta.dt <= 0) {
      throw FormatException('meta.dt must be positive, got ${meta.dt}');
    }

    final skeleton = <Body>[
      for (final (i, b) in _list(json['skeleton'], 'skeleton').indexed)
        _body(_map(b, 'skeleton[$i]'), 'skeleton[$i]'),
    ];
    if (skeleton.isEmpty) throw const FormatException('skeleton is empty');
    final index = <String, int>{};
    for (final (i, b) in skeleton.indexed) {
      if (index.containsKey(b.name)) {
        throw FormatException('skeleton has two bodies named "${b.name}"');
      }
      index[b.name] = i;
    }
    for (final b in skeleton) {
      if (b.parent != null && !index.containsKey(b.parent)) {
        throw FormatException(
            'Body "${b.name}" has unknown parent "${b.parent}"');
      }
    }
    final roots = [
      for (final (i, b) in skeleton.indexed)
        if (b.parent == null) i,
    ];
    if (roots.length != 1) {
      throw FormatException(
          'skeleton must have one root body, found ${roots.length}');
    }

    final drawOrder = <int>[];
    for (final name in meta.drawOrder) {
      final i = index[name];
      if (i == null) {
        throw FormatException('meta.draw_order names unknown body "$name"');
      }
      drawOrder.add(i);
    }
    if (drawOrder.toSet().length != skeleton.length) {
      throw const FormatException(
          'meta.draw_order must list every body exactly once');
    }

    final frames = <ReplayFrame>[];
    for (final (k, f) in _list(json['frames'], 'frames').indexed) {
      final frame = _map(f, 'frames[$k]');
      final poses = _list(frame['poses'], 'frames[$k].poses');
      if (poses.length != skeleton.length) {
        throw FormatException('frames[$k] has ${poses.length} poses for '
            '${skeleton.length} bodies');
      }
      final flat = Float64List(3 * poses.length);
      for (final (b, p) in poses.indexed) {
        final pose = _list(p, 'frames[$k].poses[$b]');
        if (pose.length != 3) {
          throw FormatException('frames[$k].poses[$b] must be [x, z, angle]');
        }
        for (var j = 0; j < 3; j++) {
          flat[3 * b + j] = _double(pose[j], 'frames[$k].poses[$b][$j]');
        }
      }
      frames.add(ReplayFrame(_double(frame['t'], 'frames[$k].t'), flat));
    }
    if (frames.isEmpty) throw const FormatException('frames is empty');
    if (frames.length != meta.numFrames) {
      throw FormatException('meta.num_frames is ${meta.numFrames} but the '
          'file has ${frames.length} frames');
    }
    final lastT = (frames.length - 1) * meta.dt;
    if ((frames.last.t - lastT).abs() > 1e-3) {
      throw FormatException('The last frame is at t = ${frames.last.t}, '
          'expected $lastT for one frame per dt = ${meta.dt}');
    }

    return Replay._(
      meta: meta,
      skeleton: skeleton,
      frames: frames,
      drawOrder: drawOrder,
      rootIndex: roots.single,
      endFrame: (meta.endS / meta.dt).round().clamp(0, frames.length - 1),
    );
  }
}

Body _body(Map<String, Object?> json, String where) {
  final geomJson = _map(json['geom'], '$where.geom');
  final radius = _double(geomJson['radius'], '$where.geom.radius');
  final BodyGeom geom = switch (geomJson['type']) {
    'capsule' => CapsuleGeom(
        _point(geomJson['from'], '$where.geom.from'),
        _point(geomJson['to'], '$where.geom.to'),
        radius,
      ),
    'sphere' =>
      SphereGeom(_point(geomJson['center'], '$where.geom.center'), radius),
    final type => throw FormatException('$where.geom has unknown type "$type"'),
  };
  final parent = json['parent'];
  return Body(
    name: _string(json['name'], '$where.name'),
    parent: parent == null ? null : _string(parent, '$where.parent'),
    offset: _point(json['offset'], '$where.offset'),
    side: switch (json['side']) {
      'left' => BodySide.left,
      'right' => BodySide.right,
      'center' => BodySide.center,
      final side => throw FormatException('$where.side is "$side"'),
    },
    depth: _double(json['depth'], '$where.depth'),
    color: _color(json['color'], '$where.color'),
    geom: geom,
  );
}

final _hexColor = RegExp(r'^#[0-9a-fA-F]{6}$');

Color _color(Object? value, String where) {
  final s = _string(value, where);
  if (!_hexColor.hasMatch(s)) {
    throw FormatException('$where must be "#rrggbb", got "$s"');
  }
  return Color(0xFF000000 | int.parse(s.substring(1), radix: 16));
}

Offset _point(Object? value, String where) {
  final p = _list(value, where);
  if (p.length != 2) throw FormatException('$where must be [x, z]');
  return Offset(_double(p[0], '$where[0]'), _double(p[1], '$where[1]'));
}

Map<String, Object?> _map(Object? value, String where) => value is Map
    ? value.cast<String, Object?>()
    : throw FormatException('$where must be an object');

List<Object?> _list(Object? value, String where) =>
    value is List ? value : throw FormatException('$where must be a list');

String _string(Object? value, String where) =>
    value is String ? value : throw FormatException('$where must be a string');

double _double(Object? value, String where) => value is num
    ? value.toDouble()
    : throw FormatException('$where must be a number');

int _int(Object? value, String where) =>
    value is int ? value : throw FormatException('$where must be an integer');

bool _bool(Object? value, String where) =>
    value is bool ? value : throw FormatException('$where must be true or false');
