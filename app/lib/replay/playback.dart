// One playback clock, shared by every replay on screen.

import 'package:flutter/scheduler.dart';
import 'package:flutter/widgets.dart';

/// Playback time in seconds, driven by a [Ticker]. Each replay picks its own
/// frame for the time (see `Replay.frameIndexAt`), so replays of different
/// lengths stay in step and the shorter one holds its last frame.
class PlaybackController extends ChangeNotifier {
  PlaybackController({required TickerProvider vsync, required this.duration})
      : assert(duration >= 0) {
    _ticker = vsync.createTicker(_tick);
  }

  static const speeds = [0.25, 0.5, 1.0, 2.0];

  /// Length of the longest replay, in seconds.
  final double duration;

  late final Ticker _ticker;
  Duration _lastElapsed = Duration.zero;
  bool _resumeAfterScrub = false;

  double get time => _time;
  double _time = 0;

  double get speed => _speed;
  double _speed = 1.0;
  set speed(double value) {
    assert(value > 0);
    if (value == _speed) return;
    _speed = value;
    notifyListeners();
  }

  bool get isPlaying => _ticker.isActive;

  bool get atEnd => _time >= duration;

  /// Plays from the current time, or from the start if playback had ended.
  void play() {
    if (isPlaying) return;
    if (atEnd) _time = 0;
    _lastElapsed = Duration.zero;
    _ticker.start();
    notifyListeners();
  }

  void pause() {
    if (!isPlaying) return;
    _ticker.stop();
    notifyListeners();
  }

  void togglePlay() => isPlaying ? pause() : play();

  void seek(double t) {
    _time = t.clamp(0.0, duration);
    notifyListeners();
  }

  /// Pauses while the user drags the slider; [endScrub] resumes if it was
  /// playing.
  void beginScrub() {
    _resumeAfterScrub = isPlaying;
    pause();
  }

  void endScrub() {
    if (_resumeAfterScrub && !atEnd) play();
    _resumeAfterScrub = false;
  }

  void _tick(Duration elapsed) {
    final dt = (elapsed - _lastElapsed).inMicroseconds / Duration.microsecondsPerSecond;
    _lastElapsed = elapsed;
    _time += dt * _speed;
    if (_time >= duration) {
      _time = duration;
      _ticker.stop();
    }
    notifyListeners();
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }
}
