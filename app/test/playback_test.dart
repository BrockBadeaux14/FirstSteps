import 'package:first_steps/replay/playback.dart';
import 'package:flutter_test/flutter_test.dart';

import 'fixtures.dart';

void main() {
  PlaybackController controller(WidgetTester tester, {double duration = 10}) {
    final c = PlaybackController(vsync: tester, duration: duration);
    addTearDown(c.dispose);
    return c;
  }

  testWidgets('starts paused at 0', (tester) async {
    final c = controller(tester);
    expect(c.time, 0);
    expect(c.isPlaying, isFalse);
    expect(c.speed, 1.0);
  });

  testWidgets('plays in real time: 40 frames a second', (tester) async {
    final c = controller(tester);
    c.play();
    await tester.pump(); // First tick.
    await tester.pump(const Duration(milliseconds: 500));
    expect(c.time, closeTo(0.5, 1e-9));
    expect(sampleAfter.frameIndexAt(c.time), 20);
    for (var i = 0; i < 20; i++) {
      await tester.pump(const Duration(milliseconds: 25));
    }
    expect(c.time, closeTo(1.0, 1e-9));
    expect(sampleAfter.frameIndexAt(c.time), 40);
    c.pause();
  });

  testWidgets('pause holds the time', (tester) async {
    final c = controller(tester);
    c.play();
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 300));
    c.pause();
    await tester.pump(const Duration(seconds: 2));
    expect(c.time, closeTo(0.3, 1e-9));
    expect(c.isPlaying, isFalse);
  });

  testWidgets('speed scales the clock', (tester) async {
    final c = controller(tester);
    c.speed = 0.25;
    c.play();
    await tester.pump();
    await tester.pump(const Duration(seconds: 1));
    expect(c.time, closeTo(0.25, 1e-9));
    c.speed = 2;
    await tester.pump(const Duration(seconds: 1));
    expect(c.time, closeTo(2.25, 1e-9));
    c.pause();
  });

  testWidgets('stops at the end and replays from the start', (tester) async {
    final c = controller(tester, duration: 1.5);
    c.play();
    await tester.pump();
    await tester.pump(const Duration(seconds: 2));
    expect(c.time, 1.5);
    expect(c.atEnd, isTrue);
    expect(c.isPlaying, isFalse);

    c.play();
    expect(c.time, 0);
    expect(c.isPlaying, isTrue);
    c.pause();
  });

  testWidgets('seek clamps to the replay', (tester) async {
    final c = controller(tester);
    c.seek(4.2);
    expect(c.time, 4.2);
    c.seek(-1);
    expect(c.time, 0);
    c.seek(12);
    expect(c.time, 10);
  });

  testWidgets('scrubbing pauses, then resumes only if it was playing',
      (tester) async {
    final c = controller(tester);
    c.play();
    c.beginScrub();
    expect(c.isPlaying, isFalse);
    c.seek(3);
    c.endScrub();
    expect(c.isPlaying, isTrue);
    expect(c.time, 3);

    c.pause();
    c.beginScrub();
    c.seek(4);
    c.endScrub();
    expect(c.isPlaying, isFalse);
  });
}
