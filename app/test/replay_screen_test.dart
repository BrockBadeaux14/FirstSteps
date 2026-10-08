import 'package:first_steps/main.dart';
import 'package:first_steps/replay/replay.dart';
import 'package:first_steps/replay/replay_screen.dart';
import 'package:first_steps/replay/replay_view.dart';
import 'package:first_steps/replay/sample_replays.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import 'fixtures.dart';

Finder inPane(String pane, Finder finder) =>
    find.descendant(of: find.byKey(ValueKey(pane)), matching: finder);

/// Expects the pane's HUD to show [replay] at playback time [t].
void expectHud(String pane, Replay replay, double t) {
  final frame = replay.frameIndexAt(t);
  expect(inPane(pane, find.text(formatSeconds(replay.frames[frame].t))),
      findsOneWidget,
      reason: '$pane time at $t s');
  expect(inPane(pane, find.text(formatMeters(replay.distanceAt(frame)))),
      findsOneWidget,
      reason: '$pane distance at $t s');
}

Future<void> pumpScreen(WidgetTester tester, {Size? size}) async {
  if (size != null) {
    tester.view.physicalSize = size * 3;
    tester.view.devicePixelRatio = 3;
    addTearDown(tester.view.reset);
  }
  await tester.pumpWidget(MaterialApp(
    home: ReplayScreen(
      before: sampleBefore,
      after: sampleAfter,
      autoplay: false,
    ),
  ));
}

void main() {
  testWidgets('the app opens on the bundled replays', (tester) async {
    await tester.pumpWidget(const FirstStepsApp());
    expect(find.byType(CircularProgressIndicator), findsOneWidget);
    // Asset loading is real I/O, outside the test's fake clock.
    for (var i = 0; i < 50 && find.byType(ReplayScreen).evaluate().isEmpty; i++) {
      await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 20)));
      await tester.pump();
    }
    expect(find.byType(ReplayScreen), findsOneWidget);
    expect(find.text('Sprint replay'), findsOneWidget);
    expect(find.text('Before · 10%'), findsOneWidget);
    expect(find.text('After'), findsOneWidget);
    expect(find.text('5.9M steps'), findsOneWidget);
    expect(find.text('59.0M steps'), findsOneWidget);
    // It starts playing on its own.
    expect(find.byTooltip('Pause'), findsOneWidget);
    await tester.pump(); // First tick.
    await tester.pump(const Duration(milliseconds: 500));
    expect(inPane('after', find.text('0.50 s')), findsOneWidget);
    // Stop the clock before the test ends.
    await tester.tap(find.byTooltip('Pause'));
    await tester.pump();
  });

  testWidgets('loads both sample assets from the bundle', (tester) async {
    final (before, after) =
        (await tester.runAsync(() => loadSampleReplays(rootBundle)))!;
    expect(before.meta.checkpointStep, 5898240);
    expect(after.meta.checkpointStep, 58982400);
    expect(after.frames, hasLength(401));
  });

  testWidgets('side by side in landscape, stacked in portrait',
      (tester) async {
    await pumpScreen(tester, size: const Size(844, 390));
    var before = tester.getRect(find.byKey(const ValueKey('before')));
    var after = tester.getRect(find.byKey(const ValueKey('after')));
    expect(after.left, greaterThan(before.right));
    expect(after.top, before.top);

    await pumpScreen(tester, size: const Size(390, 844));
    before = tester.getRect(find.byKey(const ValueKey('before')));
    after = tester.getRect(find.byKey(const ValueKey('after')));
    expect(after.top, greaterThan(before.bottom));
    expect(after.left, before.left);
  });

  testWidgets('play advances both panes at 40 frames a second',
      (tester) async {
    await pumpScreen(tester);
    expectHud('before', sampleBefore, 0);
    expectHud('after', sampleAfter, 0);

    await tester.tap(find.byTooltip('Play'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
    expectHud('before', sampleBefore, 0.5);
    expectHud('after', sampleAfter, 0.5);
    expect(inPane('after', find.text('0.50 s')), findsOneWidget);

    await tester.pump(const Duration(milliseconds: 500));
    expectHud('after', sampleAfter, 1.0);
    // The before replay ended at 1.575 s and holds its last frame.
    await tester.pump(const Duration(seconds: 1));
    expectHud('before', sampleBefore, 2.0);
    expect(inPane('before', find.text('1.58 s')), findsOneWidget);
    expectHud('after', sampleAfter, 2.0);

    await tester.tap(find.byTooltip('Pause'));
    await tester.pump();
    await tester.pump(const Duration(seconds: 1));
    expectHud('after', sampleAfter, 2.0);
  });

  testWidgets('the slider scrubs both panes', (tester) async {
    await pumpScreen(tester);
    await tester.tap(find.byType(Slider));
    await tester.pump();
    expectHud('after', sampleAfter, 5.0);
    expect(inPane('after', find.text('5.00 s')), findsOneWidget);
    expectHud('before', sampleBefore, 5.0);
    expect(find.byTooltip('Play'), findsOneWidget);
  });

  testWidgets('the speed menu changes the playback rate', (tester) async {
    await pumpScreen(tester);
    expect(find.text('1×'), findsOneWidget);
    await tester.tap(find.byTooltip('Playback speed'));
    await tester.pumpAndSettle();
    for (final label in ['0.25×', '0.5×', '2×']) {
      expect(find.text(label), findsOneWidget);
    }
    await tester.tap(find.text('2×'));
    await tester.pumpAndSettle();
    expect(find.text('2×'), findsOneWidget);

    await tester.tap(find.byTooltip('Play'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
    expectHud('after', sampleAfter, 1.0);
    await tester.tap(find.byTooltip('Pause'));
    await tester.pump();
  });

  testWidgets('the before pane shows the fall', (tester) async {
    await pumpScreen(tester);
    expect(find.textContaining('Fell'), findsNothing);

    // Just before the fall at 0.575 s, then at it.
    await tester.tap(find.byTooltip('Play'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 550));
    expect(find.textContaining('Fell'), findsNothing);
    await tester.pump(const Duration(milliseconds: 25));
    expect(inPane('before', find.text('Fell at 0.58 s')), findsOneWidget);
    await tester.tap(find.byTooltip('Pause'));
    await tester.pump();

    // The distance stops at the fall.
    expect(inPane('before', find.text(formatMeters(sampleBefore.meta.distanceM))),
        findsOneWidget);
    expect(inPane('after', find.textContaining('Fell')), findsNothing);
  });

  testWidgets('the after pane ends at 68.0 m with no fall', (tester) async {
    await pumpScreen(tester);
    await tester.tap(find.byTooltip('Play'));
    await tester.pump();
    await tester.pump(const Duration(seconds: 11));
    expect(inPane('after', find.text('10.00 s')), findsOneWidget);
    expect(inPane('after', find.text('68.0 m')), findsOneWidget);
    expect(inPane('after', find.text('No fall')), findsOneWidget);
    expect(inPane('before', find.text('Fell at 0.58 s')), findsOneWidget);
    // Stopped at the end; the button now replays.
    expect(find.byIcon(Icons.replay), findsOneWidget);
    await tester.tap(find.byTooltip('Play'));
    await tester.pump();
    expectHud('after', sampleAfter, 0);
    await tester.tap(find.byTooltip('Pause'));
    await tester.pump();
  });
}
