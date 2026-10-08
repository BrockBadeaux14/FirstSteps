// Golden images of the replay player. After an intended change to the drawing,
// regenerate them with: flutter test --update-goldens test/golden_test.dart

import 'package:first_steps/replay/replay.dart';
import 'package:first_steps/replay/replay_screen.dart';
import 'package:first_steps/replay/replay_view.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'fixtures.dart';

Future<void> pumpPane(
  WidgetTester tester,
  Replay replay,
  double time, {
  required String title,
}) async {
  await tester.pumpWidget(MaterialApp(
    debugShowCheckedModeBanner: false,
    home: Center(
      child: SizedBox(
        width: 390,
        height: 320,
        child: ReplayView(
          replay: replay,
          time: time,
          title: title,
          subtitle: formatSteps(replay.meta.checkpointStep),
        ),
      ),
    ),
  ));
}

void setSize(WidgetTester tester, Size size) {
  tester.view.physicalSize = size * 2;
  tester.view.devicePixelRatio = 2;
  addTearDown(tester.view.reset);
}

void main() {
  testWidgets('after: start', (tester) async {
    await pumpPane(tester, sampleAfter, 0, title: 'After');
    await expectLater(
        find.byType(ReplayView), matchesGoldenFile('goldens/after_start.png'));
  });

  testWidgets('after: mid-stride at 3 s', (tester) async {
    await pumpPane(tester, sampleAfter, 3.0, title: 'After');
    await expectLater(
        find.byType(ReplayView), matchesGoldenFile('goldens/after_3s.png'));
  });

  testWidgets('after: passing the 10 m line', (tester) async {
    var frame = 0;
    while (sampleAfter.frames[frame].pose(0).x < 9.6) {
      frame++;
    }
    await pumpPane(tester, sampleAfter, frame * sampleAfter.meta.dt,
        title: 'After');
    await expectLater(
        find.byType(ReplayView), matchesGoldenFile('goldens/after_10m.png'));
  });

  testWidgets('after: the end, no fall', (tester) async {
    await pumpPane(tester, sampleAfter, 10, title: 'After');
    await expectLater(
        find.byType(ReplayView), matchesGoldenFile('goldens/after_end.png'));
  });

  testWidgets('before: the fall', (tester) async {
    await pumpPane(tester, sampleBefore, 0.575, title: 'Before · 10%');
    await expectLater(
        find.byType(ReplayView), matchesGoldenFile('goldens/before_fall.png'));
  });

  testWidgets('before: on the ground', (tester) async {
    await pumpPane(tester, sampleBefore, 2, title: 'Before · 10%');
    await expectLater(find.byType(ReplayView),
        matchesGoldenFile('goldens/before_end.png'));
  });

  for (final (name, size) in [
    ('portrait', const Size(390, 844)),
    ('landscape', const Size(844, 390)),
  ]) {
    testWidgets('screen: $name', (tester) async {
      setSize(tester, size);
      await tester.pumpWidget(MaterialApp(
        debugShowCheckedModeBanner: false,
        home: ReplayScreen(
          before: sampleBefore,
          after: sampleAfter,
          autoplay: false,
        ),
      ));
      await tester.tap(find.byTooltip('Play'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 400));
      await tester.tap(find.byTooltip('Pause'));
      await tester.pump();
      await expectLater(find.byType(ReplayScreen),
          matchesGoldenFile('goldens/screen_$name.png'));
    });
  }
}
