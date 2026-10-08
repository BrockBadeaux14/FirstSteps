import 'package:flutter/material.dart';

import 'replay/sample_replays.dart';

void main() {
  runApp(const FirstStepsApp());
}

class FirstStepsApp extends StatelessWidget {
  const FirstStepsApp({super.key});

  // The runner's left-limb blue.
  static const _seed = Color(0xFF4285F5);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FirstSteps',
      theme: ThemeData(colorScheme: .fromSeed(seedColor: _seed)),
      darkTheme: ThemeData(
        colorScheme: .fromSeed(seedColor: _seed, brightness: .dark),
      ),
      home: const SampleReplayScreen(),
    );
  }
}
