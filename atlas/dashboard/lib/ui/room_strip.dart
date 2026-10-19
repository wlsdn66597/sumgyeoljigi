import 'package:flutter/material.dart';

import '../copy.dart';
import '../data/store.dart';
import '../theme.dart';

/// 방 환경 한 줄 — 온도 · 습도 · 이산화탄소 · 밝기와 지금 할 일(fusion/state의 환경 권고)
class RoomStrip extends StatelessWidget {
  const RoomStrip({super.key, required this.store});
  final NuniStore store;

  @override
  Widget build(BuildContext context) {
    final r = store.sensors.room;
    final advice = store.state.actions.map((a) => roomAdvice[a]).whereType<String>().toList();
    if (r == null) {
      return Row(children: [
        SizedBox(width: 150, child: Text('방', style: Typo.sans(22, weight: 520, color: Palette.mist))),
        Text('환경 센서를 연결하면 온도, 습도, 이산화탄소, 밝기를 보여 드려요.', style: Typo.sans(22, color: Palette.moonDim)),
      ]);
    }
    String f(double? v, int digits) => v == null ? '–' : v.toStringAsFixed(digits);
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        SizedBox(width: 150, child: Text('방', style: Typo.sans(22, weight: 520, color: Palette.mist))),
        _Value('온도', f(r.temp, 1), '℃'),
        _Value('습도', f(r.humidity, 0), '%'),
        _Value('이산화탄소', f(r.co2, 0), 'ppm'),
        _Value('밝기', f(r.lux, 0), 'lx'),
        const Spacer(),
        Text(
          advice.isEmpty ? '방 환경이 알맞아요.' : advice.first,
          style: Typo.sans(24, weight: advice.isEmpty ? 420 : 520, color: advice.isEmpty ? Palette.mist : Palette.lamp),
        ),
      ],
    );
  }
}

class _Value extends StatelessWidget {
  const _Value(this.label, this.value, this.unit);
  final String label, value, unit;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(right: 44),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.baseline,
          textBaseline: TextBaseline.alphabetic,
          children: [
            Text(label, style: Typo.sans(20, color: Palette.mist)),
            const SizedBox(width: 10),
            Text(value, style: Typo.sans(30, weight: 520, tabular: true)),
            const SizedBox(width: 4),
            Text(unit, style: Typo.sans(20, color: Palette.mist)),
          ],
        ),
      );
}
