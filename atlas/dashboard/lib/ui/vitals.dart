import 'dart:math';

import 'package:flutter/material.dart';

import '../data/store.dart';
import '../theme.dart';
import 'common.dart';

/// 호흡 · 움직임 · 울음 · 아기 위치 — 카드 없이 글자 위계로만 나눈 네 칸
class Vitals extends StatelessWidget {
  const Vitals({super.key, required this.store});
  final NuniStore store;

  @override
  Widget build(BuildContext context) {
    final fresh = store.radarFresh;
    final r = store.radar;
    final br = fresh ? r?.breathingRate : null;
    final mv = fresh ? r?.movement : null;
    final cry = store.sensors.cry;
    final crying = cry?.isCrying == true;
    final hist = store.breathHistory.where((p) => DateTime.now().difference(p.$1).inMinutes < 10).map((p) => p.$2).toList();

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _Cell(
          label: '호흡',
          value: br == null || br <= 0 ? '–' : br.round().toString(),
          unit: br == null || br <= 0 ? null : '회/분',
          note: !fresh
              ? '레이더 신호 없음'
              : (r?.apnea == true || br == null || br <= 0)
                  ? (r?.breathValid == false ? '호흡을 재지 못하는 자세예요' : '호흡이 잡히지 않아요')
                  : hist.length < 2
                      ? '기록을 모으는 중'
                      : '지난 10분 ${hist.reduce(min).round()}–${hist.reduce(max).round()}회',
          extra: Sparkline(points: store.breathHistory, width: 230, height: 40),
        ),
        _Cell(
          label: '움직임',
          value: mv == null ? '–' : (mv < 0.4 ? '적어요' : (mv < 0.7 ? '조금 많아요' : '많아요')),
          valueSize: 40,
          note: mv == null ? '레이더 신호 없음' : '레이더 움직임 ${mv.toStringAsFixed(2)}',
          extra: _Level(value: mv ?? 0, color: (mv ?? 0) >= 0.7 ? Palette.lamp : Palette.mist),
        ),
        _Cell(
          label: '울음',
          value: cry == null ? '–' : (crying ? '울고 있어요' : '조용해요'),
          valueSize: 40,
          valueColor: crying ? Palette.lamp : Palette.moon,
          note: cry == null ? '마이크 신호 없음' : '울음 소리 점수 ${cry.confidence.toStringAsFixed(2)}',
          extra: _Level(value: cry?.confidence ?? 0, color: crying ? Palette.lamp : Palette.mist),
        ),
        _Cell(
          label: '아기',
          value: !fresh ? '–' : (r?.presence == true ? '침대에 있어요' : '보이지 않아요'),
          valueSize: 40,
          note: !fresh || r?.distance == null ? '레이더 신호 없음' : '레이더에서 ${r!.distance!.toStringAsFixed(1)}m',
        ),
      ],
    );
  }
}

class _Cell extends StatelessWidget {
  const _Cell({required this.label, required this.value, this.unit, required this.note, this.extra, this.valueSize = 56, this.valueColor = Palette.moon});
  final String label, value, note;
  final String? unit;
  final Widget? extra;
  final double valueSize;
  final Color valueColor;

  @override
  Widget build(BuildContext context) => Expanded(
        child: Padding(
          padding: const EdgeInsets.only(right: 36),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label, style: Typo.sans(22, weight: 520, color: Palette.mist)),
              const SizedBox(height: 6),
              SizedBox(
                height: 76,
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.baseline,
                  textBaseline: TextBaseline.alphabetic,
                  children: [
                    Flexible(
                      child: Text(value,
                          maxLines: 1,
                          overflow: TextOverflow.fade,
                          softWrap: false,
                          style: Typo.sans(valueSize, weight: 560, color: valueColor, height: 1.3, tabular: true)),
                    ),
                    if (unit != null) ...[
                      const SizedBox(width: 8),
                      Text(unit!, style: Typo.sans(22, color: Palette.mist)),
                    ],
                  ],
                ),
              ),
              if (extra != null) ...[const SizedBox(height: 8), SizedBox(height: 40, child: extra)],
              const SizedBox(height: 8),
              Text(note, style: Typo.sans(18, color: Palette.mist)),
            ],
          ),
        ),
      );
}

/// 0~1 값 막대
class _Level extends StatelessWidget {
  const _Level({required this.value, required this.color});
  final double value;
  final Color color;

  @override
  Widget build(BuildContext context) => Align(
        alignment: Alignment.centerLeft,
        child: SizedBox(
          width: 230,
          height: 8,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: Stack(children: [
              Container(color: Palette.line),
              FractionallySizedBox(widthFactor: value.clamp(0, 1).toDouble(), child: Container(color: color)),
            ]),
          ),
        ),
      );
}
