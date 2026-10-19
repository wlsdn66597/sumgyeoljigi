import 'package:flutter/material.dart';

import '../copy.dart';
import '../data/store.dart';
import '../theme.dart';
import 'breath_horizon.dart';
import 'common.dart';

enum Mood { calm, watch, alarm, muted }

class HeroCopy {
  const HeroCopy(this.sentence, this.detail, this.tone, this.horizon);
  final String sentence;
  final String detail;
  final Mood tone;
  final HorizonMode horizon;
}

/// 지금 상태를 한 문장으로 — 우선순위: 연결 → 경보 → 울음 → 레이더 → 수면 상태
HeroCopy describe(NuniStore s) {
  final radar = s.radar;
  final br = radar?.breathingRate;
  final brText = br == null ? null : br.round().toString();
  if (s.link == Link.lost) {
    return HeroCopy('센서 서비스와 연결이 끊겼어요.', '${s.linkError ?? '응답이 없어요'}. 파이 5에서 nunifusion 서비스가 켜져 있는지 확인해 주세요.',
        Mood.muted, HorizonMode.waiting);
  }
  if (s.link == Link.connecting) {
    return const HeroCopy('센서 서비스에 연결하는 중이에요.', '잠시만 기다려 주세요.', Mood.muted, HorizonMode.waiting);
  }
  if (s.state.level == 'alert') {
    final why = s.state.reasons.map(reasonText).toSet().join(', ');
    return HeroCopy('지금 아기를 확인해 주세요.', why.isEmpty ? '경보가 울렸어요.' : '$why.', Mood.alarm,
        radar?.apnea == true || (br != null && br < 8) ? HorizonMode.stopped : HorizonMode.breathing);
  }
  final now = s.cry?.current;
  if (now != null || s.sensors.cry?.isCrying == true) {
    final secs = now == null ? 0 : DateTime.now().difference(now.t0).inSeconds;
    final hint = now?.hint;
    final detail = (hint != null && hint.show2)
        ? '${span(Duration(seconds: secs))}째예요. 오른쪽에서 먼저 해 볼 것을 확인해 보세요.'
        : '${secs > 0 ? '${span(Duration(seconds: secs))}째예요. ' : ''}울음이 그치면 무엇으로 그쳤는지 알려 주세요.';
    return HeroCopy('아기가 울고 있어요.', detail, Mood.watch, s.radarFresh ? HorizonMode.breathing : HorizonMode.waiting);
  }
  if (!s.radarFresh) {
    return const HeroCopy('레이더 신호를 기다려요.', '울음은 마이크로 계속 듣고 있어요. ESP32 레이더의 전원과 WiFi를 확인해 주세요.',
        Mood.muted, HorizonMode.waiting);
  }
  switch (s.state.sleepState) {
    case 'calm_sleep':
      return HeroCopy('편안하게 자고 있어요.', brText == null ? '호흡이 고르게 잡히고 있어요.' : '1분에 $brText번, 고르게 숨 쉬고 있어요.',
          Mood.calm, HorizonMode.breathing);
    case 'restless':
      return HeroCopy('조금 뒤척이고 있어요.', brText == null ? '움직임이 늘었어요.' : '움직임이 늘었어요. 호흡은 1분에 $brText번이에요.',
          Mood.calm, HorizonMode.breathing);
    case 'awake':
      return const HeroCopy('깨어 있는 것 같아요.', '큰 움직임이 이어지고 있어요.', Mood.watch, HorizonMode.breathing);
    case 'hold':
      return const HeroCopy('호흡을 다시 재고 있어요.', '호흡 신호가 약해서 잠깐 판단을 미뤘어요. 아기가 레이더 쪽을 향하고 있는지 봐 주세요.',
          Mood.muted, HorizonMode.waiting);
    default:
      return const HeroCopy('아기가 침대에 없는 것 같아요.', '레이더에 아기가 잡히지 않아요.', Mood.muted, HorizonMode.waiting);
  }
}

/// 상태 문장은 한 줄로 — 들어가지 않으면 84 → 76 → 68 → 60으로 줄인다(낱말 하나가 다음 줄로 떨어지지 않게)
class _OneLine extends StatelessWidget {
  const _OneLine(this.text, {required this.color});
  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) => LayoutBuilder(builder: (context, box) {
        var style = Typo.serif(84, weight: 520, color: color, height: 1.16);
        for (final size in const [84.0, 76.0, 68.0, 60.0]) {
          style = Typo.serif(size, weight: 520, color: color, height: 1.16);
          final tp = TextPainter(text: TextSpan(text: text, style: style), textDirection: TextDirection.ltr, maxLines: 1)
            ..layout();
          if (tp.width <= box.maxWidth) break;
        }
        return Text(text, style: style, maxLines: 2);
      });
}

Color toneColor(Mood t) => switch (t) {
      Mood.calm => Palette.moon,
      Mood.watch => Palette.lamp,
      Mood.alarm => Palette.alarm,
      Mood.muted => Palette.moonDim,
    };

class StatusHero extends StatelessWidget {
  const StatusHero({super.key, required this.store, required this.reduceMotion, this.onAcknowledge, this.acknowledged = false});
  final NuniStore store;
  final bool reduceMotion;
  final VoidCallback? onAcknowledge;
  final bool acknowledged;

  @override
  Widget build(BuildContext context) {
    final c = describe(store);
    final radar = store.radar;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        AnimatedSwitcher(
          duration: Duration(milliseconds: reduceMotion ? 0 : 450),
          child: Column(
            key: ValueKey(c.sentence),
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _OneLine(c.sentence, color: toneColor(c.tone)),
              const SizedBox(height: 18),
              SizedBox(
                width: 1040,
                child: Text(c.detail, style: Typo.sans(30, weight: 400, color: Palette.moonDim, height: 1.45)),
              ),
            ],
          ),
        ),
        if (c.tone == Mood.alarm && !acknowledged && onAcknowledge != null) ...[
          const SizedBox(height: 26),
          NuniButton(label: '확인했어요', keyHint: 'Enter', tone: Tone.alarm, onPressed: onAcknowledge, size: 26),
        ],
        const SizedBox(height: 18),
        SizedBox(
          height: c.tone == Mood.alarm && !acknowledged ? 120 : 190,
          width: double.infinity,
          child: Stack(children: [
            Positioned.fill(
              child: BreathHorizon(
                mode: c.horizon,
                rate: radar?.breathingRate ?? 30,
                movement: radar?.movement ?? 0,
                reduceMotion: reduceMotion,
              ),
            ),
            if (c.horizon == HorizonMode.breathing)
              Positioned(
                right: 0,
                bottom: 0,
                child: Text('호흡수로 그린 모양이에요', style: Typo.sans(16, color: Palette.mist)),
              ),
          ]),
        ),
      ],
    );
  }
}
