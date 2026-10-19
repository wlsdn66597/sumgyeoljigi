import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../copy.dart';
import '../data/store.dart';
import '../theme.dart';
import 'common.dart';
import 'cry_helper.dart';
import 'night_timeline.dart';
import 'room_strip.dart';
import 'status_hero.dart';
import 'vitals.dart';

/// 1920×1080 한 화면.
///   왼쪽  지금 상태 문장 + 숨의 수평선 + 호흡·움직임·울음·아기
///   오른쪽 울음 도우미
///   아래  방 환경 + 오늘 밤
/// 키보드: 1~7 울음 답 · Esc 나중에 · F 수유 기록 · Enter 경보 확인 · D 화면 어둡게 · M 움직임 줄이기
class Dashboard extends StatefulWidget {
  const Dashboard({super.key, required this.store});
  final NuniStore store;

  @override
  State<Dashboard> createState() => _DashboardState();
}

class _DashboardState extends State<Dashboard> with SingleTickerProviderStateMixin {
  String? _dismissed; // '나중에 답할게요'를 누른 울음
  String? _asking; // 최근 울음 목록에서 '답하기'를 누른 울음
  bool _ack = false; // 경보 확인
  bool _dim = false;
  bool _calmMotion = false;
  late final AnimationController _pulse =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 1100))..repeat(reverse: true);
  final _focus = FocusNode();

  NuniStore get s => widget.store;

  @override
  void initState() {
    super.initState();
    s.addListener(_onStore);
  }

  void _onStore() {
    if (s.state.level != 'alert' && _ack) _ack = false;
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    s.removeListener(_onStore);
    _pulse.dispose();
    _focus.dispose();
    super.dispose();
  }

  bool get _reduceMotion => _calmMotion || MediaQuery.of(context).disableAnimations;

  /// 지금 묻는 울음 — 목록에서 고른 울음, 아니면 막 끝난 울음('나중에'를 누르지 않았을 때). 답이 없는 것만.
  String? get _askingId {
    final log = s.cry;
    if (log == null || log.current != null) return null;
    String? open(String? id) {
      final e = log.episode(id);
      return e != null && e.action == null ? id : null;
    }

    return open(_asking) ?? (log.pending != _dismissed ? open(log.pending) : null);
  }

  KeyEventResult _onKey(FocusNode node, KeyEvent e) {
    if (e is! KeyDownEvent) return KeyEventResult.ignored;
    final k = e.logicalKey;
    final asking = _askingId;
    final log = s.cry;
    const digits = [
      LogicalKeyboardKey.digit1, LogicalKeyboardKey.digit2, LogicalKeyboardKey.digit3, LogicalKeyboardKey.digit4,
      LogicalKeyboardKey.digit5, LogicalKeyboardKey.digit6, LogicalKeyboardKey.digit7,
      LogicalKeyboardKey.numpad1, LogicalKeyboardKey.numpad2, LogicalKeyboardKey.numpad3, LogicalKeyboardKey.numpad4,
      LogicalKeyboardKey.numpad5, LogicalKeyboardKey.numpad6, LogicalKeyboardKey.numpad7,
    ];
    final di = digits.indexOf(k);
    if (di >= 0 && asking != null && log != null) {
      final i = di % 7;
      if (i < log.answers.length) {
        s.answer(asking, log.answers[i]);
        setState(() => _asking = null);
      }
      return KeyEventResult.handled;
    }
    if (k == LogicalKeyboardKey.escape && asking != null) {
      _later();
      return KeyEventResult.handled;
    }
    if (k == LogicalKeyboardKey.keyF) {
      s.logFeed();
      return KeyEventResult.handled;
    }
    if ((k == LogicalKeyboardKey.enter || k == LogicalKeyboardKey.numpadEnter) && s.state.level == 'alert' && !_ack) {
      setState(() => _ack = true);
      return KeyEventResult.handled;
    }
    if (k == LogicalKeyboardKey.keyD) {
      setState(() => _dim = !_dim);
      return KeyEventResult.handled;
    }
    if (k == LogicalKeyboardKey.keyM) {
      setState(() => _calmMotion = !_calmMotion);
      return KeyEventResult.handled;
    }
    return KeyEventResult.ignored;
  }

  void _later() => setState(() {
        _dismissed = s.cry?.pending;
        _asking = null;
      });

  @override
  Widget build(BuildContext context) {
    final alarm = s.state.level == 'alert' && !_ack;
    return Focus(
      focusNode: _focus,
      autofocus: true,
      onKeyEvent: _onKey,
      child: Stack(
        children: [
          Positioned.fill(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(72, 40, 72, 36),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _Header(store: s, dim: _dim, onDim: () => setState(() => _dim = !_dim)),
                  const SizedBox(height: 34),
                  Expanded(
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              StatusHero(
                                store: s,
                                reduceMotion: _reduceMotion,
                                acknowledged: _ack,
                                onAcknowledge: () => setState(() => _ack = true),
                              ),
                              const Spacer(),
                              Vitals(store: s),
                            ],
                          ),
                        ),
                        const SizedBox(width: 64),
                        SizedBox(
                          width: 600,
                          child: CryHelper(
                            store: s,
                            askingId: _askingId,
                            onAsk: (id) => setState(() => _asking = id),
                            onLater: _later,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 30),
                  Container(height: 1.5, color: Palette.line),
                  const SizedBox(height: 22),
                  RoomStrip(store: s),
                  const SizedBox(height: 22),
                  NightTimeline(store: s),
                ],
              ),
            ),
          ),
          if (alarm)
            Positioned.fill(
              child: IgnorePointer(
                child: AnimatedBuilder(
                  animation: _pulse,
                  builder: (_, __) => DecoratedBox(
                    decoration: BoxDecoration(
                      border: Border.all(
                        color: Palette.alarm.withValues(alpha: _reduceMotion ? 0.9 : 0.35 + 0.6 * _pulse.value),
                        width: 12,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          if (_dim) Positioned.fill(child: IgnorePointer(child: Container(color: Colors.black.withValues(alpha: 0.5)))),
        ],
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.store, required this.dim, required this.onDim});
  final NuniStore store;
  final bool dim;
  final VoidCallback onDim;

  @override
  Widget build(BuildContext context) {
    final (dot, text) = switch (store.link) {
      Link.ok => (Palette.breath, store.isDemo ? '시연 화면이에요. 실제 측정값이 아니에요' : '센서 서비스 연결됨'),
      Link.connecting => (Palette.mist, '연결하는 중'),
      Link.lost => (Palette.alarm, store.linkError ?? '연결이 끊겼어요'),
    };
    return SizedBox(
      height: 56,
      child: Row(
        children: [
          Text('숨결지기', style: Typo.serif(34, weight: 640)),
          const Spacer(),
          Container(width: 12, height: 12, decoration: BoxDecoration(color: dot, shape: BoxShape.circle)),
          const SizedBox(width: 12),
          Text(text, style: Typo.sans(20, color: Palette.mist)),
          const SizedBox(width: 40),
          NuniButton(label: dim ? '화면 밝게' : '화면 어둡게', keyHint: 'D', tone: Tone.quiet, size: 20, onPressed: onDim),
          const SizedBox(width: 40),
          Text(clock(DateTime.now()), style: Typo.sans(30, weight: 520, tabular: true)),
        ],
      ),
    );
  }
}
