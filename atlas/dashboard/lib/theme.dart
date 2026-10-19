import 'dart:ui' show FontFeature, FontVariation;

import 'package:flutter/material.dart';

/// 밤의 아기방 — 먹빛 밤하늘 바탕에 달빛 상아 글자. 색은 뜻으로만 쓴다.
///   청자(breath)  숨 · 정상 · 기록됨
///   수유등(lamp)  살펴보기 · 답을 기다림
///   경보(alarm)   지금 확인
class Palette {
  static const ink = Color(0xFF0E1322);
  static const inkRaised = Color(0xFF151C30);
  static const line = Color(0xFF263049);
  static const moon = Color(0xFFECE5D3);
  static const moonDim = Color(0xFFC9C2B2);
  static const mist = Color(0xFF8E97AE);
  static const breath = Color(0xFF8DD3BE);
  static const lamp = Color(0xFFF1B35A);
  static const alarm = Color(0xFFFF6A5C);
}

/// 1920×1080 기준으로 그리고 화면 크기에 맞춰 통째로 줄이거나 늘린다(main.dart의 FittedBox).
/// 글자 크기 단계: 16 · 18 · 22 · 26 · 30 · 40 · 56 · 84
class Typo {
  /// 명조 — 기기가 부모에게 건네는 말(상태 문장·질문)
  static TextStyle serif(double size, {double weight = 500, Color color = Palette.moon, double height = 1.22}) =>
      TextStyle(
        fontFamily: 'NotoSerifKR',
        fontSize: size,
        height: height,
        color: color,
        fontWeight: _weight(weight),
        fontVariations: [FontVariation('wght', weight)],
        letterSpacing: size >= 40 ? -0.012 * size : 0,
      );

  /// 고딕 — 화면 글자·숫자
  static TextStyle sans(double size,
          {double weight = 420, Color color = Palette.moon, double height = 1.38, bool tabular = false}) =>
      TextStyle(
        fontFamily: 'NotoSansKR',
        fontSize: size,
        height: height,
        color: color,
        fontWeight: _weight(weight),
        fontVariations: [FontVariation('wght', weight)],
        fontFeatures: tabular ? const [FontFeature.tabularFigures()] : null,
        letterSpacing: size >= 40 ? -0.01 * size : 0,
      );

  static FontWeight _weight(double w) {
    final i = ((w / 100).round().clamp(1, 9)) - 1;
    return FontWeight.values[i];
  }
}

ThemeData nuniTheme() => ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: Palette.ink,
      fontFamily: 'NotoSansKR',
      colorScheme: const ColorScheme.dark(
        surface: Palette.ink,
        primary: Palette.breath,
        secondary: Palette.lamp,
        error: Palette.alarm,
        onSurface: Palette.moon,
      ),
      focusColor: Palette.moon.withValues(alpha: 0.16),
      hoverColor: Palette.moon.withValues(alpha: 0.06),
      splashFactory: NoSplash.splashFactory,
    );
