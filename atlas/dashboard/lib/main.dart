// 숨결지기 대시보드 — LG atlas(파이 5)의 Flutter 앱.
// 융합 서비스 com.atlas.NuniFusion1(D-Bus)에서 상태를 받아 보여 주고, 울음이 끝나면 "무엇으로 그쳤나요?" 답을 보낸다.
// 센서 없이 화면만 볼 때: flutter-atlas build atlas --ipk --release --dart-define=NUNI_DEMO=calm (data/demo_store.dart)
import 'package:flutter/material.dart';

import 'data/dbus_store.dart';
import 'data/demo_store.dart';
import 'data/store.dart';
import 'theme.dart';
import 'ui/dashboard.dart';

const _demo = String.fromEnvironment('NUNI_DEMO');

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  final NuniStore store = _demo.isEmpty ? DbusStore() : DemoStore(_demo);
  store.start();
  runApp(NuniApp(store: store));
}

class NuniApp extends StatelessWidget {
  const NuniApp({super.key, required this.store});
  final NuniStore store;

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: '숨결지기',
        debugShowCheckedModeBanner: false,
        theme: nuniTheme(),
        home: Scaffold(
          backgroundColor: Palette.ink,
          // 1920×1080으로 그리고 실제 화면(15.8인치 1920×1080, 7인치 1024×600 등)에 맞춰 비율 그대로 맞춘다
          body: SizedBox.expand(
            child: FittedBox(
              fit: BoxFit.contain,
              child: SizedBox(width: 1920, height: 1080, child: Dashboard(store: store)),
            ),
          ),
        ),
      );
}
