# 숨결지기 대시보드 (`com.atlas.app.nuni_dashboard`)

LG atlas 파이 5에서 도는 Flutter 앱이다. 융합 서비스 `com.atlas.NuniFusion1`(D-Bus)에서 상태를 받아 보여 주고, 울음이 끝나면
"무엇으로 그쳤나요?" 답을 받아 서비스에 보낸다. 답은 기기의 울음 기록(`/data/nuni/cry`)에 쌓이고, 10번 모이면 다음 울음부터
1·2순위 힌트가 나온다(`atlas/nuni` CryLog, `cry_personal.py`와 같은 계산).

- 화면: 15.8인치 1920×1080, 터치 없음(마우스·키보드). 1920×1080으로 그리고 다른 화면(7인치 1024×600 등)에는 비율 그대로 맞춘다.
- 시연 화면: `--dart-define=NUNI_DEMO=<장면>`로 빌드하면 센서 없이 가상 데이터로 뜬다(머리줄에 '시연 화면이에요' 표시).

## 화면

```
 숨결지기                                  센서 서비스 연결됨   화면 어둡게 D   오전 2:21
 아기가 울고 있어요.                                   ┌ 울음 도우미 ───────────────┐
 17초째예요. 오른쪽에서 먼저 해 볼 것을 확인해 보세요.   │ 먼저 해 볼 것  수유  52%   │
 ～～～～～ 숨의 수평선(호흡수로 오르내림) ～～～～～      │ [해 봤는데 안 그쳤어요]     │
                                                      │ 그다음  기저귀  31%        │
 호흡 41회/분 · 움직임 · 울음 · 아기                     │ [F 수유했어요]             │
 방   온도 22.6℃  습도 38%  이산화탄소 640ppm  밝기 3lx      습도가 낮아요. 가습기를…
 오늘 밤  19 ─ 20 ─ … ─ 7   (울음: 답함 청자 / 답 기다림 수유등, 수유 ●, 경보 ▲, 지금)
```

| 상태 | 왼쪽 큰 문장 | 울음 도우미 |
|---|---|---|
| 평소 | 편안하게 자고 있어요 / 조금 뒤척이고 있어요 … | 답이 몇 번 모였는지(점 10개), 힌트가 맞은 횟수, 최근 울음 |
| 울음 중 | 아기가 울고 있어요(수유등 색) | 답 10번 전: '아직 답이 적어요' / 뒤: 먼저 해 볼 것 1·2순위 + '해 봤는데 안 그쳤어요' |
| 울음 뒤 | (수면 상태) | **무엇으로 그쳤나요?** 수유 · 기저귀 · 안아주기·자세 바꾸기 / 저절로 · 아파 보임 · 기타 · 모름 |
| 경보 | 지금 아기를 확인해 주세요(경보색) + 이유, 화면 테두리 깜빡임 | — |
| 연결 끊김 | 센서 서비스와 연결이 끊겼어요 + 확인할 것 | — |

키보드: `1`~`7` 울음 답(버튼 순서) · `Esc` 나중에 답하기 · `F` 수유 기록 · `Enter` 경보 확인 · `D` 화면 어둡게 · `M` 움직임 줄이기.
버튼은 마우스로 누르거나 `Tab`으로 옮겨 `Enter`로 누른다(초점은 상아색 테두리).

### 디자인

- 밤의 아기방: 먹빛 밤하늘 `#0E1322` 바탕, 달빛 상아 `#ECE5D3` 글자. 색은 뜻으로만 — 청자 `#8DD3BE` 숨·정상·기록됨, 수유등
  `#F1B35A` 살펴보기·답 기다림, 경보 `#FF6A5C` 지금 확인.
- 기기가 하는 말(상태 문장·질문)은 명조(Noto Serif KR), 숫자·버튼은 고딕(Noto Sans KR). 수치는 카드로 나누지 않고 글자 위계로만.
- 화면에서 늘 움직이는 것은 '숨의 수평선' 하나 — 측정한 호흡수에 맞춰 제자리에서 오르내린다(실제 레이더 파형이 아니라는 설명을 붙임).
- 힌트는 '기록으로 본 가능성 %'와 '참고용 추측'으로만 말한다(의료기기가 아님).

## 서비스와 주고받는 것 (`com.atlas.NuniFusion1`, 시스템 버스)

| 종류 | 이름 | 쓰는 곳 |
|---|---|---|
| 신호 | `StateChanged(s)` | 왼쪽 문장·경보(`fusion/state`) |
| 신호 | `Alert(ss)` · `CryEpisode(s)` | 받으면 바로 울음 기록·경보를 다시 읽음 |
| 메서드 | `GetSensors()` 1초마다 | 호흡·움직임·울음 점수·방 환경 |
| 메서드 | `GetCryLog()` 4초마다 | 지금 울음·힌트·답 기다리는 울음·최근 울음·마지막 수유 |
| 메서드 | `GetAlerts()` | 오늘 밤 줄의 경보 |
| 메서드 | `AnswerCry(id, action)` · `AnswerCryNot(id, action)` · `LogFeed()` | 답 버튼 · '안 그쳤어요' · 수유했어요 |

앱은 `package-manager` 그룹으로 실행된다. 패키지 관리자가 서비스 설치 때 만드는 D-Bus 정책은 기본으로 앱을 막으므로,
`atlas/nuni/deploy/setup_on_device.sh`가 이 그룹을 허용하는 정책을 넣는다(없으면 화면에 '접근할 권한이 없어요').

## 빌드 · 설치

개발 컨테이너(atlas SDK, flutter-atlas 3.27.4)에서:

```bash
cd atlas/dashboard
flutter-atlas pub get
flutter-atlas build atlas --ipk --release                              # 실제(D-Bus)
flutter-atlas build atlas --ipk --release --dart-define=NUNI_DEMO=live  # 시연: calm · crying · learning · question · alert · live
# → build/atlas/arm64/release/ipk/com.atlas.app.nuni_dashboard.ipk
```

파이 5(root)에서 — 같은 버전은 다시 설치되지 않으므로 지우고 설치한다:

```bash
busctl --system call com.atlas.PackageManager1 /com/atlas/PackageManager1 com.atlas.PackageManager1 Remove s com.atlas.app.nuni_dashboard
busctl --system call com.atlas.PackageManager1 /com/atlas/PackageManager1 com.atlas.PackageManager1 Install ss com.atlas.app.nuni_dashboard /tmp/com.atlas.app.nuni_dashboard.ipk
busctl --system call com.atlas.AppManager1 /com/atlas/AppManager1 com.atlas.AppManager1 Start s com.atlas.app.nuni_dashboard
busctl --system call com.atlas.WindowManager1 /com/atlas/WindowManager1 com.atlas.WindowManager1 CaptureScreen s /tmp/shot.png   # 화면 캡처
```

홈 화면 앱 목록에도 '숨결지기'로 보인다. 앱 매니저 등록(RUNNING)은 Flutter 실행기가 알아서 한다.

## 글꼴

`assets/fonts/` — Noto Sans KR · Noto Serif KR 가변 글꼴(SIL OFL 1.1)을 `tool/subset_fonts.py`로 줄였다. 명조는 코드에 적힌 한글만
담으므로, 명조로 쓰는 문구를 바꾸면 이 스크립트를 다시 돌린다. 자세한 것은 `assets/fonts/FONTS.md`.

## 기기에서 확인한 것 (2026-10-20, 파이 5)

- 시연 장면 6개를 기기 화면 캡처로 확인(평온 · 울음 중 힌트 · 학습 중 · 질문 · 경보 · 반복).
- 실제 모드: 서비스 연결, 울음 녹음(ESC-50 아기 울음 15초 + 정적 14초 반복)으로 울음 시작 → 끝 → `AnswerCry(수유)` →
  `AnswerCryNot(기저귀)`까지 기록 파일·화면 갱신 확인(시험 기록은 `/tmp`에 따로, 실제 기록은 그대로).
- 아직: 마우스·키보드로 직접 누르는 확인(원격으로는 누를 수 없음), 15.8인치 화면, 레이더·환경 센서가 붙은 상태.
