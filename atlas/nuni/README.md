# nunifusion — 숨결지기 atlas 네이티브 서비스 (C++)

파이에서 Python으로 돌던 실시간 경로(레이더 변환 · 울음 감지 · 융합 판단)를 atlas용 C++로 옮긴 것이다.
서비스 이름은 `com.atlas.NuniFusion1`이고, 프로세스 하나에서 세 모듈이 돈다.

```
 MR60BHA2(ESPHome) ─ mr60bha2/* ─┐                         ┌─ sensor/radar ─┐
 ESP32(SCD41 등) ─── sensor/env ─┤      MQTT 브로커         │                 │
 ESP32(VL53L1X) ──── sensor/tof ─┤                          │                 │
                                 └──▶ (nuni-mosquitto) ◀───┤  [radar 모듈]   │
                                          │   ▲             │                 ▼
                                          │   └─ audio/cry ─┤  [cry 모듈] ◀── 마이크(PulseAudio)
                                          │                 │   YAMNet(TFLite)
                                          └── sensor/* ────▶│  [fusion 모듈] ── fusion/state · fusion/alert · control/action ─▶ MQTT
                                                            │        │
                                                            └────────┴─ D-Bus com.atlas.NuniFusion1 (상태 조회·신호) ─▶ 기기 안 앱
```

모듈끼리는 Python 분산 모드(`NUNI_BUS=mqtt`)와 똑같이 브로커를 거친다. 그래서 대시보드·기록기·Pi Zero 같은 다른 구독자는 Python 때와 같은 토픽과 같은 JSON을 그대로 받는다.

## Python → C++ 대응

| Python 원본 | C++ | 비고 |
|---|---|---|
| `topics.py` | `include/nuni/topics.hpp` | 토픽 이름·메시지 형식 동일 |
| `fusion.py` | `src/fusion.cpp` | `decide`, `single_sensor_decide`, `Fusion`(디바운스·쿨다운) |
| `sleep_state.py` | `src/sleep_state.cpp` | `NUNI_FUSION_CRY_THRESHOLD` 그대로 |
| `env_control.py` | `src/env_control.cpp` | |
| `personalization.py` | `src/personalization.cpp` | |
| `radar_bridge.py` | `src/radar_bridge.cpp` | 게이팅·중앙값·무호흡 유도, 환경변수 이름·기본값 동일 |
| `cry_model/yamnet_cry.py` | `src/cry_detector.cpp`, `src/yamnet_tflite.cpp` | 선형 리샘플(numpy와 같은 계산), 클래스 19·20 |
| `cry_classifier.py`(실물 경로) | `src/cry_detector.cpp` + `apps/nunifusion.cpp` | 1.5초 창, 임계 0.3, `cls`는 `"none"` |
| `cry_model/infer.py`의 `Mic` | `RingBuffer` + `src/audio_pulse.cpp` | 기기에 libpulse-simple이 없어 libpulse 기본 API로 녹음 |
| `bus.py`(MQTT) | `src/mqtt.cpp` | 의존성 없는 MQTT 3.1.1(QoS 0). 접속 확인 뒤 구독, 재접속 때 구독 복원 |
| (새로 만듦) | `src/dbus_service.cpp` | sdbus-c++ 2.x, LG 샘플과 같은 방식 |
| `cry_personal.py`(일부) | `src/cry_log.cpp` | 울음 구간 나누기(새로 만듦) · 답 쌓기(feedback · feedback_not · log_feed) · 맥락 × 이력 힌트 · 미리 낸 예측. 아래 '울음 기록' |

옮기지 않은 것: 기록·리포트(`recorder.py`, `cloud.py` 등), 울음 힌트의 소리 비교(`cry_personal.py`의 블록 특징 · 아기별 판별 분석 · 자가 검증), SCD41 읽기(ESP32로 옮기거나 2단계). 대시보드는 Flutter로 새로 만들었다(`atlas/dashboard`).

## 검증 결과 (개발 PC, 2026-10-19)

같은 입력을 Python 원본과 C++에 넣어 출력을 줄 단위로 비교했다(`tests/run_parity.sh`).

| 시험 | 규모 | 결과 |
|---|---|---|
| 판단 규칙(`decide`·`single`·수면 상태·환경 권고) | 시드 5개, 7.8만 건 | **완전 일치** |
| 융합 스트림(디바운스·쿨다운·개인화·발행 메시지 전체) | 6.6만 메시지 | **완전 일치** |
| 레이더 브리지(ESPHome 평문 → `sensor/radar`, 잡값·끊김·멀어짐·움직임·호흡 멈춤 섞음) | 10.6만 수신, 6.2만 스텝 | **완전 일치** |
| 리샘플(48k·44.1k·22.05k·8k → 16k), Python `round` | 520건, 8천 건 | **완전 일치** |
| YAMNet 출력(실수형 모델, 같은 WAV) | 50창 | 임베딩 상대 오차 최대 6.5×10⁻⁷ (TFLite 버전 차이 수준) |
| YAMNet 울음 점수(정수 양자화 모델) | 50창 | 47창 완전 일치, 3창은 TFLite 버전 차이로 다름 → **기기에서는 실수형 `yamnet_emb.tflite`를 쓴다** |
| MQTT 통합(mosquitto + nunifusion, 브로커 재시작 포함) | 40초 | 상태 50개 원본과 일치, 재접속 뒤 구독 복원, JSON 아닌 메시지 무시 |
| PulseAudio 녹음 → 울음 감지 | 25초 WAV 재생 | 울음 구간 감지(0.50·0.74), 판정 1회 24ms |
| D-Bus(세션 버스) | | `Ping`·`GetState`·`GetSensors`, `StateChanged`·`Alert`·`CryDetected` 신호 확인 |
| 같은 비교를 ARM64(aarch64, qemu)로 | 약 10만 줄 | **완전 일치** (`-ffp-contract=off`가 없으면 리샘플 2건이 마지막 자리에서 달랐음 → 넣음) |
| 메모리 검사(AddressSanitizer·UBSan) | 2.4만 줄 | 오류 없음 |

## 기기 구성과 실측 (2026-10-19)

```
[파이 5 메인 172.20.10.13] 마이크(AB13X) → nunifusion(ECO.AI YAMNet) ─┐
                           nuni-mosquitto(브로커) ◀───────────────────┘
                                  │ fusion/alert, control/vibrate
[파이 4 172.20.10.12]      nunivibe → PeripheralManager1 GPIO_18 → 진동 모듈
```

| 확인 | 결과 |
|---|---|
| 파이 5: IPK 설치(`PackageManager1.Install`) + `setup_on_device.sh` | ✅ `nunifusion`·`nuni-mosquitto` active, 부팅 자동 실행 등록, D-Bus `Ping` → `pong:2` |
| 파이 5: 울음 추론 | ✅ ECO.AI(TfLiteStrategy, cpu)로 마이크 1.5초 창 판정 13ms (2스레드) |
| 파이 4: `nunivibe` | ✅ 파이 5 브로커 접속, `control/vibrate`(test)·`fusion/alert`(alert) 받아 GPIO_18 패턴 출력 |
| 노트북 → 파이 5 브로커 → 파이 4 | ✅ 경보 한 건이 진동 패턴까지 이어짐 |

아직 안 한 것: 진동 모듈 실물 연결(핀 값은 ReadValue로만 확인), 레이더(ESP32) 연결, Flutter 앱의 D-Bus 신호 구독.

## 빌드

atlas 개발 컨테이너(SDK 환경이 잡힌 셸)에서:

```bash
cd atlas/nuni
mkdir -p build && cd build
cmake .. -DNUNI_REQUIRE_ALL=ON     # TFLite·libpulse·sdbus-c++ 중 하나라도 못 찾으면 멈춘다
make -j
```

- TFLite는 `pkg-config tensorflowlite`(SDK)로 찾고, CPU(XNNPACK)에 필요한 `tensorflowlite`·`tensorflowlite_xnnpack_delegate`만 링크한다(`atlas/yamnet_test`와 같음).
- 실행 파일은 옆의 `lib/`에서 공유 라이브러리를 먼저 찾는다(RPATH `$ORIGIN/lib`). 기기 시스템 경로는 건드리지 않는다.
- IPK는 LG 샘플처럼 `arc build`로 만든다. 단, `arc build`는 CMake install 규칙을 쓰지 않고 실행 파일·`serviceinfo.json`·D-Bus 파일만 묶는다(2026-10-19 확인). 모델·브로커·systemd 유닛·`setup_on_device.sh`는 아래 '기기에 올리기'처럼 서비스 폴더에 따로 넣는다.
- ECO.AI(`pkg-config ecoai-inference`)가 있으면 같이 빌드한다. 실행 때 `NUNI_YAMNET_ENGINE=auto`면 ECO.AI를 먼저 쓰고, 추론 플러그인이 없는 기기(Pi 4)에서는 TFLite 직접으로 넘어간다.
- `nunivibe`(파이 4 진동 노드)도 같은 빌드에서 나온다.

일반 리눅스 PC에서도 같은 명령으로 빌드된다. 없는 라이브러리 기능만 빠진다(예: TFLite가 없으면 울음 감지 없이 레이더·융합만).

### 기기에 같이 넣을 파일 (저장소에는 없음)

| 넣을 곳 | 파일 | 어디서 |
|---|---|---|
| `files/model/` | `yamnet_emb.tflite`, `yamnet_class_map.csv` | `cry_model/export_yamnet_tflite.py` 결과 / TF models 저장소 |
| `files/model/` | `cry_personal_ctx.json` | `cry_model/artifacts`(sim_personal.py ctx) — 울음 힌트의 맥락 모델. 없으면 맥락은 균등 |
| `files/lib/` | `libtensorflowlite.so`, `libtensorflowlite_xnnpack_delegate.so` | SDK sysroot (`atlas/yamnet_test`에서 쓴 것) |
| `files/mosquitto/` | `mosquitto`, `mosquitto.conf` | `deploy/build_mosquitto.sh` (기기에 브로커가 없음) |

## 기기에 올리기

```bash
# 개발 컨테이너
cd atlas/nuni && arc build          # → build/arm64/ipk/com.atlas.NuniFusion1.ipk

# 파이 5(root) — IPK를 /tmp에 복사한 뒤 플랫폼 패키지 매니저로 설치한다
#   (WSL 컨테이너에서는 핫스팟 대역으로 접속이 안 돼 arc install 대신 이렇게 했다)
busctl --system call com.atlas.PackageManager1 /com/atlas/PackageManager1 com.atlas.PackageManager1 \
  Install ss com.atlas.NuniFusion1 /tmp/com.atlas.NuniFusion1.ipk
SVC=/data/share/usr/atlas/services/com.atlas.NuniFusion1
mkdir -p $SVC/systemd $SVC/mosquitto
cp yamnet_emb.tflite cry_personal_ctx.json setup_on_device.sh $SVC/   # 맥락 모델은 cry_model/artifacts
cp deploy/nunifusion.service deploy/nuni-mosquitto.service $SVC/systemd/
cp mosquitto mosquitto.conf $SVC/mosquitto/        # build_mosquitto.sh 결과
# 설치만으로는 서비스가 안 뜬다. 샘플에서 겪은 함정 5가지를 이 스크립트가 처리한다
sh $SVC/setup_on_device.sh
journalctl -u nunifusion -f

# 파이 4(root) — 진동 노드. /data/nuni/vibe/에 nunivibe, deploy/nunivibe.service, deploy/setup_vibe_on_device.sh를 둔다
sh /data/nuni/vibe/setup_vibe_on_device.sh
journalctl -u nunivibe -f
```

`setup_on_device.sh`가 하는 일:
- 루트를 잠깐 읽기·쓰기로 다시 마운트한다.
- D-Bus 정책·활성화 파일을 시스템 경로에 둔다. 소유자는 `root:messagebus`, 권한은 640이고, 설치 시점 UID가 박힌 설치본 대신 알려진 내용으로 새로 쓴다.
- systemd 유닛(`nunifusion.service`, 브로커가 있으면 `nuni-mosquitto.service`)을 등록해 부팅 때 자동으로 켜지게 한다. D-Bus는 `SystemdService=`로 이 유닛에 위임한다.
- 마지막으로 `busctl ... Ping`으로 확인한다.

## 설정 (환경변수, systemd 유닛의 `Environment=`)

| 변수 | 기본 | 뜻 |
|---|---|---|
| `NUNI_MQTT_HOST` / `NUNI_MQTT_PORT` | `localhost` / `1883` | 브로커 |
| `NUNI_MQTT_USER` / `NUNI_MQTT_PASS` | (없음) | 브로커 인증 |
| `NUNI_ENABLE_RADAR` / `_CRY` / `_FUSION` / `_DBUS` | 1 | 모듈 켜고 끄기 |
| `NUNI_DBUS_BUS` | `system` | `session`이면 세션 버스(개발 PC 시험용) |
| `NUNI_YAMNET_MODEL` | 실행 파일 옆 `yamnet_emb.tflite` → 없으면 `yamnet.tflite` | 모델 |
| `NUNI_YAMNET_CLASSMAP` | 실행 파일 옆 `yamnet_class_map.csv` | 없으면 클래스 19·20 |
| `NUNI_YAMNET_THREADS` | 1 | 추론 스레드(Pi 4 TFLite 4스레드 24ms·1스레드 49ms, Pi 5 ECO.AI 8.8ms) |
| `NUNI_YAMNET_ENGINE` | `auto` | `auto`(ECO.AI → TFLite 직접) / `ecoai` / `tflite` |
| `NUNI_ECOAI_BACKEND` | `cpu` | ECO.AI 백엔드. Pi 5에서 `gpu`는 17.5ms로 cpu(8.8ms)보다 느렸다 |
| `NUNI_CRY_THRESHOLD` | 0.3 | 울음 판정 임계 |
| `NUNI_MIC_SR` | 48000 | 마이크 샘플레이트(16000이면 리샘플 안 함) |
| `NUNI_PULSE_SERVER` / `NUNI_PULSE_SOURCE` | 기본 서버·기본 입력 | 기기 PulseAudio는 시스템 모드(`/run/pulse/native`). 유닛에 USB 마이크 AB13X 이름을 넣어 두었다 |
| `NUNI_CRY_WAV` | (없음) | 마이크 대신 WAV를 반복 재생(시연·시험) |
| `NUNI_FUSION_CRY_THRESHOLD` | 0.7 | 수면 상태의 울음 확정 임계(`sleep_state.py`와 같음) |
| `NUNI_BR_MIN`, `NUNI_APNEA_HOLD_S` 등 | `radar_bridge.py`와 같음 | 레이더 게이팅 |
| `NUNI_ENABLE_CRYLOG` | 1 | 울음 기록(답 쌓기 + 힌트) 켜고 끄기 |
| `NUNI_CRY_DIR` | `/data/nuni/cry` | 울음 기록 폴더(패키지를 다시 깔아도 남는 곳) |
| `NUNI_CRY_CTX` | 실행 파일 옆 `cry_personal_ctx.json` | 맥락 모델 |
| `NUNI_CRY_RECORD` | 1 | 울음 소리(WAV)·임베딩 저장. 0이면 답만 |
| `NUNI_CRY_END_GAP` / `NUNI_CRY_MIN_WINDOWS` | 8 / 3 | 울음이 끝났다고 볼 끊김(초) / 잡음으로 버릴 울음 1초 수 |

명령줄로 덮어쓸 수 있다: `nunifusion --help`.

## D-Bus 인터페이스 `com.atlas.NuniFusion1`

| 종류 | 이름 | 형식 | 내용 |
|---|---|---|---|
| 메서드 | `Ping` | → s | `"pong:<가동 초>"` |
| 메서드 | `GetState` | → s | 마지막 `fusion/state` JSON |
| 메서드 | `GetSensors` | → s | 마지막으로 받은 `{"radar", "cry", "env"}` |
| 신호 | `StateChanged` | s | `fusion/state` JSON(발행할 때마다) |
| 신호 | `Alert` | s, s | 경보 수준, 이유 |
| 신호 | `CryDetected` | d | 울음 점수(울음이 아님 → 울음으로 바뀔 때) |
| 메서드 | `GetAlerts` | → s | 최근 경보 `[{"level","reason","ts"}]`(24시간, 최대 200개) |
| 메서드 | `GetCryLog` | → s | 울음 기록 상태: 지금 울음(+힌트) · 답 기다리는 울음(`pending`) · 최근 울음 · 답 수 · 힌트가 맞은 수 · 마지막 수유 |
| 메서드 | `AnswerCry` | s id, s action → s | "무엇으로 그쳤나요?" 답. action = feed · diaper · hold(학습) / self · sick · other · unknown(기록만) |
| 메서드 | `AnswerCryNot` | s id, s action → s | "해 봤는데 안 그쳤어요"(feed · diaper · hold). 울음 중에도 누를 수 있다 |
| 메서드 | `LogFeed` | → s | 지금 수유했다고 기록(맥락의 '마지막 수유 후 시간') |
| 신호 | `CryEpisode` | s | 울음 기록 사건 `{"event": start · hint · end · drop · answer · not · feed, …}` |

답 메서드는 `{"ok":true,"event":…}` 또는 `{"ok":false,"error":"…"}`(한국어 문구)를 돌려준다.

```bash
busctl call com.atlas.NuniFusion1 /com/atlas/NuniFusion1 com.atlas.NuniFusion1 GetState
busctl call com.atlas.NuniFusion1 /com/atlas/NuniFusion1 com.atlas.NuniFusion1 AnswerCry ss 20261020-023351 feed
arc run -d rpi -s com.atlas.NuniFusion1 -p '[GetState]'
```

앱(대시보드)은 `package-manager` 그룹으로 실행된다. 패키지 관리자가 설치 때 만드는 `/data/share/usr/share/dbus-1/system.d/com.atlas.NuniFusion1.conf`는
마지막에 읽히면서 기본으로 모두 막는다. `setup_on_device.sh`가 `/etc`에 넣는 정책의 `<policy group="package-manager">`가 이를 연다
(D-Bus는 group 정책을 default 정책보다 나중에 적용한다. 앱 UID는 설치마다 바뀌어 user로는 적을 수 없다). 2026-10-20 파이 5에서 확인.

## 울음 기록 (CryLog, `src/cry_log.cpp`)

부모가 "무엇으로 그쳤나요?"에 답할 때마다 그 울음과 함께 기기에 쌓고, 답이 10번 모이면 다음 울음부터 먼저 해 볼 1·2순위를 낸다.
흐름과 이름은 `cry_personal.py`(CryPersonal)와 같다.

1. **울음 구간 나누기**(새로 만듦): 1초 판정 점수가 0.3 이상이면 울음 1초. 울음이 시작되면 구간을 열고, 울음 판정이 8초 끊기면 닫는다.
   울음 1초가 3개보다 적으면 잡음으로 버린다(`drop`). `NUNI_CRY_END_GAP`·`NUNI_CRY_MIN_WINDOWS`로 바꾼다.
2. **힌트**: 울음이 시작되면 낸다. 맥락(시간대 · 직전 울음 후 시간 · 마지막 수유 후 시간, `cry_personal_ctx.json` 공통 로지스틱 회귀) ×
   이 아기의 이전 답 빈도(최근 답일수록 크게, 답 8개마다 절반). 답이 10번 모이기 전에는 `show`·`show2`가 거짓(대시보드가 숨김).
3. **답**: 학습하는 답(수유 · 기저귀 · 안아주기)만 이력에 들어가고, 저절로 · 아파 보임 · 기타 · 모름은 기록만 한다. 수유 답은 마지막 수유 시각도 바꾼다.
   답을 받기 전에 '맥락 × 이력' 예측(`pred_base`)과 맥락 특징(`c`)을 남겨, 나중에 소리 모델의 자가 검증과 아기별 맥락 학습에 쓴다.
4. **저장**: `/data/nuni/cry/cry_log.json`(답할 때마다 통째로 다시 씀) · `audio/<id>.wav`(울음 구간 16kHz, 시작 전 2초 포함, 최대 10분, 합계 2GB를
   넘으면 오래된 것부터 지움) · `emb/<id>.f32`(울음 1초마다 YAMNet 임베딩 1024개). 소리는 기기 밖으로 나가지 않는다. `NUNI_CRY_RECORD=0`이면 답만 쌓는다.

소리 비교(블록 특징 82개 + 아기별 판별 분석 + 자가 검증)는 아직 옮기지 않았다. 지금 힌트는 `cry_personal.py`에서 자가 검증이 소리를 끈 상태와
같다. 실제 아기 기록(Enes) 재생에서도 자가 검증은 대부분 소리를 꺼 두었다. 쌓인 소리·임베딩으로, 소리 모델을 옮긴 뒤 지금까지의 답을 그대로
다시 배울 수 있다.

MQTT로도 내보낸다(다른 구독자·나중의 소리 모델용): `cry/episode`(start · end · drop), `cry/hint`(`cry_personal.hint`와 같은 키 + `id`),
`cry/feedback`(answer · not · feed).

**Python과 비교**: `tests/cry_log_parity.py`가 같은 사건 순서(울음 시작 · 1초 · 끝 · 답 · 안 그쳤어요 · 수유 · 힌트)를 `tools/cry_log_replay`와
`CryPersonal`에 넣어 비교한다. 2026-10-20: 시드 5개 × 울음 150개, 힌트·예측 **1,605건 완전 일치**(힌트가 보이는 경우 포함). 맥락 특징은 현지
시각을 쓰므로 두 쪽 시간대를 같게 둔다(`TZ=KST-9`).

```bash
python atlas/nuni/tests/cry_log_parity.py --replay "wsl -d Ubuntu-22.04 -u jinu -- env TZ=KST-9 /home/jinu/nuni_native/cry_log_replay" \
  --replay-ctx /home/jinu/nuni_native/cry_personal_ctx.json
```

**기기 시험**(2026-10-20 파이 5): 서비스에 임시 설정(`/run/systemd/system/nunifusion.service.d/`, 재부팅하면 사라짐)으로 울음 녹음
(ESC-50 아기 울음 15초 + 정적 14초 반복)과 시험 폴더를 주고, 울음 시작 → 끝 → `AnswerCry(feed)` → 다음 울음 중 `AnswerCryNot(diaper)`를 확인했다.
수유 답 뒤 다음 울음의 1순위가 수유에서 안아주기로 바뀌었다(방금 수유한 맥락). 휴대폰 녹음 칭얼거림(donateacry)은 YAMNet 울음 점수가 낮아
(대부분 0.1 미만) 잡음으로 버려졌다.

## ToF 일어섬 감지 (`sensor/tof`)

VL53L1X를 침대 난간 높이에 수평으로 달아 침대를 가로질러 본다. 빔에 무언가 걸리면(거리가 침대 폭보다 짧아지면) 아기가 일어선 것으로 본다.

| 항목 | 내용 |
|---|---|
| 메시지 | `{"distance_mm": 320, "detected": true, "ts": 1792589447.4}` — ESP32가 0.5초쯤마다 보낸다 |
| `detected` | ESP32가 정한다: 거리 < (침대 안쪽 폭 − 여유). 기준값은 설치하고 재서 정한다 |
| 일어섬 | `detected`가 **2초** 넘게 이어지면 `attention` "아기 일어섬(난간 높이 감지)" → 진동 짧게 1번 |
| 경보 | 일어선 채 **울음**이면 "일어서서 울음", **30초** 넘게 서 있으면 "오래 서 있음(낙상 주의)" → `alert` |
| 끊김 | ToF 메시지가 **5초** 넘게 안 오면 감지 없음으로 본다 |
| 상태 | `fusion/state`에 `standing`(참/거짓), 일어서 있는 동안 수면 상태는 `awake`("일어섬 감지"). D-Bus `GetSensors`에 `tof` |

시간 값은 `fusion.py`의 `STAND_MIN_S` · `STAND_LONG_S` · `TOF_STALE_S`(C++은 `fusion.hpp`)에 있다. 하드웨어 없이 시험할 때:

```bash
for i in $(seq 1 10); do mosquitto_pub -t sensor/tof -m '{"distance_mm":320,"detected":true,"ts":0}'; sleep 0.5; done
```

## 진동 노드 `nunivibe` (파이 4)

| 받는 토픽 | 내용 | 패턴 |
|---|---|---|
| `fusion/alert` | `level: alert` | 0.8초 켬·0.4초 끔 × 3 |
| `fusion/alert` | `level: attention` | 0.3초 1번 |
| `control/vibrate` | `pattern: alert / attention / test` | 손으로 시험할 때 |

- GPIO는 PeripheralManager1로 다룬다: `EnablePeripheral` → `ActiveType=0`(ACTIVE_HIGH) → `Open(1)` → `WriteValue`. 사양서의 기본값은 ACTIVE_LOW라 1을 쓰면 핀이 LOW가 될 수 있어 먼저 확인한다(Pi 4에서는 이미 0이었고 Set은 internal error를 냈다).
- PWM이 없어 세기 조절은 못 한다. 진동 모듈은 드라이버(트랜지스터) 모듈을 거쳐 연결한다.
- 설정: `NUNI_MQTT_HOST`(기본 172.20.10.13 = 파이 5), `NUNI_VIBE_GPIO`(기본 18), `NUNI_VIBE_DRY=1`(로그만). `nunivibe --test`는 브로커 없이 시험 패턴을 한 번 낸다.
- IP는 두 파이 모두 ConnMan으로 고정했다(아이폰 핫스팟 `jinu` 프로필에만 적용, 다른 WiFi는 DHCP). 아이폰은 `.2`부터 차례로 나눠 주므로 겹치지 않게 뒤쪽 번호를 쓴다.
  ```
  connmanctl config wifi_<ID>_managed_psk --ipv4 manual 172.20.10.13 255.255.255.240 172.20.10.1 --nameservers 172.20.10.1   # 파이 5 (파이 4는 .12)
  ```
  되돌리기: `--ipv4 dhcp`. 주소를 바꾸면 `nunivibe.service`의 `NUNI_MQTT_HOST`를 고치고 `setup_vibe_on_device.sh`를 다시 실행한다.

## 기기에서 확인할 것

1. **진동 모듈 실물:** 지금은 GPIO_18 값만 확인했다. 모듈을 연결해 실제로 떨리는지, 레이더 측정에 간섭이 없는지 본다.
2. **판정 시간:** 1분마다 로그에 판정 시간이 찍힌다. Pi 5 ECO.AI 2스레드로 13ms(마이크 창 포함).
3. **D-Bus 신호 구독:** Flutter 앱이 신호를 받을 수 있는지(권한 키 포함) 확인한다.

## 원본에서 발견한 점 (동작은 원본과 똑같이 옮겼다)

비교 시험이 원본과 같아야 하므로 아래는 고치지 않았다. 고친다면 Python과 C++을 같이 고치고 `tests/`로 다시 비교한다.

1. **울음 + 호흡 미측정이 경보가 된다.** `decide()`의 '다중 신호 동시 이상'은 `br < 8`만 보고 `breath_valid`와 `presence`를 보지 않는다. 레이더 브리지는 호흡을 못 재면 `breathing_rate`를 0으로 보내므로, 그때 울음(신뢰도 0.7 이상)이 들어오면 경보로 올라간다. 재실이 없을 때(`presence: false`)도 마찬가지다(두 구현 모두 직접 확인). 위쪽 규칙처럼 `presence and br_valid and br < 8`로 바꾸는 것을 검토할 만하다.
2. **개인화 기준선이 메시지마다 갱신된다.** `Fusion._process`가 레이더뿐 아니라 울음·환경 메시지를 받을 때도 마지막 레이더 값으로 `Personalizer.update`를 부른다. 그래서 `samples`가 메시지 수만큼 늘고, 같은 레이더 값이 여러 번 반영된다.
3. **JSON이 아닌 메시지가 오면 원본은 예외로 멈춘다.** C++은 그 메시지만 버리고 로그를 한 번 남긴다.
4. **NaN 표기:** Python은 `NaN`(표준 JSON이 아님)을 내보내고, C++은 `null`을 내보낸다. Flutter `jsonDecode`는 `null`만 읽을 수 있다. C++은 Python이 보낸 `NaN`도 읽는다.

## 폴더 구조

```
atlas/nuni/
  CMakeLists.txt        빌드·설치(= IPK 구조)
  apps/nunifusion.cpp   서비스 진입점(모듈 3개 + D-Bus) — 파이 5
  apps/nunivibe.cpp     진동 노드(fusion/alert → GPIO) — 파이 4
  include/nuni/, src/   이식한 로직과 입출력(MQTT·PulseAudio·TFLite·D-Bus)
  files/                serviceinfo.json, .apptype, D-Bus 정책·활성화 파일(LG 샘플과 같은 자리)
  deploy/               systemd 유닛, 기기 마무리 스크립트, 브로커 빌드 스크립트·설정
  tools/nuni_replay.cpp 비교 시험용 재생 도구
  tools/cry_log_replay.cpp 울음 기록 재생(tests/cry_log_parity.py가 cry_personal.py와 비교)
  tests/                시나리오 생성, Python 재생, 비교, MQTT 통합 시험, 울음 기록 비교
  third_party/nlohmann/ JSON 헤더(MIT, v3.11.3)
```
