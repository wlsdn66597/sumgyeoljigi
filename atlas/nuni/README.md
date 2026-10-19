# nunifusion — 숨결지기 atlas 네이티브 서비스 (C++)

파이에서 Python으로 돌던 실시간 경로(레이더 변환 · 울음 감지 · 융합 판단)를 atlas용 C++로 옮긴 것이다.
서비스 이름은 `com.atlas.NuniFusion1`이고, 프로세스 하나에서 세 모듈이 돈다.

```
 MR60BHA2(ESPHome) ─ mr60bha2/* ─┐                         ┌─ sensor/radar ─┐
 ESP32(SCD41 등) ─── sensor/env ─┤      MQTT 브로커         │                 │
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

옮기지 않은 것: 대시보드(`app.py` → Flutter, 담당 A), 기록·리포트(`recorder.py`, `cloud.py` 등), 울음 힌트(`cry_personal.py`, 2단계), SCD41 읽기(ESP32로 옮기거나 2단계).

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

기기(atlas Pi 4)에서는 아직 돌려 보지 않았다. 아래 '기기에서 확인할 것'을 먼저 본다.

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
- IPK는 LG 샘플처럼 `arc build`로 만든다. `CMakeLists.txt`의 install 규칙과 `files/` 구조를 샘플(`lifecycleservice`)과 같게 맞췄다.

일반 리눅스 PC에서도 같은 명령으로 빌드된다. 없는 라이브러리 기능만 빠진다(예: TFLite가 없으면 울음 감지 없이 레이더·융합만).

### 기기에 같이 넣을 파일 (저장소에는 없음)

| 넣을 곳 | 파일 | 어디서 |
|---|---|---|
| `files/model/` | `yamnet_emb.tflite`, `yamnet_class_map.csv` | `cry_model/export_yamnet_tflite.py` 결과 / TF models 저장소 |
| `files/lib/` | `libtensorflowlite.so`, `libtensorflowlite_xnnpack_delegate.so` | SDK sysroot (`atlas/yamnet_test`에서 쓴 것) |
| `files/mosquitto/` | `mosquitto`, `mosquitto.conf` | `deploy/build_mosquitto.sh` (기기에 브로커가 없음) |

## 기기에 올리기

```bash
# 개발 컨테이너
cd atlas/nuni && arc build
arc install ./build/arm64/ipk/com.atlas.NuniFusion1.ipk -d rpi

# 기기(root) — 설치만으로는 서비스가 안 뜬다. 샘플에서 겪은 함정 5가지를 이 스크립트가 처리한다
sh /data/share/usr/atlas/services/com.atlas.NuniFusion1/setup_on_device.sh
journalctl -u nunifusion -f
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
| `NUNI_YAMNET_THREADS` | 1 | XNNPACK 스레드(기기에서 4스레드 24ms, 1스레드 49ms) |
| `NUNI_CRY_THRESHOLD` | 0.3 | 울음 판정 임계 |
| `NUNI_MIC_SR` | 48000 | 마이크 샘플레이트(16000이면 리샘플 안 함) |
| `NUNI_PULSE_SERVER` / `NUNI_PULSE_SOURCE` | 기본 서버·기본 입력 | 기기 PulseAudio는 시스템 모드(`/var/run/pulse/native`) |
| `NUNI_CRY_WAV` | (없음) | 마이크 대신 WAV를 반복 재생(시연·시험) |
| `NUNI_FUSION_CRY_THRESHOLD` | 0.7 | 수면 상태의 울음 확정 임계(`sleep_state.py`와 같음) |
| `NUNI_BR_MIN`, `NUNI_APNEA_HOLD_S` 등 | `radar_bridge.py`와 같음 | 레이더 게이팅 |

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

```bash
busctl call com.atlas.NuniFusion1 /com/atlas/NuniFusion1 com.atlas.NuniFusion1 GetState
arc run -d rpi -s com.atlas.NuniFusion1 -p '[GetState]'
```

## 기기에서 확인할 것

1. **브로커:** `deploy/build_mosquitto.sh`를 SDK 환경에서 빌드한다. x86에서는 빌드·동작을 확인했다.
2. **마이크:** USB 마이크를 꽂고 `pactl list short sources`로 입력 장치 이름을 확인한다(9-28에는 입력 장치 0개였음). 필요하면 `NUNI_PULSE_SOURCE`를 지정한다.
3. **샘플레이트:** 마이크가 48kHz가 아니면 `NUNI_MIC_SR`을 맞춘다. PulseAudio가 알아서 변환하므로 16000으로 받아도 된다.
4. **판정 시간:** 기기에서 `NUNI_YAMNET_THREADS`를 정한다. 1분마다 로그에 판정 시간이 찍힌다.
5. **D-Bus 신호 구독:** Flutter 앱이 신호를 받을 수 있는지(권한 키 포함) 확인한다.

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
  apps/nunifusion.cpp   서비스 진입점(모듈 3개 + D-Bus)
  include/nuni/, src/   이식한 로직과 입출력(MQTT·PulseAudio·TFLite·D-Bus)
  files/                serviceinfo.json, .apptype, D-Bus 정책·활성화 파일(LG 샘플과 같은 자리)
  deploy/               systemd 유닛, 기기 마무리 스크립트, 브로커 빌드 스크립트·설정
  tools/nuni_replay.cpp 비교 시험용 재생 도구
  tests/                시나리오 생성, Python 재생, 비교, MQTT 통합 시험
  third_party/nlohmann/ JSON 헤더(MIT, v3.11.3)
```
