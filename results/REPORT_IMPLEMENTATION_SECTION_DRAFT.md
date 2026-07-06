# 구현기능 및 검증 결과

## 1. 전체 소프트웨어 파이프라인

NUNI는 MQTT/inproc 기반 모듈 구조로 가상 센서, radar DSP, 멀티모달 융합, 알림, 대시보드를 연결한다. `run_all.sh`는 compileall, radar DSP, radar robustness, fusion, streaming, voice intent, cloud sleep report, dashboard import smoke, summary 생성을 9단계로 실행했다. 이번 실행은 하드웨어 없는 synthetic signal 및 controlled scenario 기반이다.

## 2. 레이더 호흡 신호 처리

`radar_sim_signal.py`가 생성한 synthetic raw radar-like signal을 `radar_dsp.py`가 FFT 기반으로 처리해 BPM을 추정하고, 최근 RMS 저하로 apnea smoke test를 판단하며, 고주파 에너지 변화로 motion burst를 감지한다.

| true BPM | predicted BPM | abs error |
|---:|---:|---:|
| 20 | 20.0 | 0.0 |
| 25 | 26.0 | 1.0 |
| 30 | 30.0 | 0.0 |
| 35 | 34.0 | 1.0 |
| 40 | 40.0 | 0.0 |
| 45 | 44.0 | 1.0 |

| 항목 | 결과 |
|---|---:|
| BPM MAE | 0.500 breaths/min |
| apnea detected | True |
| apnea detection delay | 4.0s |
| motion detected | True |

이 결과는 실제 60GHz radar hardware 성능이나 실제 영유아 apnea 검증이 아니다.

## 3. 레이더 강건성 평가

SNR 5/10/15/20dB와 window 15/30/60s 조합을 synthetic signal로 sweep했다.

| window(s) \ SNR(dB) | 5 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|
| 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| 30 | 0.000 | 0.000 | 0.000 | 0.000 |
| 60 | 0.000 | 0.000 | 0.000 | 0.000 |

이번 실행에서는 window 길이가 핵심 차이를 만들었고, SNR 조건 간 MAE 차이는 관측되지 않았다.

## 4. 멀티모달 융합 판단

11개 controlled scenario에서 single baseline과 modality ablation을 비교했다.

| policy | false alarm | miss |
|---|---:|---:|
| single_sensor | 6 | 0 |
| radar_only | 0 | 2 |
| audio_only | 2 | 2 |
| env_only | 2 | 3 |
| full_fusion | 0 | 0 |

현재 synthetic scenario에서는 radar가 핵심 기여 모달이지만, radar_only는 경계 호흡 조건을 2건 놓쳤다. full fusion은 울음/환경과의 교차검증으로 이 miss를 줄였고, single-sensor baseline 대비 false alarm도 줄였다.

## 5. 디바운스/쿨다운 기반 알림 안정화

100초 synthetic timeline에서 sustained apnea와 1초 transient blip을 주입해 streaming evaluation을 실행했다.

| config | total alerts | apnea detect latency (s) | transient false alerts |
|---|---:|---:|---:|
| raw (debounce=1, cooldown=0) | 4 | 0 | 3 |
| tuned (debounce=2, cooldown=10) | 1 | 1 | 0 |

디바운스/쿨다운 적용으로 일시적 오경보는 3건에서 0건으로 줄었고, sustained apnea는 1초 지연으로 감지됐다.

## 6. 부모 음성 인텐트 및 클라우드 리포트

`voice_intent.py`는 5개 intent, 11개 curated Korean phrase에 대해 keyword matching 평가를 수행했고 100.0%(11/11)를 기록했다. 이는 실제 STT/마이크/소음 환경 검증이 아니다.

`cloud.py`는 synthetic night timeline으로 비식별 이벤트 기반 아침 수면 리포트를 생성했다. 실행 결과 울음 4회, 무호흡 의심 1회, 평균 호흡 39.9회/분, 최장 안정 수면 298분이 기록됐다. 이는 실제 클라우드 서비스나 실제 영유아 수면 데이터가 아니다.

## 7. 울음 분류 실험과 한계

이번 checkout에는 `results/CRY_CLASSIFICATION_LIMITATION.md`가 존재하지만 `reports/*` 상세 ML 산출물과 `cry_model/artifacts/confusion_matrix.png`는 없다. 기존 한계 문서 기준 Donate-a-Cry는 457개 wav이며 `hungry`가 382개(83.6%)로 불균형하다.

| 실험 | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority baseline | 0.878 | 0.234 |
| 기존 aug2 | 0.167 | 0.107 |
| balanced_aug | 0.578 | 0.164 |

class-balanced augmentation은 기존 aug2보다 개선됐지만 majority baseline보다 낮다. 따라서 울음 이유 분류는 실사용 성능으로 주장하지 않고, 데이터 한계가 확인된 실험적 기능으로 정리한다.

## 8. 한계 및 2차 데모 계획

이번 결과는 실제 60GHz 레이더 통합, 실제 마이크 소음 환경, 실제 영유아 대상, 임상 안전성 검증을 포함하지 않는다. 2차 데모에서는 실제 레이더/환경센서 통합, 실제 마이크 입력, 대시보드 정상/울음/무호흡 캡처, 이벤트 발생 후 경보까지의 E2E 지연 관측값 기록이 필요하다.
