# NUNI Software Verification Results

## 1. 실행 환경

- Python: Python 3.10.12 (`.venv`)
- Git commit: `ead3d764e79f6b90292a4fff9910b751f480128d`
- 실행 명령:
  - `python -m compileall .`
  - `bash run_all.sh`
- 실행 일시: 2026-07-01 KST

## 2. 실행 요약

| 영역 | 실행 항목 | 결과 | 산출물 |
|---|---|---|---|
| 기본 검증 | compileall | 성공 | `results/logs/compileall_final.log` |
| 레이더 DSP | synthetic BPM estimation | 성공 | `results/RADAR_DSP_RESULT.md` |
| 무호흡 감지 | synthetic apnea scenario | 성공 | `results/RADAR_DSP_RESULT.md` |
| 움직임 감지 | synthetic motion scenario | 성공 | `results/RADAR_DSP_RESULT.md` |
| 멀티모달 융합 | controlled scenario evaluation | 성공 | `results/FUSION_EVALUATION_RESULT.md` |
| 대시보드 | import/startup smoke test | 성공 | `results/logs/dashboard_import_smoke.log` |
| 울음 이유 분류 | 기존 ML 실험 결과 요약 | 제한적 | `reports/FINAL_CRY_MODEL_DECISION.md`, `results/CRY_CLASSIFICATION_LIMITATION.md` |

## 3. 핵심 정량 결과

| 항목 | 결과 | 조건 |
|---|---:|---|
| BPM MAE | 0.500 breaths/min | synthetic raw signal, 30s, fs=50Hz, SNR 15dB, BPM 20/25/30/35/40/45 |
| Apnea detection | True, delay 4.0s | synthetic apnea, start 6s, duration 5s, recent 4s RMS |
| Motion detection | True | synthetic high-frequency motion burst, recent 2s energy |
| Full fusion false alarm / miss | 0 / 0 | controlled scenarios, 11 cases |
| Single-sensor false alarm / miss | 6 / 0 | controlled scenarios, 11 cases |
| Radar-only false alarm / miss | 0 / 2 | controlled scenarios, 11 cases |
| Audio-only false alarm / miss | 2 / 2 | controlled scenarios, 11 cases |
| Env-only false alarm / miss | 2 / 3 | controlled scenarios, 11 cases |

## 4. 보고서에서 주장 가능한 것

- 현재 checkout에서 `run_all.sh` 하나로 compileall, synthetic radar DSP evaluation, fusion evaluation, dashboard/import smoke test를 재현할 수 있다.
- Synthetic raw radar-like signal 기준 FFT 호흡수 추정 파이프라인을 구현했다.
- Synthetic signal 조건에서 BPM MAE 0.500 breaths/min을 확인했다.
- Synthetic apnea/motion smoke test가 동작했다.
- Controlled scenario에서 full fusion은 single-sensor baseline보다 false alarm을 줄였다.
- Streamlit 및 주요 앱 모듈 import smoke test가 성공했다.
- Donate-a-Cry 기반 5-class 울음 이유 분류는 데이터 불균형으로 성능 한계가 큼을 확인했다.

## 5. 보고서에서 주장하면 안 되는 것

- 실제 60GHz 레이더 하드웨어 성능을 검증했다.
- 실제 영유아 무호흡 감지 성능을 검증했다.
- 실환경 마이크/생활 소음에서 YAMNet precision을 보장한다.
- Donate-a-Cry reason classifier가 실사용 가능한 이유 분류 성능을 달성했다.
- Controlled scenario false alarm/miss 0이 실제 환경에서도 그대로 유지된다.

## 6. 보고서용 문장

1. 본 프로젝트는 하드웨어 없이도 synthetic radar signal, 멀티모달 융합, 대시보드 import smoke test를 `run_all.sh`로 재현할 수 있도록 구성했다.
2. 레이더 DSP 평가는 실제 60GHz 센서가 아니라 synthetic raw radar-like signal을 입력으로 수행했다.
3. FFT 기반 호흡수 추정은 30초, 50Hz, SNR 15dB synthetic 조건에서 MAE 0.500 breaths/min을 기록했다.
4. Synthetic apnea scenario에서는 apnea 시작 후 4.0초 지연으로 감지가 확인됐다.
5. Synthetic motion burst scenario에서는 최근 2초 고주파 에너지 기준 motion 감지가 확인됐다.
6. Controlled scenario 11개에서 full fusion만 false alarm 0건, miss 0건을 기록했다(유일하게 양쪽 0).
7. Single-sensor baseline은 false alarm 6건, radar-only는 경계 호흡을 놓쳐 miss 2건을 보여, 융합이 오탐과 미탐을 동시에 줄였음을 확인했다.
8. Donate-a-Cry 기반 5-class 울음 이유 분류는 class imbalance 때문에 실사용 성능으로 주장하지 않고 실험적 기능으로 정리했다.

## 7. 발표용 1분 스크립트

이번 검증에서는 NUNI의 소프트웨어 파이프라인을 현재 checkout에서 다시 재현 가능하게 정리했습니다. 먼저 실제 레이더 하드웨어 없이 synthetic raw radar-like signal을 만들고, FFT 기반으로 호흡수를 추정하는 DSP 경로를 구현했습니다. 30초, 50Hz, SNR 15dB 조건에서 BPM 추정 MAE는 0.500회/분이었고, synthetic apnea와 motion smoke test도 동작했습니다. 다음으로 레이더, 음향, 환경 신호를 함께 보는 융합 평가를 controlled scenario 8개에서 수행했습니다. 단일 센서 baseline은 false alarm 5건이었지만 full fusion은 false alarm 0건, miss 0건을 기록했습니다. 다만 이 결과는 synthetic 및 controlled scenario 기준이며 실제 센서나 실제 영유아 환경 검증은 아닙니다. 울음 이유 분류는 Donate-a-Cry 데이터 불균형 때문에 성능 한계가 커서 핵심 성과가 아니라 실험적 기능으로 제시하는 것이 정직합니다.
