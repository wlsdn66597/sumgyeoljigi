# CHANGES_AND_EXPERIMENTS Review

## 문서가 요구하는 실험

1. 소프트웨어 전체 재현
2. 대시보드 시연
3. 울음 분류 학습
4. 증강 효과 비교
5. 레이더 강건성 스윕

## 신규 모듈 목록

| 파일 | 역할 |
|---|---|
| `radar_dsp.py` | synthetic radar-like raw signal에서 BPM, apnea, motion 판단 |
| `radar_sim_signal.py` | synthetic breathing/apnea/motion/noise 신호 생성 |
| `eval_radar.py` | 레이더 DSP 기본 평가 |
| `eval_radar_robustness.py` | SNR x window 강건성 스윕 |
| `eval_stream.py` | debounce/cooldown streaming 평가 |
| `voice_intent.py` | 부모 음성 인텐트 keyword 평가 |
| `cloud.py` | 비식별 이벤트 기반 sleep report stub |
| `cry_model/` | YAMNet 기반 울음 분류 학습 파이프라인 |

## 수정된 모듈 목록

| 파일 | 변경 내용 |
|---|---|
| `sim_sensors.py` | synthetic raw signal -> DSP 경로 반영 |
| `fusion.py` | 경계 호흡 교차검증, `now_fn`, streaming `update()` |
| `eval_fusion.py` | 11개 controlled scenario 및 ablation |
| `run_all.sh` | 9단계 전체 재현 파이프라인 |

## 문서가 요구하는 결과물

| 구분 | 요구 결과물 |
|---|---|
| 전체 재현 | `results/RESULTS.md` |
| 레이더 DSP | `results/RADAR_DSP_RESULT.md`, `results/artifacts/radar_bpm_*.csv` |
| 레이더 강건성 | `results/RADAR_ROBUSTNESS_RESULT.md`, `results/artifacts/radar_robustness.csv` |
| 융합 | `results/FUSION_EVALUATION_RESULT.md`, `results/artifacts/fusion_*.csv` |
| 스트리밍 | `results/STREAM_EVAL_RESULT.md`, `results/artifacts/stream_eval_summary.csv` |
| 음성 인텐트 | `results/VOICE_INTENT_RESULT.md` |
| 클라우드 리포트 | `results/SLEEP_REPORT.md`, `results/artifacts/sleep_timeline.*` |
| 대시보드 | 정상/울음/무호흡 스크린샷 3장, 이벤트->경보 지연 관측값 |
| 울음 ML | `cry_model/artifacts/confusion_matrix.png`, Macro-F1, 클래스별 성능, aug0 vs aug2 비교 |

## 자동 생성 결과물

- `results/RESULTS.md`
- `results/RADAR_DSP_RESULT.md`
- `results/RADAR_ROBUSTNESS_RESULT.md`
- `results/FUSION_EVALUATION_RESULT.md`
- `results/STREAM_EVAL_RESULT.md`
- `results/VOICE_INTENT_RESULT.md`
- `results/SLEEP_REPORT.md`
- `results/artifacts/radar_bpm_estimation.csv`
- `results/artifacts/radar_bpm_error_summary.csv`
- `results/artifacts/radar_robustness.csv`
- `results/artifacts/fusion_eval_summary.csv`
- `results/artifacts/fusion_ablation_summary.csv`
- `results/artifacts/stream_eval_summary.csv`
- `results/artifacts/sleep_timeline.csv`
- `results/artifacts/sleep_timeline.png`

## 수동 수집 결과물

- 대시보드 정상 상태 캡처
- 대시보드 울음 주입 상태 캡처
- 대시보드 무호흡 주입 상태 캡처
- 이벤트 발생 후 경보까지 걸린 시간 관측값

## 기대 기준값과 이번 실행값

| 항목 | 문서 기대값 | 이번 실제 실행값 |
|---|---:|---:|
| Radar BPM MAE | 약 0.5회/분 | 0.500 breaths/min |
| Apnea delay | 약 4.0s | 4.0s |
| Fusion full_fusion false alarm / miss | 0 / 0 | 0 / 0 |
| Fusion single_sensor false alarm / miss | 6 / 0 | 6 / 0 |
| radar_only false alarm / miss | 0 / 2 | 0 / 2 |
| audio_only false alarm / miss | 2 / 2 | 2 / 2 |
| env_only false alarm / miss | 2 / 3 | 2 / 3 |
| Streaming alerts | 4 -> 1 | 4 -> 1 |
| Streaming transient false alerts | 3 -> 0 | 3 -> 0 |
| Streaming apnea latency | 1s | 1s |
| Voice intent | curated set 100% | 100.0% (11/11) |
| Radar robustness | 15s MAE 약 1.0, 30s 이상 약 0.0 | 15s=1.000, 30s/60s=0.000 across SNR 5/10/15/20 |

## 반드시 명시해야 할 한계

- 모든 자동 검증 결과는 하드웨어 없는 synthetic signal 또는 controlled scenario 기반이다.
- 실제 60GHz 레이더 하드웨어 검증이 아니다.
- 실제 영유아 검증이나 임상 검증이 아니다.
- Streamlit 화면 캡처와 E2E 지연 관측값은 수동 수집이 필요하다.
- Voice intent 결과는 curated text set keyword 평가이며 실제 STT/마이크 소음 검증이 아니다.
- Cloud sleep report는 synthetic night timeline 기반 stub 결과이며 실제 클라우드 서비스나 실제 수면 데이터 검증이 아니다.
- Donate-a-Cry 5-class 울음 이유 분류는 class imbalance 때문에 실사용 성능으로 주장하지 않는다.
