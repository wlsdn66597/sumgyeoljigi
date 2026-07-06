# File Existence Check

로그: `results/logs/file_existence_check.log`

| 파일 | 존재 여부 | 역할 | 비고 |
|---|---|---|---|
| `radar_dsp.py` | 존재 | radar-like raw signal DSP | 실행 가능 |
| `radar_sim_signal.py` | 존재 | synthetic radar-like signal 생성 | 실행 가능 |
| `eval_radar.py` | 존재 | 레이더 DSP 기본 평가 | 실행 완료 |
| `eval_radar_robustness.py` | 존재 | SNR x window 강건성 스윕 | 실행 완료 |
| `eval_stream.py` | 존재 | debounce/cooldown streaming 평가 | 실행 완료 |
| `voice_intent.py` | 존재 | curated text intent 평가 | 실행 완료 |
| `cloud.py` | 존재 | synthetic sleep report stub | 실행 완료 |
| `sim_sensors.py` | 존재 | 가상 레이더/환경 센서 | 실행 가능 |
| `fusion.py` | 존재 | 멀티모달 융합 판단 | 실행 가능 |
| `eval_fusion.py` | 존재 | controlled scenario 융합 평가 | 실행 완료 |
| `run_all.sh` | 존재 | 9단계 전체 재현 파이프라인 | 실행 완료 |
| `app.py` | 존재 | Streamlit 대시보드 | import smoke 성공, 서버 socket은 환경 권한으로 실패 |
| `cry_model/train.py` | 존재 | 울음 분류 학습 | 이번 턴 재학습 미실행 |
| `cry_model/evaluate.py` | 존재 | 울음 분류 평가 | 이번 턴 재평가 미실행 |
