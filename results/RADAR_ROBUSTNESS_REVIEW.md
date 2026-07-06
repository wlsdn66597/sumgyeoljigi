# Radar Robustness Review

실행 명령: `source .venv/bin/activate && python eval_radar_robustness.py`

## SNR x window 결과

| window(s) \ SNR(dB) | 5 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|
| 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| 30 | 0.000 | 0.000 | 0.000 | 0.000 |
| 60 | 0.000 | 0.000 | 0.000 | 0.000 |

## 산출물

- `results/RADAR_ROBUSTNESS_RESULT.md`
- `results/artifacts/radar_robustness.csv`
- `results/logs/eval_radar_robustness_latest.log`

## 해석

- 이번 실제 실행에서는 SNR 5/10/15/20dB 간 MAE 차이가 나타나지 않았다.
- window 15s에서는 모든 SNR에서 MAE 1.000, window 30s/60s에서는 모든 SNR에서 MAE 0.000이었다.
- 따라서 "SNR이 낮을수록 MAE가 커진다"는 일반 설명은 이번 실행값으로는 뒷받침되지 않는다.

## 한계

- synthetic raw signal 전용 강건성 스윕이다.
- 실제 레이더 하드웨어, 실제 거리/자세/반사 환경, 실제 영유아 움직임을 검증하지 않았다.
