# Radar DSP Result Review

실행 명령: `source .venv/bin/activate && python eval_radar.py`

## 결과

| 항목 | 실제 실행값 | 산출물 |
|---|---:|---|
| BPM MAE | 0.500 breaths/min | `results/artifacts/radar_bpm_error_summary.csv` |
| apnea detected | True | `results/RADAR_DSP_RESULT.md` |
| apnea start | 6s | `results/RADAR_DSP_RESULT.md` |
| apnea detected at | 10.0s | `results/RADAR_DSP_RESULT.md` |
| apnea detection delay | 4.0s | `results/RADAR_DSP_RESULT.md` |
| motion detected | True | `results/RADAR_DSP_RESULT.md` |

## BPM 상세

| true BPM | predicted BPM | abs error |
|---:|---:|---:|
| 20 | 20.0 | 0.0 |
| 25 | 26.0 | 1.0 |
| 30 | 30.0 | 0.0 |
| 35 | 34.0 | 1.0 |
| 40 | 40.0 | 0.0 |
| 45 | 44.0 | 1.0 |

## 한계

- synthetic raw radar-like signal만 사용했다.
- 실제 60GHz 레이더 하드웨어 검증이 아니다.
- 실제 영유아 무호흡 또는 임상 검증이 아니다.
