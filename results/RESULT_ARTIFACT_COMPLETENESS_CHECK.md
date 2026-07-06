# Result Artifact Completeness Check

| 요구 결과물 | 존재 여부 | 실제 경로 | 보고서 사용 가능 여부 | 비고 |
|---|---|---|---|---|
| `results/RESULTS.md` | 존재 | `results/RESULTS.md` | 가능 | run_all 생성 |
| `results/RADAR_DSP_RESULT.md` | 존재 | `results/RADAR_DSP_RESULT.md` | 가능 | synthetic signal 한계 명시 |
| `results/RADAR_ROBUSTNESS_RESULT.md` | 존재 | `results/RADAR_ROBUSTNESS_RESULT.md` | 가능 | SNR별 차이는 이번 실행에서 없음 |
| `results/FUSION_EVALUATION_RESULT.md` | 존재 | `results/FUSION_EVALUATION_RESULT.md` | 가능 | controlled scenario |
| `results/STREAM_EVAL_RESULT.md` | 존재 | `results/STREAM_EVAL_RESULT.md` | 가능 | synthetic timeline |
| `results/VOICE_INTENT_RESULT.md` | 존재 | `results/VOICE_INTENT_RESULT.md` | 가능 | curated text keyword 평가 |
| `results/SLEEP_REPORT.md` | 존재 | `results/SLEEP_REPORT.md` | 가능 | synthetic night stub |
| `results/artifacts/radar_bpm_estimation.csv` | 존재 | `results/artifacts/radar_bpm_estimation.csv` | 가능 | BPM 상세 |
| `results/artifacts/radar_bpm_error_summary.csv` | 존재 | `results/artifacts/radar_bpm_error_summary.csv` | 가능 | MAE summary |
| `results/artifacts/radar_robustness.csv` | 존재 | `results/artifacts/radar_robustness.csv` | 가능 | SNR x window |
| `results/artifacts/fusion_eval_summary.csv` | 존재 | `results/artifacts/fusion_eval_summary.csv` | 가능 | scenario별 결과 |
| `results/artifacts/fusion_ablation_summary.csv` | 존재 | `results/artifacts/fusion_ablation_summary.csv` | 가능 | policy별 오탐/미탐 |
| `results/artifacts/stream_eval_summary.csv` | 존재 | `results/artifacts/stream_eval_summary.csv` | 가능 | debounce/cooldown |
| `results/artifacts/sleep_timeline.csv` | 존재 | `results/artifacts/sleep_timeline.csv` | 가능 | synthetic timeline |
| `results/artifacts/sleep_timeline.png` | 존재 | `results/artifacts/sleep_timeline.png` | 가능 | synthetic timeline plot |

## 추가 누락

| 항목 | 상태 | 비고 |
|---|---|---|
| 대시보드 스크린샷 3종 | 누락 | 수동 캡처 필요 |
| E2E 이벤트->경보 관측값 | 누락 | 수동 기록 필요 |
| `cry_model/artifacts/confusion_matrix.png` | 누락 | 이번 checkout에는 ML artifact 없음 |
