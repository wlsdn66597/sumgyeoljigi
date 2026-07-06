# Document Consistency Check

## README vs CHANGES vs 실제 실행값

| 항목 | README.md | CHANGES_AND_EXPERIMENTS.md | 이번 실제 실행값 | 판정 |
|---|---:|---:|---:|---|
| radar BPM MAE | 0.500 | 약 0.5 | 0.500 | 일치 |
| apnea delay | 4.0s | 약 4.0s | 4.0s | 일치 |
| full_fusion false alarm / miss | 0 / 0 | 0 / 0 | 0 / 0 | 일치 |
| single_sensor false alarm / miss | 6 / 0 | 6 / 0 | 6 / 0 | 일치 |
| radar_only false alarm / miss | 0 / 2 | 0 / 2 | 0 / 2 | 일치 |
| audio_only false alarm / miss | 2 / 2 | 2 / 2 | 2 / 2 | 일치 |
| env_only false alarm / miss | 2 / 3 | 2 / 3 | 2 / 3 | 일치 |
| stream alerts | `4->1`, latency 1s | `4->1`, latency 1s | `4->1`, latency 1s | 일치 |
| voice intent | 결과 파일 언급 | curated 세트 100% | 100.0% (11/11) | 일치 |
| cry classifier 한계 | 실사용 성능 주장 불가 | class imbalance 한계 | 기존 한계 문서와 일치 | 일치 |
| radar robustness | 15s 약 1.0, 30s 이상 약 0.0 | 15s 약 1.0, 30s 이상 약 0.0, SNR 영향 미미 | 15s=1.000, 30s/60s=0.000 across all SNR | 일치 |

## 불일치 또는 수정 제안

- README와 CHANGES의 핵심 정량 수치는 이번 실행값과 일치한다.
- 단, 자동 생성된 `results/RADAR_ROBUSTNESS_RESULT.md`의 해석 문장 중 “SNR이 낮을수록 MAE가 커진다”는 이번 실제 결과와 맞지 않는다. 실제 실행에서는 SNR 5/10/15/20dB 간 MAE 차이가 없었다.
- README/CHANGES 자동 수정은 하지 않았다.
