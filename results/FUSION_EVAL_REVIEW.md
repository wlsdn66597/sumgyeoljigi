# Fusion Evaluation Review

실행 명령: `source .venv/bin/activate && python eval_fusion.py`

## 결과

| 항목 | 실제 실행값 |
|---|---:|
| controlled scenario 개수 | 11 |
| single_sensor false alarm / miss | 6 / 0 |
| radar_only false alarm / miss | 0 / 2 |
| audio_only false alarm / miss | 2 / 2 |
| env_only false alarm / miss | 2 / 3 |
| full_fusion false alarm / miss | 0 / 0 |

## 해석

- 이번 11개 controlled scenario에서는 full_fusion만 false alarm 0, miss 0을 기록했다.
- radar_only는 명확한 apnea는 잡지만, cry 또는 bad environment와 결합될 때 위험해지는 경계 호흡 시나리오를 놓쳐 miss 2가 발생했다.
- 따라서 이번 synthetic scenario에서는 radar가 핵심 기여 모달이지만, full fusion은 경계 신호 교차검증으로 radar_only의 miss를 줄였다.
- 결과는 controlled scenario 기반이며 실제 센서/영유아/임상 검증이 아니다.
