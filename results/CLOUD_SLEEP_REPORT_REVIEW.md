# Cloud Sleep Report Review

실행 명령: `source .venv/bin/activate && python cloud.py`

## 결과

| 항목 | 실제 실행값 |
|---|---:|
| monitored time | 8.0시간 |
| cry events | 4회 |
| cry minutes | 10분 |
| apnea suspicious events | 1회 |
| avg BPM | 39.9 회/분 |
| BPM range | 34.4~46.0 |
| longest calm sleep | 298분 |
| CO2 range | 549~897 ppm |
| temp range | 20.9~23.1 ℃ |

## 산출물

- `results/SLEEP_REPORT.md`
- `results/artifacts/sleep_timeline.csv`
- `results/artifacts/sleep_timeline.png`

## 한계

- synthetic night timeline 기반 cloud stub이다.
- 실제 클라우드 서비스, 실제 영유아 수면, 실제 임상 검증이 아니다.
