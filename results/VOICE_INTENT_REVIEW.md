# Voice Intent Review

실행 명령:

- `python voice_intent.py --help`
- `python voice_intent.py`

## 결과

| 항목 | 실제 실행값 |
|---|---:|
| 지원 intent 종류 | 5 |
| curated phrase 개수 | 11 |
| correct | 11 |
| accuracy | 100.0% |

## 지원 인텐트

- `baby_status_query`
- `humidity_control`
- `temperature_control`
- `light_control`
- `air_control`

## 주의

- `voice_intent.py`는 argparse help를 제공하지 않아 `--help` 실행도 동일 평가를 수행했다.
- 결과는 curated text set keyword matching 평가다.
- 실제 음성 마이크, STT, 잡음 환경 검증이 아니다.
