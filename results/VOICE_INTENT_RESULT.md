# Voice Intent Result

- 제한 명령 집합 5종, 평가 문장 11개
- 인식 정확도: 100.0%
- 방식: 키워드 매칭(경량). 인식 실패(unknown) 시 대시보드 입력으로 대체.

| 문장 | 정답 | 예측 | 결과 |
|---|---|---|---|
| 아기 지금 상태 어때? | baby_status_query | baby_status_query | O |
| 아기 잘 자고 있어? | baby_status_query | baby_status_query | O |
| 호흡 괜찮아? | baby_status_query | baby_status_query | O |
| 방이 너무 건조한가? | humidity_control | humidity_control | O |
| 가습기 켜줘 | humidity_control | humidity_control | O |
| 좀 더운 것 같아 | temperature_control | temperature_control | O |
| 에어컨 틀어줘 | temperature_control | temperature_control | O |
| 불 좀 꺼줘 | light_control | light_control | O |
| 너무 어두워 | light_control | light_control | O |
| 공기가 답답해 | air_control | air_control | O |
| 환기 좀 하자 | air_control | air_control | O |

## 한계

- 키워드 기반이라 표현이 크게 벗어나면 unknown 처리된다. 실물은 Vosk STT + 동일 규칙.
