# Proactive Environment Control

- 통제 시나리오 8개, 동작 일치율 100.0%
- 입력: 환경 센서값 + 수면 상태. 울음 이유 분류에 의존하지 않음.

| 시나리오 | 기대 동작 | 산출 동작 | 결과 |
|---|---|---|---|
| comfort | - | - | O |
| co2_high | ventilate | ventilate | O |
| dry | humidify | humidify | O |
| hot | cool | cool | O |
| cold | heat | heat | O |
| bright_sleep | dim_light | dim_light | O |
| bright_awake | - | - | O |
| multi | cool|dim_light|humidify|ventilate | cool|dim_light|humidify|ventilate | O |

## 한계

- 규칙 기반 권고이며 실제 가전 구동/실환경 검증은 2차 하드웨어 단계.
