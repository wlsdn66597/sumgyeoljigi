# Sleep/Wake State Classification

- 통제 시나리오 7개, 정확도 100.0%
- 입력: 레이더 움직임 + 울음 감지(is_crying). 울음 이유 분류에 의존하지 않음.

| 시나리오 | 기대 | 예측 | 결과 |
|---|---|---|---|
| calm_sleep | calm_sleep | calm_sleep | O |
| low_move | calm_sleep | calm_sleep | O |
| restless | restless | restless | O |
| restless_high | restless | restless | O |
| awake_move | awake | awake | O |
| awake_cry | awake | awake | O |
| absent | unknown | unknown | O |

## 한계

- 합성/통제 시나리오 기준이며 실제 영유아 수면 검증이 아니다.
