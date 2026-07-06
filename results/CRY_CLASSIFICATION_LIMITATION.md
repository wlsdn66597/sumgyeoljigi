# Cry Classification Limitation

## 데이터 분포

- Donate-a-Cry 총 wav 수: 457개
- `hungry`: 382개, 83.6%
- rare class:
  - `belly_pain`: 16개
  - `burping`: 8개
  - `tired`: 24개

## split 한계

기본 group split에서 `belly_pain`은 train 16개, test 0개로 배치되어 해당 class를 평가할 수 없었다. 이는 group leakage를 막는 분할을 유지할 때 rare class support가 쉽게 사라지는 구조적 문제다.

## baseline과 모델 성능

| 모델/실험 | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority baseline | 0.878 | 0.234 |
| 기존 aug2 | 0.167 | 0.107 |
| balanced_aug | 0.578 | 0.164 |

balanced augmentation은 기존 aug2보다 개선됐지만 majority baseline보다 낮다. 따라서 현재 5-class reason classifier는 실사용 성능으로 주장하면 안 된다.

## YAMNet cry score

YAMNet Baby cry score는 synthetic silence/white noise/tone negative 기준 threshold 0.1에서 Precision 1.000, Recall 0.777, F1 0.874를 보였다. 그러나 실제 non-cry audio가 없으므로 실환경 울음 감지 성능을 보장한다고 주장할 수 없다.

## 보고서에 넣을 수 있는 문장

- Donate-a-Cry 기반 5-class 울음 이유 분류는 데이터 불균형과 rare class 부족으로 성능 한계가 컸다.
- class-balanced augmentation은 기존 aug2 대비 성능을 개선했지만 majority baseline보다 낮아 실사용 성능으로 보기 어렵다.
- YAMNet Baby cry score는 울음 여부 감지 이벤트 생성을 위한 smoke test로 사용할 수 있으나, 실제 생활 소음 기반 검증은 필요하다.

## 피해야 할 표현

- 울음 이유 분류 성능 검증 완료
- 실사용 가능한 울음 분류 정확도 달성
- `belly_pain` 분류 성능 확인
- 실제 생활 소음에서도 YAMNet precision이 높다고 보장
