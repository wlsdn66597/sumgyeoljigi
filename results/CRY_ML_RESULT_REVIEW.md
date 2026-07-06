# Cry ML Result Review

## 확인한 파일

| 파일 | 존재 여부 | 비고 |
|---|---|---|
| `results/CRY_CLASSIFICATION_LIMITATION.md` | 존재 | 기존 한계 문서 |
| `reports/CRY_IMPROVED_EXPERIMENT_RESULTS.md` | 없음 | 이번 checkout에는 reports 산출물 없음 |
| `reports/FINAL_CRY_MODEL_DECISION.md` | 없음 | 이번 checkout에는 reports 산출물 없음 |
| `reports/YAMNET_CRY_DETECTION_RESULT.md` | 없음 | 이번 checkout에는 reports 산출물 없음 |
| `cry_model/artifacts/confusion_matrix.png` | 없음 | 이번 checkout에는 학습 artifact 없음 |

## 기존 한계 문서 기준 수치

| 항목 | 값 |
|---|---:|
| Donate-a-Cry wav 수 | 457 |
| hungry | 382개, 83.6% |
| belly_pain | 16개 |
| burping | 8개 |
| tired | 24개 |
| majority baseline Accuracy / Macro-F1 | 0.878 / 0.234 |
| 기존 aug2 Accuracy / Macro-F1 | 0.167 / 0.107 |
| balanced_aug Accuracy / Macro-F1 | 0.578 / 0.164 |
| YAMNet synthetic negative 기준 F1 | 0.874 |

## 이번 턴에서 하지 않은 일

- TensorFlow/YAMNet 기반 재학습은 수행하지 않았다.
- `reports/*`와 `cry_model/artifacts/confusion_matrix.png`가 현재 checkout에 없어 신규 ML 결과로 주장하지 않는다.

## 보고서 반영

- Donate-a-Cry reason classifier는 class imbalance와 rare class support 부족 때문에 실사용 성능으로 주장하지 않는다.
- 이번 checkout에서 보고서에 쓸 수 있는 것은 “학습 파이프라인과 한계 문서가 존재한다”는 수준이며, 혼동행렬 그림은 현재 파일이 없으므로 첨부하지 않는다.
