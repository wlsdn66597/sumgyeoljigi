# 울음 분류 (YAMNet 전이학습)

**2단계 설계**
1. **울음 감지(is_crying)**: YAMNet에 내장된 AudioSet `Baby cry, infant cry` 점수를 그대로 쓴다. 별도 학습은 필요 없고, 실시간 경로는 `yamnet_cry.py`(tflite)다.
2. **울음 이유 3클래스 분류(실험적)**: YAMNet 세그먼트 임베딩(1024-d)을 커스텀 헤드에 넣는다.
   - `hungry` 배고픔
   - `sleepy` 졸림·피곤
   - `discomfort` 신체 불편: 복통, 트림 필요(가스), 일반 불편, 덥거나 추움

   `lonely`/`scared`(정서), `unknown`, 웃음·소음 같은 비울음은 학습에서 뺀다. 매핑은 `config.LABEL_MAP` 한 곳에서 관리한다.

## 설치

```bash
pip install -r requirements-ml.txt      # Python 3.9~3.11
```
Donate-a-Cry 원본 버킷(.3gp/.caf)을 쓰려면 ffmpeg가 필요하다. PATH에 두거나 `NUNI_FFMPEG`로 경로를 지정한다.

## 데이터

원본 데이터는 용량과 라이선스 때문에 저장소 밖에 둔다. `NUNI_CRY_DATA` 기본값은 `cry_model/data`다.

```bash
export NUNI_CRY_DATA=D:/datasets/cry
python download_data.py all     # Donate-a-Cry · ESC-50 · OpenSLR 28
python prepare_data.py          # → manifest.csv (출처·라벨·아기 group·중복 제거)
python check_env.py
```

| 출처 | 용도 | 라이선스 |
|---|---|---|
| Donate-a-Cry 정제본 457개 + 원본 버킷(정제본에 없는 녹음) | 학습 | ODbL |
| `extra/<출처명>/…/<라벨>/` (Kaggle Cry Sense, Infant cry Dataset 등, 직접 받아서 풀기) | 학습 | 각 데이터 약관 |
| ESC-50 (crying_baby 제외) + OpenSLR 28 등방성 소음 | 소음 증강 | CC BY-NC / Apache 2.0 |
| OpenSLR 28 실측 RIR + 소·중형 방 시뮬레이션 RIR | 잔향 증강 | Apache 2.0 |

원본 버킷은 정제 과정에서 빠진 녹음이라 무음이나 잡음이 섞여 있다. 그래서 YAMNet 울음 점수가 `CRY_MIN`보다 낮은 세그먼트는 버리고, 울음 세그먼트가 하나도 없는 클립은 통째로 제외한다.

## 학습 · 평가

```bash
python train.py                  # 5겹 교차검증 → 최종 모델 학습 → artifacts/
                                 #   클립 50개 이상인 클래스만 자동 선택(--min-clips)
python train.py --classes hungry,discomfort   # 클래스 직접 지정
python train.py --no-pitch       # 피치 증강 제거 비교 (울음 음높이가 이유 단서일 수 있음)
python train.py --holdout-source crysense   # 특정 출처를 학습에서 빼고 일반화 확인
python evaluate.py <라벨폴더>      # 학습에 안 쓴 녹음(예: 직접 녹음)으로 평가
python infer.py a.wav b.wav      # 파일 추론 (인자 없으면 마이크)
```

- **입력 단위 일치**: 학습과 추론 모두 YAMNet 연속 2프레임(≈1.5초) 평균 세그먼트를 쓴다. 실시간 마이크 창 1.5초와 같은 길이다. 클립 판정은 세그먼트 확률의 평균이다.
- **누수 방지**: 같은 아기(앱 설치 UUID)는 한 fold에만 들어간다(StratifiedGroupKFold). 재포장·재인코딩 사본은 파일명 키, 해시, 임베딩 유사도로 같은 group에 묶는다. 조기 종료용 검증셋도 group 단위로 따로 뗀다.
- **기준선 비교**: 최빈 클래스, 로지스틱 회귀, MLP 헤드를 세그먼트 단위와 클립 단위로 비교한다(Macro-F1, Balanced Accuracy, 클래스별 recall).
- **판정 보류**: 최고 확률이 `CONF_MIN`보다 낮으면 `uncertain`으로 둔다. 커버리지와, 판정한 것의 정확도를 함께 보고한다.

산출물(`artifacts/`): `cry_head.keras`, `cry_head.tflite`(온디바이스용 헤드), `labels.json`, `meta.json`, `cv_results.json`, `CV_REPORT.md`, `confusion_matrix.png`

## 결과 (2026-09-24, Donate-a-Cry만 사용)

**데이터**: 정제본 457개 + 원본 버킷 436개(정제본에 없는 녹음) = 893개. 울음 세그먼트가 없는 클립을 빼면 **470개**가 남는다(hungry 397 / discomfort 52 / sleepy 21, 아기 229명).
- 원본 버킷에만 있는 녹음은 YAMNet 울음 점수 중앙값이 0.006이라 대부분 울음이 없다. 같은 녹음의 원본과 정제본을 비교하면 점수가 같게 나오므로(중앙값 0.969 vs 0.948) 변환 문제는 아니다. 애초에 울음이 없어서 정제에서 빠진 녹음이다. 실제로 늘어난 건 52개뿐이다.

**5겹 교차검증, 클립 단위 Macro-F1 (평균 ± 표준편차)**

| 설정 | MLP | 로지스틱 회귀 | 최빈 클래스 | sleepy recall (MLP) |
|---|---:|---:|---:|---:|
| 증강 없음 | 0.298 ± 0.019 | 0.296 ± 0.008 | 0.305 ± 0.021 | 0.00 |
| 증강 2개 (피치 포함) | 0.300 ± 0.040 | 0.308 ± 0.018 | 0.305 ± 0.021 | 0.08 |
| 증강 2개 (피치 제외) | 0.280 ± 0.054 | 0.289 ± 0.014 | 0.305 ± 0.021 | 0.07 |

판정 보류를 적용하면 판정한 클립의 정확도는 0.58~0.79다. 하지만 같은 구간을 전부 hungry로 찍어도 0.83~0.85가 나온다. 즉 모델이 최빈 클래스 추측보다 낫지 않다.

**결론**: 공개 데이터(부모 자가 태깅, sleepy는 21개)로는 3클래스 이유 분류가 **기준선을 넘지 못한다**. 버그를 고치고 평가를 제대로 한 결과이므로, 이것이 현재 데이터의 실제 한계다.

## 추가 결과 (2026-09-26, Kaggle 데이터 검증 + 2클래스)

**Kaggle 데이터 검증**
- Infant cry Dataset은 `cry`/`not_cry` 두 폴더뿐이라 이유 라벨이 없다.
- Cry Sense는 대부분 Donate-a-Cry 재포장본이다.
- Cry Sense의 숫자 이름 파일 532개는 이유 라벨 없는 일반 울음 파일과 바이트 단위로 같고, 같은 파일이 복통·트림·불편·피곤 폴더에 복사돼 있다. 근거 없는 라벨이라 `prepare_data.py`가 제외한다.
- 결과적으로 새로 추가된 건 출처 불명 47개뿐이다.

**2클래스(hungry vs discomfort)**: sleepy는 사용 가능 클립이 27개(아기 19명)라 뺐다.

| 학습 데이터 | 로지스틱 회귀 클립 AUC | Balanced Acc |
|---|---:|---:|
| Donate-a-Cry + Kaggle 47개 | 0.729 ± 0.074 | 0.648 |
| Donate-a-Cry만 (`--holdout-source crysense`) | **0.512 ± 0.095** | 0.472 |

Kaggle 파일이 들어가면 신호가 생기는 것처럼 보이지만, 빼면 무작위 수준이다. 출처 불명 파일이 discomfort에 몰려 있어서(26 : 11) 모델이 울음이 아니라 **데이터 출처를 학습한 지름길**이었다.

**최종 결론**: 공개 데이터로는 2클래스로 줄여도 울음 이유를 학습할 수 없다. 보고서에는 이렇게 적는다. "공개 울음 데이터를 검증한 결과 재포장본과 근거 없는 라벨이 섞여 있었고, 정제된 데이터로는 이유 분류가 기준선 수준이라 채택하지 않았다. 대신 맥락 기반 추정(`cry_context`)을 쓴다."

## 시스템 연동 상태

실시간 경로(`../cry_classifier.py`)는 지금 울음 **유무**만 판정하고 `cls="none"`을 발행한다. 이유 분류를 붙이려면 두 가지가 필요하다.
1. 파이의 YAMNet tflite가 **임베딩을 출력**하는지 확인해야 한다. 점수만 나오는 분류용 tflite라면 SavedModel에서 임베딩 출력판을 다시 변환한다.
2. `cry_head.tflite`를 LiteRT로 이어 붙인다.

## 한계 (보고서 표현 주의)

- 라벨은 부모가 앱에서 직접 고른 자가 태깅이라 잡음이 크다. 정답 자체가 불확실하다.
- 공개 데이터는 스마트폰 근접 녹음이다. 침실 마이크 환경과의 차이는 증강으로 일부만 보완된다.
- 이유 분류는 **실험적 힌트**다. 실사용에서는 맥락(`../cry_context.py`)과 함께 보여주고, "울음 이유를 판별한다"고 주장하지 않는다.
