# 울음 분류 (YAMNet 전이학습)

**2단계 설계**
1. **울음 감지(is_crying)** — YAMNet 내장 AudioSet `Baby cry, infant cry` 점수 (별도 학습 불필요)
2. **울음 이유 분류** — YAMNet 임베딩(1024-d) → 커스텀 헤드 (공개 데이터로 학습)

## 설치
```bash
pip install tensorflow tensorflow-hub librosa soundfile scikit-learn matplotlib pandas sounddevice
```

## 데이터 준비
Donate-a-Cry Corpus 내려받아 라벨별 폴더로 배치:
```
data/
  hungry/*.wav   tired/*.wav   discomfort/*.wav   belly_pain/*.wav   burping/*.wav
```
(github.com/gveres/donateacry-corpus 의 cleaned 폴더 구조 그대로)

## 학습 · 평가
```bash
python prepare_data.py data           # 클래스 분포 확인
python train.py data --aug 2          # 학습 → artifacts/ (모델·라벨·혼동행렬)
python evaluate.py data               # held-out 재평가 (classification report + Macro-F1)
```
- **화자/녹음(group) 단위 held-out** 분할로 누수 방지
- **증강**(잔향·거리감쇠·생활소음·시간/피치)으로 도메인 갭 보완
- **class_weight**로 클래스 불균형 대응

## 실시간 추론 / 시스템 연동
```bash
python infer.py                       # 마이크로 단독 테스트
```
데모에 연결하려면 `../cry_classifier.py` 에서 `REAL = True` 로 변경.
→ `artifacts/` 의 모델을 로드해 마이크 최근 1.5초를 분류하고 `audio/cry` 토픽으로 발행.

## 보고서 매핑 (`구현기능`)
- 데이터 파이프라인 도식 · 클래스 분포
- 전이학습 구조(YAMNet → 헤드) 그림
- **혼동행렬**(`artifacts/confusion_matrix.png`) · **Macro-F1**/클래스별 성능표
- 증강 전후 비교 · 화자단위 held-out 설명 (검증의 정직성)
