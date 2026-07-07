# NUNI — 비접촉 영유아 케어 AI (소프트웨어)

레이더·음향·환경 센서로 **카메라·웨어러블 없이** 영유아의 **수면·상태를 일상적으로
모니터링**하고, 울음 대응·환경 제어·개인화로 선제 케어하는 AI 시스템의 **소프트웨어
파이프라인**.

---

## 1. 프로젝트 개요

- **문제**: 부모는 영유아 상태를 계속 확인해야 하는 불안·수면부족을 겪는다. 기존 제품은 웨어러블 착용·카메라 사용이라 프라이버시·거부감 문제가 있다.
- **해결**: 60GHz 레이더(호흡·움직임)·마이크(울음·음성)·환경센서(온습도·CO₂·조도)를 **멀티모달 융합**해, 카메라 없이 상태를 해석하고 선제 알림/제어한다.
- **핵심 차별점**: ①비접촉·비영상 ②멀티모달 융합으로 단일센서 오탐 감소 ③엣지 처리로 프라이버시 보호.

## 2. 시스템 구조

```
[센서/음성 입력]  ──(MQTT 토픽)──▶  [엣지 AI 추론]  ──▶  [멀티모달 융합]  ──▶  [대시보드·알림]
 레이더/환경(가상·실물)              울음분류(YAMNet)        시간동기·교차검증
 마이크                             음성 인텐트             위험도·불편 판단
```
모든 모듈은 MQTT 토픽으로만 통신 → **가상 센서를 실물로 무중단 교체 가능**. (토픽 규격은 §부록)

## 3. 구현 상태

| 구분 | 항목 |
|---|---|
| **메인 (일상)** | 수면/각성 상태 분류 · 울음 감지 · 환경 모니터링·선제 제어 · 개인화(적응형 baseline) · 수면 리포트 |
| **안전 백업 (드묾)** | 호흡 이상/무호흡 감지 알림 (의료기기 아님) |
| **기반 (완료)** | MQTT 아키텍처 · 멀티모달 융합 엔진 · 대시보드 · 평가 프레임워크 |
| **2차 데모까지 예정** | 실물 60GHz 레이더·환경센서 통합 · 원거리 마이크 실수음 · 실환경 재검증 |

## 4. 저장소 구성

```
nuni_demo/
├─ topics.py            # MQTT 토픽 규격(이름 + 메시지 스키마)
├─ bus.py               # pub/sub 추상화 (inproc 기본 / mqtt 선택)
├─ sim_sensors.py       # 레이더·환경 가상 발행자  ← 실물 교체 지점
├─ cry_classifier.py    # 울음 분류 (시뮬 기본 / REAL=True 시 YAMNet)
├─ fusion.py            # 멀티모달 융합 판단 + 단일센서 비교 정책
├─ workers.py           # inproc 백그라운드 워커 기동
├─ app.py               # Streamlit 실시간 대시보드
├─ eval_radar.py        # 레이더 호흡 DSP 검증 (MAE)
├─ eval_radar_robustness.py  # 레이더 강건성 스윕 (SNR·윈도우)
├─ eval_radar_realistic.py   # 현실적 레이더 DSP (range-bin 선택+디트렌드+대역통과)
├─ eval_fusion.py       # 융합 vs 단일센서 오탐/미탐 평가 (+경계신호 교차검증)
├─ eval_fusion_noisy.py # 노이즈 몬테카를로 융합 강건성 재증명
├─ eval_stream.py       # 스트리밍 시계열 평가 (디바운스·쿨다운 검증)
├─ sleep_state.py       # 수면/각성 상태 분류 (일상 헤드라인) + eval_sleep_state.py
├─ personalization.py   # 경량 개인화(적응형 baseline) + eval_personalization.py
├─ env_control.py       # 선제 환경 제어 권고 + eval_env_control.py
├─ cry_context.py       # 울음-맥락 상관 분석 + eval_cry_context.py
├─ sleep_rhythm.py      # 수면 리듬 → 루틴 인사이트 + eval_sleep_rhythm.py
├─ voice_intent.py      # 부모 음성 인텐트 인식 (키워드/STT)
├─ cloud.py             # 클라우드 스텁 + 아침 수면 리포트
├─ requirements.txt     # 데모 실행 의존성
└─ cry_model/           # 울음 분류 학습 파이프라인
   ├─ config.py         # 경로·하이퍼파라미터
   ├─ features.py       # YAMNet 로드 + 임베딩/울음감지(+캐시)
   ├─ augment.py        # 잔향·소음·거리 증강 (실측 파일 or 합성)
   ├─ prepare_data.py   # 데이터 인덱싱(화자 group 분리)
   ├─ train.py          # 전이학습 + held-out 평가 + 혼동행렬
   ├─ evaluate.py       # 저장 모델 재평가
   ├─ infer.py          # 마이크/파일 추론
   ├─ check_env.py      # 학습 전 환경 점검
   ├─ download_data.py  # 데이터 자동 다운로드
   └─ requirements-ml.txt
```

## 5. 데모 실행 (하드웨어 불필요)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py          # 대시보드 (사이드바로 울음/무호흡 주입)
python eval_fusion.py         # 융합 성능 비교표
```

## 6. 울음 분류 학습 (핵심 AI)

```bash
source .venv/bin/activate
pip install -r cry_model/requirements-ml.txt

python cry_model/download_data.py cry     # Donate-a-Cry → cry_model/data/
python cry_model/download_data.py noise   # (선택) ESC-50 → cry_model/noise/
python cry_model/check_env.py             # 라이브러리·YAMNet·데이터 점검
python cry_model/prepare_data.py cry_model/data   # 클래스 분포 확인
python cry_model/train.py cry_model/data --aug 2  # 학습 → cry_model/artifacts/
python cry_model/evaluate.py cry_model/data       # held-out 재평가
```
학습 후 데모에 연결: `cry_classifier.py`에서 `REAL = True`.

## 7. 필요한 데이터셋

| 데이터 | 용도 | 출처 |
|---|---|---|
| Donate-a-Cry Corpus | 울음 이유 분류 학습(필수) | github.com/gveres/donateacry-corpus |
| Infant Cry Audio Corpus | 표본 보강(권장) | Kaggle |
| ESC-50 | 증강 소음 + 감지 오탐 검증 | github.com/karoldvl/ESC-50 |
| MUSAN / RIR(OpenAIR 등) | 실측 소음·잔향 증강 | openslr.org/17 |
| YAMNet | 사전학습 백본(자동 로드) | TF Hub |

> ⚠️ Donate-a-Cry는 표본이 적고 불균형 → 증강·class_weight·전이학습으로 보완, rare 클래스는 합쳐 거친 카테고리로.

## 8. 받아와야 할 결과물 (보고서용)

| 산출물 | 생성 방법 | 보고서 위치 |
|---|---|---|
| **융합 vs 단일 오탐/미탐 표** | `python eval_fusion.py` | 구현기능 — 융합 효과 |
| **혼동행렬 이미지** | `train.py` → `cry_model/artifacts/confusion_matrix.png` | 구현기능 — 울음 분류 |
| **Macro-F1 / 클래스별 성능** | `train.py`·`evaluate.py` 출력 | 구현기능 — 울음 분류 |
| **대시보드 스크린샷** | `streamlit run app.py` 실행 화면 | 작품 사진 / 구현기능 |
| **E2E 지연 측정** | 이벤트→알림 타임스탬프 | 검증 방법 및 결과 |

## 9. Reproducible Software Verification

하드웨어 없이 재현 가능한 검증 스크립트를 제공한다. 아래 결과는 현재 코드 기준으로 `bash run_all.sh`를 실행해 얻은 값이며, 실제 60GHz 레이더 하드웨어나 실제 영유아 환경 검증이 아니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
streamlit run app.py
```

### 핵심 결과

| 영역 | 항목 | 결과 | 조건/산출물 |
|---|---|---:|---|
| Radar DSP | BPM MAE | 0.500 breaths/min | synthetic raw radar-like signal, 30s, fs=50Hz, SNR 15dB |
| Radar DSP | Apnea detection | True, delay 4.0s | synthetic apnea scenario |
| Radar DSP | Motion detection | True | synthetic motion burst |
| Fusion | full_fusion false alarm / miss | 0 / 0 | controlled scenarios 11개 |
| Fusion | single_sensor false alarm / miss | 6 / 0 | controlled scenarios 11개 |
| Fusion ablation | radar_only false alarm / miss | 0 / 2 | 경계 호흡을 단독으론 놓침 |
| Fusion ablation | audio_only false alarm / miss | 2 / 2 | cry-only 정책의 오탐/미탐 |
| Fusion ablation | env_only false alarm / miss | 2 / 3 | 환경 단독 정책의 한계 |
| Cry reason classifier | Donate-a-Cry 5-class | 실사용 성능 주장 불가 | hungry 83.6% class imbalance |

### 결과 해석

- 레이더 DSP는 실제 센서가 아니라 synthetic raw radar-like signal을 입력으로 검증했다.
- controlled scenario 11개에서 **full fusion만 false alarm 0, miss 0** 을 기록했다. radar_only는 명확한 무호흡은 잡지만 '경계 호흡'(임계 미만의 이상 호흡)을 단독으론 놓쳐 miss 2가 발생했고, 이는 울음·환경과 교차검증할 때만 경보가 된다. 즉 융합의 가치는 (1) 과민한 단일신호 정책 대비 오탐 감소, (2) 어떤 단일 모달보다 적은 미탐 둘 다에서 나타난다.
- single_sensor baseline은 단일 신호에 민감하게 반응해 false alarm 6건이 발생했다.
- Donate-a-Cry 기반 울음 이유 분류는 `hungry`가 382/457개(83.6%)로 치우쳐 있고 rare class가 적어, 현재 결과를 실사용 가능한 이유 분류 성능으로 주장하지 않는다.

### 한계

- 실제 60GHz 레이더 하드웨어 성능을 검증한 결과가 아니다.
- synthetic apnea/motion smoke test는 실환경 무호흡/움직임 탐지 검증이 아니다.
- 실제 영유아 대상 성능 또는 임상적 안전성을 검증하지 않았다.
- YAMNet cry detection 결과는 synthetic negative 기준 smoke test이며 실제 생활 소음에서의 precision을 보장하지 않는다.
- Donate-a-Cry 5-class reason classifier는 class imbalance 때문에 실험적 기능으로만 다룬다.

주요 산출물:

- `results/RADAR_DSP_RESULT.md`
- `results/FUSION_EVALUATION_RESULT.md`
- `results/STREAM_EVAL_RESULT.md` (디바운스·쿨다운 시계열 검증)
- `results/FUSION_NOISY_RESULT.md` (노이즈 몬테카를로 융합 강건성)
- `results/RADAR_REALISTIC_RESULT.md` (현실적 레이더 DSP)
- `results/SLEEP_STATE_RESULT.md` (수면/각성 상태 분류)
- `results/PERSONALIZATION_RESULT.md` (개인화 적응)
- `results/ENV_CONTROL_RESULT.md` (선제 환경 제어)
- `results/CRY_CONTEXT_RESULT.md` (울음-맥락 상관)
- `results/SLEEP_RHYTHM_RESULT.md` (수면 리듬 루틴 인사이트)
- `results/VOICE_INTENT_RESULT.md` (음성 인텐트 인식)
- `results/SLEEP_REPORT.md` (클라우드 아침 수면 리포트)
- `results/RESULTS.md`
- `results/RESULTS_FINAL.md`

## 10. 실물 교체 지점 (2차 데모까지)

1. **센서**: `sim_sensors.py`의 `radar_step()/env_step()`를 실물 UART/I2C 읽기로 교체 (토픽 동일)
2. **울음**: `cry_classifier.py` `REAL=True` + 학습된 `artifacts/` 사용
3. **분산 실행**: `NUNI_BUS=mqtt` + 로컬 Mosquitto → 각 모듈 별도 프로세스

## 11. 검증 방법 & 정직성

- 울음/음성: **공개 데이터 라벨 = GT**, 화자단위 held-out.
- 레이더/환경: **시뮬레이션·공개데이터로 검증**(코드에 시뮬 명시), 실물 재검증은 2차.
- 융합: **통제된 시나리오 정답과 비교**(`eval_fusion.py`).
- 본 검증은 실제 영유아 임상검증이 아니며, 제품화 시 IRB·임상이 필요하다.

---

## 부록 — MQTT 토픽 규격

| 토픽 | 페이로드 |
|---|---|
| `sensor/radar` | `{breathing_rate, movement(0~1), presence, ts}` |
| `sensor/env` | `{temp, humidity, co2, lux, ts}` |
| `audio/cry` | `{is_crying, cls, confidence, ts}` |
| `audio/voice` | `{intent, ts}` |
| `fusion/state` | `{level(normal/attention/alert), reasons[], ts}` |
| `fusion/alert` | `{level, reason, ts}` |
| `control/action` | `{actions[], ts}` |
