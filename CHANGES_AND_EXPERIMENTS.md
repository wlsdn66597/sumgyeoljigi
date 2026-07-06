# NUNI — 변경 사항 및 실험 수행 가이드

> 목적: (1) 기존 코드에서 무엇이 바뀌었는지 정리하고, (2) 어떤 실험을 실행해
> 어떤 결과물을 남겨야 하는지 규정한다. 실행 환경은 리눅스 기준.
> 전 과정은 하드웨어 없이 재현 가능하며, **합성 신호·통제 시나리오 기반**이다
> (실제 60GHz 센서·실제 영유아·임상 검증이 아님).

---

## 1. 기존 코드 대비 변경 사항

### 1.1 신규 모듈
| 파일 | 역할 | 핵심 |
|---|---|---|
| `radar_dsp.py` | 레이더 원신호 → 호흡수/무호흡/움직임 | FFT 호흡대역 피크, 무호흡=최근 4s RMS, 움직임=최근 2s 고주파 |
| `radar_sim_signal.py` | 합성 원신호 생성 | 설정 BPM=GT, 무호흡/움직임/잡음 포함 |
| `eval_radar.py` | 호흡 DSP 검증 | 설정 BPM 대비 MAE + 무호흡/움직임 스모크 |
| `eval_radar_robustness.py` | 호흡 강건성 스윕 | SNR×윈도우별 MAE |
| `eval_stream.py` | 시계열 융합 검증 | 디바운스·쿨다운의 알림/지연/오경보 효과 |
| `voice_intent.py` | 부모 음성 인텐트 | 키워드(+STT 훅) 인식, 정확도 평가 |
| `cloud.py` | 클라우드 스텁 | 비식별 이벤트 → 아침 수면 리포트 |
| `cry_model/` | 울음 분류 학습 파이프라인 | YAMNet 전이학습, 증강, 화자단위 held-out |

### 1.2 변경된 모듈
| 파일 | 이전 | 변경 후 |
|---|---|---|
| `sim_sensors.py` | 호흡수를 난수로 반환 | **실제 DSP 경로**로 합성 원신호→호흡수 산출 |
| `fusion.py` | 규칙 + 디바운스/쿨다운(초기) | **경계 신호 교차검증 규칙** 추가 + `now_fn` 주입 + `update()`(시계열 구동) |
| `eval_fusion.py` | 8 시나리오 | **11 시나리오**(융합 우위 케이스 3종 포함) + ablation |
| `run_all.sh` | 5단계 | **8단계** 재현 파이프라인 + 요약 자동 생성 |

### 1.3 핵심 로직 변경 요점
- **경계 신호 교차검증**(`fusion.decide`): 무호흡 임계 미만의 '경계 호흡'(8~15/분 등)은 단독으론 정상이지만, 울음/나쁜 환경과 겹치면 경보로 상향 → *레이더 단독은 놓치나 융합은 잡음*.
- **디바운스/쿨다운**(`fusion.Fusion`): 연속 2회 확정 + 동일 경보 10초 억제. `now_fn`으로 가짜 시계 주입해 시계열 평가 가능.

---

## 2. 수행할 실험

### 실험 1 — 소프트웨어 전체 재현 (필수, 무하드웨어)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```
- 목적: 컴파일 + 레이더/융합/스트리밍/음성/클라우드 결과를 한 번에 생성.
- 기록: 콘솔 마지막 `results/RESULTS.md` 및 `results/` 산출물 전부.

### 실험 2 — 대시보드 시연 (필수, 수동 캡처)
```bash
streamlit run app.py
```
- 사이드바로 **정상 / 울음 / 무호흡** 상태를 각각 재현하고 화면 캡처.
- 기록: 스크린샷 3장 + 이벤트 발생→경보까지 걸린 시간(초) 관측값.

### 실험 3 — 울음 분류 학습 (데이터 있을 때)
```bash
pip install -r cry_model/requirements-ml.txt
python cry_model/download_data.py cry
python cry_model/check_env.py
python cry_model/train.py cry_model/data --aug 2
python cry_model/evaluate.py cry_model/data
```
- 기록: `cry_model/artifacts/confusion_matrix.png`, 콘솔 Macro-F1·클래스별 성능.

### 실험 4 — 증강 효과 비교 (권장, 보고서 강화용)
```bash
python cry_model/train.py cry_model/data --aug 0   # 증강 없음 → Macro-F1 기록
python cry_model/train.py cry_model/data --aug 2   # 증강 적용 → Macro-F1 기록
```
- 목적: 데이터 증강이 일반화에 주는 효과를 정량 비교(ablation).
- 기록: 두 경우의 Macro-F1 값(표로).

### 실험 5 — 레이더 강건성 스윕 (권장, 실행 가능)
```bash
python eval_radar_robustness.py     # run_all.sh 에도 포함됨
```
- 목적: SNR(5·10·15·20dB) × 관측 윈도우(15·30·60s)에 따른 호흡수 MAE 표 확보.
- 기록: `results/RADAR_ROBUSTNESS_RESULT.md`, `results/artifacts/radar_robustness.csv`.

---

## 3. 남겨야 할 결과물

### 3.1 자동 생성 (실험 1) — 파일 그대로 보존
| 파일 | 담긴 값 | 보고서 위치 |
|---|---|---|
| `results/RESULTS.md` | 전체 요약(환경·커밋·핵심 수치) | 구현기능 개요 |
| `results/RADAR_DSP_RESULT.md` + `artifacts/radar_bpm_*.csv` | BPM MAE, 무호흡 지연, 움직임 | 구현기능 — 호흡 |
| `results/RADAR_ROBUSTNESS_RESULT.md` + `artifacts/radar_robustness.csv` | SNR×윈도우별 MAE | 구현기능 — 호흡 강건성 |
| `results/FUSION_EVALUATION_RESULT.md` + `artifacts/fusion_*.csv` | 정책별 오탐/미탐, ablation | 구현기능 — 융합 |
| `results/STREAM_EVAL_RESULT.md` + `artifacts/stream_eval_summary.csv` | 디바운스/쿨다운 효과 | 구현기능 — 알림 신뢰성 |
| `results/VOICE_INTENT_RESULT.md` | 음성 인텐트 인식률 | 구현기능 — 음성 |
| `results/SLEEP_REPORT.md` + `artifacts/sleep_timeline.*` | 아침 수면 리포트 | 구현기능 — 클라우드/개인화 |

### 3.2 수동 수집 (실험 2)
| 항목 | 형태 |
|---|---|
| 대시보드 스크린샷 3종(정상/울음/무호흡) | PNG |
| E2E 지연 관측값 | "이벤트→경보 O초" |

### 3.3 학습 시 (실험 3·4)
| 항목 | 위치/형태 |
|---|---|
| 혼동행렬 | `cry_model/artifacts/confusion_matrix.png` |
| Macro-F1 / 클래스별 성능 | 콘솔 출력 복사 |
| 증강 유무 Macro-F1 비교 | 표(aug0 vs aug2) |

### 3.4 기대 기준값 (현재 코드 기준, 재현 시 참고)
- 레이더 BPM MAE ≈ 0.5회/분, 무호흡 검출 지연 ≈ 4.0s
- 융합: full_fusion 오탐 0·미탐 0 / single 6·0 / radar_only 0·2
- 스트리밍: 알림 4→1, 일시 오경보 3→0, 무호흡 지연 1s
- 음성 인텐트: curated 세트 100%(실환경은 낮아짐, 한계 명시)
- 레이더 강건성: 15s 창 MAE≈1.0, 30s 이상 ≈0.0 (합성, SNR 5~20dB 영향 미미)

---

## 4. 결과 전달 방법
실행 후 `results/` 내용과 (있으면) 학습 산출물·스크린샷을 모아 전달하며, 보고서
`구현기능` 절 정리를 요청한다. 모든 수치는 **합성/통제 조건**임을 함께 표기한다.
