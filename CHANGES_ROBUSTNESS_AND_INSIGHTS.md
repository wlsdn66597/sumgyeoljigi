# NUNI — 강건성 재증명 · 레이더 현실화 · 맥락/리듬 인사이트 (변경·실행·결과)

> 평가자 피드백 반영: (2) 융합 우위를 통제 시나리오가 아닌 노이즈 분포에서 재증명,
> (3) 레이더 DSP 현실화(range-bin·디트렌드·대역통과), 그리고 SW로 가능한 추가 기능
> (울음-맥락 상관, 수면 리듬 루틴 인사이트)을 구현했다. 모두 합성/통제 기반이며 실센서·
> 실영유아 검증이 아니다. 울음 '이유' 분류에는 의존하지 않는다.

---

## 1. 변경 사항

### 1.1 신규 모듈
| 파일 | 역할 |
|---|---|
| `eval_fusion_noisy.py` | 무작위 노이즈 2000회 몬테카를로로 단일센서 vs 융합 오탐률 비교 |
| `eval_radar_realistic.py` | 다중 bin·클러터·드리프트 신호에서 naive vs 현실적 DSP 비교 |
| `cry_context.py` (+eval) | 울음-환경/시간 맥락 상관 분석 (이유 분류 대체) |
| `sleep_rhythm.py` (+eval) | 여러 밤 수면 상태 → 루틴 인사이트(자주 깨는 시간대 등) |

### 1.2 변경된 모듈
| 파일 | 변경 |
|---|---|
| `radar_dsp.py` | `select_range_bin`·`detrend`·`bandpass_fft`·`estimate_bpm_realistic` 추가 |
| `radar_sim_signal.py` | `generate_multibin`(공통 대역내 간섭 포함 다중 bin) 추가 |
| `cloud.py` | 수면 리포트에 **울음-맥락 상관** 섹션 + 습도 필드 추가 |
| `run_all.sh` | 16단계로 확장(신규 평가 4종 편입) |
| 결과 기록 인코딩 | `cloud/eval_radar/eval_fusion/eval_stream/voice_intent`의 `write_text`에 `encoding="utf-8"` 추가(플랫폼 무관 저장) |

## 2. 실행 절차 (리눅스)
```bash
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh           # 16단계 전체 → results/ 생성
```
개별 실행:
```bash
python eval_fusion_noisy.py       # 노이즈 융합 강건성
python eval_radar_realistic.py    # 현실적 레이더 DSP
python eval_cry_context.py        # 울음-맥락 상관
python eval_sleep_rhythm.py       # 수면 리듬 루틴 인사이트
```

## 3. 남겨야 할 결과물
| 파일 | 담긴 값 | 보고서 위치 |
|---|---|---|
| `results/FUSION_NOISY_RESULT.md` + `artifacts/fusion_noisy_summary.csv` | 노이즈 하 오탐률(단일 vs 융합) | 구현기능 — 융합 강건성 |
| `results/RADAR_REALISTIC_RESULT.md` + `artifacts/radar_realistic.csv` | naive vs 현실적 DSP MAE | 구현기능 — 레이더 전처리 |
| `results/CRY_CONTEXT_RESULT.md` | 울음-환경/시간 상관 인사이트 | 구현기능 — 울음-맥락 |
| `results/SLEEP_RHYTHM_RESULT.md` | 수면 리듬 루틴 인사이트 | 구현기능 — 개인화/리듬 |
| `results/SLEEP_REPORT.md` | 울음-맥락 섹션 포함 야간 리포트 | 구현기능 — 클라우드 |

### 3.1 현재 코드 기준 기대값 (재현 대조용)
- 융합 강건성(노이즈 2000회): 단일센서 오탐률 **≈57.6%** → 융합 **≈2.7%**, 미탐 0
- 현실적 레이더 DSP: naive MAE **15.0** → realistic **0.5** 회/분 (공통 간섭 30bpm에 naive가 끌려감)
- 울음-맥락: "울음 시 평균 습도↓ → 건조 연관" + "자주 우는 시간대" 검출
- 수면 리듬: "밤 평균 각성 N회, 자주 깨는 시간대 X시경" 등 루틴 인사이트

## 4. 의의 (평가자 피드백 대응)
- **융합 우위가 "우리가 짠 시나리오"라는 반박 방어**: 무작위 노이즈 분포에서도 융합이 오탐률을 대폭 낮춤을 보임.
- **"합성이 쉬워서 잘 됐다" 반박 방어**: 다중 bin·간섭이 있는 어려운 신호에서 naive는 실패(MAE 15), 현실적 전처리가 회복(0.5).
- **울음 이유 분류 약점 우회**: 이유를 못 맞혀도 울음-맥락 상관으로 선제 대응 근거 제공.
- **일상 가치 강화**: 수면 리듬 루틴 인사이트로 매일 쓰는 개인화 코칭 제공.

## 5. 한계
- 모든 결과는 합성 노이즈·통제 데이터 기반이며 실제 센서/영유아 인과를 증명하지 않는다.
- 실물 통합·실환경 재보정은 2차 데모 과제로 남는다.
