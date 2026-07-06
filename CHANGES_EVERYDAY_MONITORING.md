# NUNI — 일상 모니터링 기능 추가 (변경·실행·결과)

> 포지셔닝 재정립: **메인 = 일상 비접촉 수면·상태 모니터링 + 울음 대응 + 환경/개인화
> 선제 케어**. 무호흡/호흡이상은 드문 **안전 백업**이며 의료기기 주장은 하지 않는다.
> (모두 합성/통제 시나리오 검증이며, 울음 '이유' 분류에는 의존하지 않는다.)

---

## 1. 변경 사항

### 1.1 신규 모듈
| 파일 | 역할 |
|---|---|
| `sleep_state.py` | 수면/각성 상태 분류(안정수면·뒤척임·각성) — 레이더 움직임 + 울음 감지 |
| `personalization.py` | 경량 개인화: 호흡·움직임 baseline EMA 학습 → 개인 정상범위 |
| `env_control.py` | 선제 환경 제어 권고(환기·가습·냉난방·조명) |
| `eval_sleep_state.py` / `eval_personalization.py` / `eval_env_control.py` | 각 기능 평가 |

### 1.2 변경된 모듈
| 파일 | 변경 |
|---|---|
| `fusion.py` | `_process`에서 수면상태·개인화 갱신·환경제어 권고를 계산해 store에 반영, `control/action` 발행 |
| `state_store.py` | `sleep_state`·`actions`·`personal` 필드 + setter + snapshot 추가 |
| `topics.py` | `CONTROL = "control/action"` 토픽 추가 |
| `app.py` | 대시보드에 **수면 상태**, **개인화 정상범위**, **환경 제어 권고** 표시 |
| `cloud.py` | 수면 리포트에 **학습된 개인 정상 호흡범위** 반영 |
| `run_all.sh` | 12단계로 확장(수면상태·개인화·환경제어 평가 편입) |

### 1.3 울음 이유 분류와의 관계
- 세 기능 모두 울음 **이유** 분류에 **비의존**. 울음은 감지(is_crying)만 각성 판단에 사용.

---

## 2. 실행 절차 (리눅스)

```bash
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh                 # 12단계 전체 → results/ 생성
streamlit run app.py            # (수동) 수면상태·환경권고 화면 캡처
```

개별 실행:
```bash
python eval_sleep_state.py       # 수면/각성 분류 정확도
python eval_personalization.py   # 가구별 baseline 적응
python eval_env_control.py       # 환경 상황 → 동작 일치율
```

---

## 3. 남겨야 할 결과물

| 파일 | 담긴 값 | 보고서 위치 |
|---|---|---|
| `results/SLEEP_STATE_RESULT.md` + `artifacts/sleep_state_eval.csv` | 수면/각성 분류 정확도 | 구현기능 — 일상 상태 모니터링(헤드라인) |
| `results/PERSONALIZATION_RESULT.md` + `artifacts/personalization_eval.csv` | 가구별 학습 baseline·오차 | 구현기능 — 개인화 |
| `results/ENV_CONTROL_RESULT.md` + `artifacts/env_control_eval.csv` | 환경→동작 일치율 | 구현기능 — 선제 케어 |
| `results/SLEEP_REPORT.md` | 개인 정상 호흡범위 포함 야간 요약 | 구현기능 — 클라우드/개인화 |
| 대시보드 스크린샷(수면상태·환경권고 표시) | 화면 | 작품 사진 |

### 3.1 현재 코드 기준 기대값 (재현 대조용)
- 수면/각성 분류 정확도: **100%** (통제 7 시나리오)
- 개인화: babyA(40) → 학습 40.1 / babyB(52) → 학습 51.7 (고정 seed·재현 가능)
- 환경 제어 동작 일치율: **100%** (통제 8 시나리오)

---

## 4. 포지셔닝 반영 포인트 (보고서 작성 시)
- 헤드라인 = **"매일 쓰는 비접촉 수면·상태 모니터링 + 울음 대응 + 환경/개인화 선제 케어"**
- 무호흡 데모/결과 = **"드물지만 놓치지 않는 안전 백업"** 섹션에 배치(헤드라인 아님)
- 의료 표현 금지: 임상/진단 성능 주장하지 않음(생활 가전 포지셔닝 유지)
