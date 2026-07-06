# Everyday Monitoring Verification Progress Summary

## 1. 실행 기준

- 작업 경로: `/home/user/nuni_run/nuni`
- 브랜치: `main`
- 최신 커밋: `1d2b5f5f2ae027663d200226e58bd93ea3bb7c1d`
- 실행 일시: `2026-07-02 03:20:21 KST`
- Python 환경: `Python 3.10.12`
- 가상환경 특이사항: 기존 `.venv`를 사용했다. `requirements.txt` 의존성은 현재 환경에서 이미 충족됐다.
- 검증 범위: 하드웨어 없는 synthetic signal / controlled scenario 기반 소프트웨어 검증.

## 2. 실행한 명령 요약

| 순서 | 명령/작업 | 성공 여부 | 로그 |
|---:|---|---|---|
| 1 | `pwd`, `git branch --show-current`, `git log --oneline -5`, `git status --short` | 성공 | `results/logs/git_status_everyday.log` |
| 2 | `CHANGES_EVERYDAY_MONITORING.md` 검토 | 성공 | 기준 문서 직접 확인 |
| 3 | 필수 파일 `ls -lh ...` 존재 확인 | 성공 | `results/logs/everyday_file_existence.log` |
| 4 | `.venv` 활성화 및 `python --version` | 성공 | `results/logs/python_version_everyday.log` |
| 5 | `pip install -r requirements.txt` | 성공 | `results/logs/install_requirements_everyday.log` |
| 6 | `python -m compileall .` | 성공 | `results/logs/compileall_everyday.log` |
| 7 | `bash run_all.sh` | 성공 | `results/logs/run_all_everyday.log` |
| 8 | `python eval_sleep_state.py` | 성공 | `results/logs/eval_sleep_state.log` |
| 9 | `python eval_personalization.py` | 성공 | `results/logs/eval_personalization.log` |
| 10 | `python eval_env_control.py` | 성공 | `results/logs/eval_env_control.log` |
| 11 | `timeout 25s streamlit run app.py --server.headless true --server.port 8501` | 실패 | `results/logs/streamlit_everyday.log` |
| 12 | 통합 연결 코드 확인: `fusion.py`, `state_store.py`, `topics.py`, `app.py`, `cloud.py` | 성공 | 코드 직접 확인 |

## 3. 성공한 실행 항목

| 항목 | 결과 | 산출물 |
|---|---|---|
| 전체 파이프라인 | `run_all.sh` 12단계 모두 완료 | `results/RESULTS.md` |
| 기본 문법 검증 | compileall 성공 | `results/logs/compileall_everyday.log` |
| 수면/각성 상태 평가 | 7/7 correct, 정확도 100.0% | `results/SLEEP_STATE_RESULT.md`, `results/artifacts/sleep_state_eval.csv` |
| 개인화 baseline 평가 | babyA 40.0 -> 40.4, babyB 52.0 -> 51.7 | `results/PERSONALIZATION_RESULT.md`, `results/artifacts/personalization_eval.csv` |
| 환경 제어 권고 평가 | 8/8 correct, 일치율 100.0% | `results/ENV_CONTROL_RESULT.md`, `results/artifacts/env_control_eval.csv` |
| dashboard import smoke | 신규 모듈 포함 import 성공 | `results/logs/dashboard_import_smoke.log` |

## 4. 새 일상 모니터링 기능 검증 결과

| 기능 | 평가 스크립트 | 결과 | 산출물 |
|---|---|---|---|
| 수면/각성 상태 | `eval_sleep_state.py` | 통제 시나리오 7개, 정확도 100.0%. 레이더 움직임과 `is_crying`만 사용. | `results/SLEEP_STATE_RESULT.md`, `results/artifacts/sleep_state_eval.csv` |
| 개인화 baseline | `eval_personalization.py` | babyA 학습 BPM 40.4, babyB 학습 BPM 51.7. EMA 기반 baseline. | `results/PERSONALIZATION_RESULT.md`, `results/artifacts/personalization_eval.csv` |
| 환경 제어 권고 | `eval_env_control.py` | 통제 시나리오 8개, 권고 action 일치율 100.0%. | `results/ENV_CONTROL_RESULT.md`, `results/artifacts/env_control_eval.csv` |

## 5. 기존 기능 재검증 결과

| 기능 | 결과 |
|---|---|
| radar DSP | synthetic raw signal 기준 BPM MAE 0.500, apnea detection delay 4.0s, motion detected True |
| radar robustness | SNR 5/10/15/20dB x window 15/30/60s 실행. 15s MAE 1.000, 30s/60s MAE 0.000 |
| fusion | controlled scenario 11개. full_fusion false alarm/miss 0/0, radar_only 0/2, single_sensor 6/0 |
| stream | raw alerts 4 -> tuned alerts 1, transient false alerts 3 -> 0, tuned apnea latency 1s |
| voice intent | curated text 11개 기준 100.0%. 실제 STT/마이크 검증 아님 |
| cloud sleep report | synthetic night 기준 울음 4회, 무호흡 의심 1회, 평균 호흡 39.9회/분, 개인 정상 호흡 범위 `(32.4, 47.4)` 포함 |

## 6. 생성/갱신된 주요 파일

- `results/EVERYDAY_PROGRESS_SUMMARY.md`
- `results/EVERYDAY_REQUIREMENTS_STATUS.md`
- `results/EVERYDAY_FAILURES_AND_LIMITATIONS.md`
- `results/RESULTS.md`
- `results/SLEEP_STATE_RESULT.md`
- `results/PERSONALIZATION_RESULT.md`
- `results/ENV_CONTROL_RESULT.md`
- `results/SLEEP_REPORT.md`
- `results/artifacts/sleep_state_eval.csv`
- `results/artifacts/personalization_eval.csv`
- `results/artifacts/env_control_eval.csv`

## 7. 현재 repo 상태

- git status 요약: results 문서와 small artifact CSV가 변경/생성됨. `results/logs/`는 `.gitignore` 대상이다.
- 커밋 대상: `results/EVERYDAY_*.md`, 관련 `results/*.md`, `results/artifacts/*.csv` 중 작은 정량 결과 파일.
- 제외 대상: `results/logs/`, `.venv/`, `reports/`, `cry_model/artifacts*`, `__pycache__/`, `*.pyc`.
- 커밋/푸시 상태: `git add` 단계에서 `.git/index.lock` 생성이 읽기 전용 파일 시스템으로 차단되어 로컬 커밋을 만들지 못했다. 사용자가 쓰기 가능한 Git 환경에서 직접 stage/commit/push해야 한다.
