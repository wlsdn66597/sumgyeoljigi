# Everyday Monitoring Failures, Skips, and Limitations

## 1. 실패한 명령

| 명령 | 실패 여부 | 원인 | 영향 | 대응 |
|---|---|---|---|---|
| `timeout 25s streamlit run app.py --server.headless true --server.port 8501` | 실패 | sandbox에서 `socket.socket(...)` 호출이 `PermissionError: [Errno 1] Operation not permitted`로 차단됨 | 자동 서버 smoke 및 스크린샷 생성 불가 | dashboard import smoke는 성공했으므로 코드 import 문제로 보지 않고 수동 캡처 필요로 기록 |
| `git add ...` | 실패 | `.git/index.lock` 생성이 읽기 전용 파일 시스템으로 차단됨 | stage/commit/push 불가 | 쓰기 가능한 로컬 Git 환경에서 직접 stage/commit/push 필요 |

## 2. 스킵한 항목

| 항목 | 이유 | 영향 | 후속 작업 |
|---|---|---|---|
| 실제 대시보드 스크린샷 자동 캡처 | Streamlit 서버 socket 권한 제한 | UI 증빙 이미지 없음 | 로컬 브라우저에서 수동 캡처 |
| 실제 UI E2E 지연 관측 | UI를 띄울 수 없어 이벤트->경보 시간을 관측하지 못함 | 보고서에 실제 UI 지연 수치 사용 불가 | 대시보드 실행 후 수동 기록 |
| 실제 레이더/마이크/환경센서 검증 | 하드웨어 없는 검증 범위 | 실환경 성능 주장 불가 | 2차 데모에서 실물 통합 |
| 실제 IoT 기기 제어 | `env_control.py`는 권고 action 산출만 수행 | 자동제어 완료 주장 불가 | 스마트플러그/IR 허브 연동 필요 |
| 울음 이유 분류 재학습 | 이번 목표는 일상 모니터링 검증이며 오래 걸리는 ML 학습은 제외 | 신규 울음 이유 분류 성능 없음 | 별도 ML 검증에서 수행 |

## 3. 수동으로 필요한 작업

| 작업 | 이유 | 결과물 |
|---|---|---|
| 정상 수면 상태 대시보드 캡처 | 서버 socket 권한 제한 가능 | `dashboard_normal_sleep.png` |
| 뒤척임/각성 상태 대시보드 캡처 | UI에서 상태 표시 확인 필요 | `dashboard_restless_awake.png` |
| 울음 감지 상태 대시보드 캡처 | `is_crying` 기반 각성/주의 표시 확인 필요 | `dashboard_cry_detected.png` |
| 환경 제어 권고 표시 캡처 | 선제 케어 권고 UI 확인 필요 | `dashboard_env_recommendation.png` |
| 개인화 정상범위 표시 캡처 | 개인화 baseline UI 확인 필요 | `dashboard_personal_range.png` |
| 무호흡 안전 백업 경보 캡처 | 안전 백업 기능 UI 확인 필요 | `dashboard_apnea_backup.png` |
| E2E 지연 관측값 기록 | 실제 UI에서 이벤트 주입 후 경보 표시까지 시간을 관측해야 함 | 수동 기록 표 |
| Git 커밋/푸시 | 현재 sandbox에서 `.git/index.lock` 생성 불가 | 아래 명령으로 직접 수행 |

## 4. 현재 검증의 한계

- 모든 검증은 synthetic signal 기준이다.
- 모든 상태/권고 평가는 controlled scenario 기준이다.
- 실제 60GHz 레이더 하드웨어 검증이 아니다.
- 실제 영유아 데이터 검증이 아니다.
- 실제 마이크/STT/생활 소음 검증이 아니다.
- 실제 IoT 기기 제어 검증이 아니다.
- 의료/진단 목적 성능 검증이 아니다.
- 무호흡/호흡이상은 드문 안전 백업 기능으로만 배치해야 한다.
- 울음은 `is_crying` 감지 신호로만 사용하며, 울음 이유 분류 실사용 성능을 주장하면 안 된다.
- `eval_personalization.py`는 이제 고정 seed(1000+index)를 사용해 재현 가능하다. 결정적 값은 babyA 40.1, babyB 51.7이다. (이전 hash 기반 비결정성 수정됨)

## 5. 보고서/README에서 피해야 할 표현

- 실제 영유아 대상 성능 검증 완료
- 실제 60GHz 레이더 성능 입증
- 의료기기 수준 무호흡 탐지
- 임상 검증 완료
- 진단 정확도 달성
- 실사용 가능한 울음 이유 분류 정확도 달성
- 실제 IoT 자동제어 완료
- 실제 마이크/STT 환경에서 음성 인텐트 100% 달성
- 실제 가정 환경에서 수면/각성 분류 100% 달성

## 6. 사용자가 직접 실행할 Git 명령

```bash
cd /home/user/nuni_run/nuni
git add results/EVERYDAY_PROGRESS_SUMMARY.md
git add results/EVERYDAY_REQUIREMENTS_STATUS.md
git add results/EVERYDAY_FAILURES_AND_LIMITATIONS.md
git add results/*.md
git add results/artifacts/*.csv
git add results/artifacts/*.png 2>/dev/null || true
git restore --staged results/logs 2>/dev/null || true
git restore --staged reports 2>/dev/null || true
git restore --staged cry_model/artifacts 2>/dev/null || true
git restore --staged .venv 2>/dev/null || true
git commit -m "docs: summarize everyday monitoring verification status"
git push origin main
```
