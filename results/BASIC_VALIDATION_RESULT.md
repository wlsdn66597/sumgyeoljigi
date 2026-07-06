# Basic Validation Result

## Git 상태

| 항목 | 결과 |
|---|---|
| `git fetch origin` | 실패: `.git/FETCH_HEAD` 읽기 전용 파일 시스템 |
| 현재 브랜치 | `main` |
| 최신 로컬 커밋 | `e813f22 feat: 레이더 강건성 스윕 추가 + 실험 가이드 실행 가능화` |
| 처리 | 사용자 지시에 따라 현재 checkout 기준으로 계속 진행 |

## 가상환경 및 의존성

| 단계 | 결과 | 로그 |
|---|---|---|
| `python3 -m venv .venv` | 실패: `ensurepip` 없음 | `results/logs/venv_create_attempt.log` |
| `virtualenv .venv` | 성공 | `results/logs/virtualenv_create_attempt.log` |
| isolated `.venv` requirements 설치 | 실패: PyPI DNS 조회 불가 | `results/logs/install_requirements.log` |
| `virtualenv --system-site-packages .venv` | 성공 | `results/logs/virtualenv_system_site_create.log` |
| system-site 기반 dependency import | 성공: `numpy`, `streamlit`, `paho.mqtt.client` | `results/logs/import_dependency_check.log` |
| Python | 3.10.12 | `results/logs/python_version.log` |

## Compileall

| 검증 | 결과 | 로그 |
|---|---|---|
| `python -m compileall .` | 성공 | `results/logs/compileall_latest.log` |
| `run_all.sh` 내부 compileall | 성공 | `results/logs/run_all_latest.log` |

## 한계

- 네트워크 제한으로 PyPI 신규 다운로드는 불가했다.
- 실험은 system-site packages를 볼 수 있는 `.venv`에서 실행했다.
- `compileall .`은 `.venv`도 순회하므로 로그가 크다.
