# Dashboard Demo Checklist

## 자동 smoke test 결과

| 항목 | 결과 |
|---|---|
| dashboard import smoke | 성공 |
| Streamlit server startup | 실패 |
| 실패 원인 | sandbox에서 socket 생성이 `PermissionError: [Errno 1] Operation not permitted`로 차단됨 |
| 로그 | `results/logs/streamlit_startup_latest.log` |

## 수동 캡처해야 할 화면

1. 정상 상태
2. 울음 주입 상태
3. 무호흡 주입 상태

## 각 화면에서 확인할 것

- 현재 상태 표시
- 센서 값 변화
- 경보/주의 표시
- 이벤트 발생 후 경보까지 걸린 시간

## 기록 양식

| 시나리오 | 스크린샷 파일명 | 이벤트→경보 시간 | 비고 |
|---|---|---:|---|
| 정상 상태 | `dashboard_normal.png` | - | 수동 캡처 |
| 울음 주입 상태 | `dashboard_cry.png` | 직접 기록 | 사이드바 울음 주입 |
| 무호흡 주입 상태 | `dashboard_apnea.png` | 직접 기록 | 사이드바 무호흡 주입 |

## 수동 실행 안내

현재 sandbox에서는 서버 socket 생성이 차단되어 자동 스크린샷을 만들 수 없었다. 로컬 환경에서는 다음 명령으로 실행한다.

```bash
cd /home/user/nuni_run/nuni
source .venv/bin/activate
streamlit run app.py
```
