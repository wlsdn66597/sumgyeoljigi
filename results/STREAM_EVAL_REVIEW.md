# Stream Evaluation Review

실행 명령: `source .venv/bin/activate && python eval_stream.py`

## 결과

| config | total alerts | apnea detect latency (s) | transient false alerts |
|---|---:|---:|---:|
| raw (debounce=1, cooldown=0) | 4 | 0 | 3 |
| tuned (debounce=2, cooldown=10) | 1 | 1 | 0 |

## 확인

- 디바운스/쿨다운 적용 후 총 알림 수는 4에서 1로 감소했다.
- 1초 transient false alert는 3에서 0으로 감소했다.
- sustained apnea는 계속 감지됐고, tuned 설정에서 latency는 1초였다.

## 한계

- synthetic 100초 timeline 기반이다.
- 실제 센서 지연, MQTT 브로커, Streamlit 렌더링, 실제 보호자 알림 채널 지연은 포함하지 않는다.
