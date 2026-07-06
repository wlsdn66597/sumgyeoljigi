# Code Review Summary

## 검토 대상

- `radar_dsp.py`
- `radar_sim_signal.py`
- `eval_radar.py`
- `sim_sensors.py`
- `fusion.py`
- `eval_fusion.py`
- `run_all.sh`

## 결과

| 항목 | 확인 결과 |
|---|---|
| `radar_dsp.py` FFT 범위 | `estimate_bpm_fft()`가 15~70 BPM 범위만 peak search |
| sampling/window/BPM 변환 | `fs` 기반 `rfftfreq`, peak Hz * 60으로 BPM 변환 |
| apnea 판단 | 최근 4초 RMS를 이전 baseline RMS와 비교하는 threshold 방식 |
| motion 판단 | 최근 2초 high-frequency energy를 이전 baseline과 비교 |
| `sim_sensors.py` | synthetic raw buffer 생성 후 `process_radar_buffer()`를 통과시켜 기존 radar message schema로 발행 |
| `fusion.py` debounce | live `Fusion` subscriber에서 non-normal 상태 2회 연속 시 확정 |
| `fusion.py` cooldown | 동일 alert reason 반복 발행을 10초간 억제 |
| `eval_fusion.py` ablation | `single_sensor`, `radar_only`, `audio_only`, `env_only`, `full_fusion` 비교 |
| `run_all.sh` | `set -euo pipefail`, 단계별 echo, logs/artifacts 생성 |

## 주의

- Radar DSP는 synthetic raw signal 전용 검증 코드다.
- `eval_fusion.py`는 stateless policy evaluation이므로 live debounce/cooldown 시간 동작을 직접 측정하지 않는다.
- 실제 센서 하드웨어, 실제 영유아, 실환경 마이크 성능은 검증하지 않는다.
