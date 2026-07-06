# Changes After Review

## 코드 수정

- `radar_sim_signal.py` 추가: 실제 레이더 없이 synthetic raw radar-like signal을 생성하기 위해 추가했다.
- `radar_dsp.py` 추가: synthetic signal 기준 FFT BPM 추정, 4초 RMS 기반 apnea 감지, 2초 high-frequency energy 기반 motion 감지를 구현했다.
- `eval_radar.py` 추가: synthetic BPM estimation, apnea smoke test, motion smoke test를 실행하고 결과 CSV/Markdown을 생성한다.
- `sim_sensors.py` 수정: BPM 숫자를 직접 생성하던 레이더 시뮬레이션을 synthetic raw buffer 생성 후 `process_radar_buffer()`로 처리하는 경로로 변경했다. 기존 `topics.radar_msg()` schema는 유지했다.
- `fusion.py` 수정: `apnea` 필드 호환, live subscriber debounce 2회, cooldown 10초를 추가했다. 기존 `decide()`와 `single_sensor_decide()` public 함수는 유지했다.
- `eval_fusion.py` 수정: 단일 센서 baseline, radar/audio/env ablation, full fusion을 controlled scenarios에서 평가하고 CSV/Markdown을 생성하도록 확장했다.
- `run_all.sh` 추가: compileall, radar 평가, fusion 평가, dashboard/import smoke, `results/RESULTS.md` 생성을 한 번에 수행한다.

## 이유

이전 문서에는 레이더 DSP와 `run_all.sh` 결과가 존재한다고 가정되어 있었지만 현재 checkout에는 해당 파일이 없었다. 이번 변경은 그 불일치를 해소하고, 현재 checkout에서 실제 실행 가능한 synthetic-only 검증 경로를 만들기 위한 최소 구현이다.

## 생성/갱신한 문서와 산출물

- `results/CODE_REVIEW_SUMMARY.md`
- `results/RUN_ALL_STATUS.md`
- `results/RADAR_DSP_RESULT.md`
- `results/FUSION_EVALUATION_RESULT.md`
- `results/CRY_CLASSIFICATION_LIMITATION.md`
- `results/RESULTS_FINAL.md`
- `results/artifacts/fusion_eval_summary.csv`
- `results/artifacts/fusion_ablation_summary.csv`
- `results/artifacts/fusion_ablation_bar.png`

## 주의

레이더 DSP 결과는 synthetic raw signal 기준이다. 실제 60GHz 레이더 하드웨어, 실제 영유아, 실환경 무호흡 감지 성능으로 해석하면 안 된다.
