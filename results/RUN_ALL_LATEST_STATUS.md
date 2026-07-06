# run_all.sh Latest Status

실행 명령: `source .venv/bin/activate && export MPLCONFIGDIR=/tmp/matplotlib-nuni && bash run_all.sh`

| 단계 | 실행 여부 | 성공 여부 | 로그/산출물 |
|---|---|---|---|
| `[1/9] compileall` | 실행 | 성공 | `results/logs/run_all_latest.log`, `results/logs/compileall.log` |
| `[2/9] radar synthetic DSP evaluation` | 실행 | 성공 | `results/logs/eval_radar.log`, `results/RADAR_DSP_RESULT.md` |
| `[3/9] radar robustness sweep` | 실행 | 성공 | `results/logs/eval_radar_robustness.log`, `results/RADAR_ROBUSTNESS_RESULT.md` |
| `[4/9] fusion controlled scenario evaluation` | 실행 | 성공 | `results/logs/eval_fusion.log`, `results/FUSION_EVALUATION_RESULT.md` |
| `[5/9] streaming fusion evaluation` | 실행 | 성공 | `results/logs/eval_stream.log`, `results/STREAM_EVAL_RESULT.md` |
| `[6/9] voice intent evaluation` | 실행 | 성공 | `results/logs/voice_intent.log`, `results/VOICE_INTENT_RESULT.md` |
| `[7/9] cloud sleep report` | 실행 | 성공 | `results/logs/cloud.log`, `results/SLEEP_REPORT.md` |
| `[8/9] dashboard/import smoke` | 실행 | 성공 | `results/logs/dashboard_import_smoke.log` |
| `[9/9] write summary` | 실행 | 성공 | `results/RESULTS.md` |

## 확인

- `run_all.sh`는 9단계로 구성되어 있다.
- compileall, eval_radar, eval_radar_robustness, eval_fusion, eval_stream, voice_intent, cloud report, dashboard import smoke를 모두 포함한다.
- 최종 `results/RESULTS.md`가 생성됐다.
