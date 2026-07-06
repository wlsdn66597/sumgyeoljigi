#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/logs results/artifacts

echo "[1/16] compileall"
python -m compileall . | tee results/logs/compileall.log

echo "[2/16] radar synthetic DSP evaluation"
python eval_radar.py | tee results/logs/eval_radar.log

echo "[3/16] radar robustness sweep (SNR x window)"
python eval_radar_robustness.py | tee results/logs/eval_radar_robustness.log

echo "[4/16] fusion controlled scenario evaluation"
python eval_fusion.py | tee results/logs/eval_fusion.log

echo "[5/16] streaming fusion evaluation (debounce/cooldown)"
python eval_stream.py | tee results/logs/eval_stream.log

echo "[6/16] fusion robustness under realistic noise (Monte Carlo)"
python eval_fusion_noisy.py | tee results/logs/eval_fusion_noisy.log

echo "[7/16] realistic radar DSP (range-bin + detrend + band-pass)"
python eval_radar_realistic.py | tee results/logs/eval_radar_realistic.log

echo "[8/16] sleep/wake state classification"
python eval_sleep_state.py | tee results/logs/eval_sleep_state.log

echo "[9/16] personalization (adaptive baseline)"
python eval_personalization.py | tee results/logs/eval_personalization.log

echo "[10/16] proactive environment control"
python eval_env_control.py | tee results/logs/eval_env_control.log

echo "[11/16] cry-context correlation"
python eval_cry_context.py | tee results/logs/eval_cry_context.log

echo "[12/16] sleep rhythm -> routine insight"
python eval_sleep_rhythm.py | tee results/logs/eval_sleep_rhythm.log

echo "[13/16] voice intent evaluation"
python voice_intent.py | tee results/logs/voice_intent.log

echo "[14/16] cloud sleep report"
python cloud.py | tee results/logs/cloud.log

echo "[15/16] dashboard/import smoke"
python - <<'PY' | tee results/logs/dashboard_import_smoke.log
mods = ["streamlit", "topics", "bus", "sim_sensors", "fusion", "workers",
        "state_store", "sleep_state", "personalization", "env_control"]
for mod in mods:
    __import__(mod)
    print(f"[OK] import {mod}")
PY

echo "[16/16] write summary"
python - <<'PY'
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

results = Path("results")
art = results / "artifacts"

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

mae = read_csv(art / "radar_bpm_error_summary.csv")[0]["value"]
fusion = read_csv(art / "fusion_ablation_summary.csv")
full = next(r for r in fusion if r["policy"] == "full_fusion")
single = next(r for r in fusion if r["policy"] == "single_sensor")
radar_only = next(r for r in fusion if r["policy"] == "radar_only")
stream = read_csv(art / "stream_eval_summary.csv")
raw = next(r for r in stream if r["debounce_n"] == "1")
tuned = next(r for r in stream if r["debounce_n"] == "2")
sleep_rows = read_csv(art / "sleep_state_eval.csv")
sleep_acc = sum(int(r["ok"]) for r in sleep_rows) / len(sleep_rows)
env_rows = read_csv(art / "env_control_eval.csv")
env_acc = sum(int(r["ok"]) for r in env_rows) / len(env_rows)
noisy = read_csv(art / "fusion_noisy_summary.csv")
nsingle = next(r for r in noisy if r["policy"] == "single_sensor")
nfusion = next(r for r in noisy if r["policy"] == "full_fusion")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

lines = [
    "# NUNI Software Verification Results",
    "",
    "## Environment",
    "",
    f"- Python: {sys.version.split()[0]}",
    f"- Git commit: `{commit}`",
    f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## Summary",
    "",
    "| Area | Result | Artifact |",
    "|---|---|---|",
    "| Basic validation | compileall success | `results/logs/compileall.log` |",
    f"| Radar DSP | synthetic BPM MAE {mae} breaths/min | `results/RADAR_DSP_RESULT.md` |",
    "| Radar robustness | SNR x window MAE sweep | `results/RADAR_ROBUSTNESS_RESULT.md` |",
    f"| Fusion | full fusion false alarm {full['false_alarm']}, miss {full['miss']} | `results/FUSION_EVALUATION_RESULT.md` |",
    f"| Fusion vs radar-only | radar-only miss {radar_only['miss']} (fusion catches) | `results/artifacts/fusion_ablation_summary.csv` |",
    f"| Single baseline | false alarm {single['false_alarm']}, miss {single['miss']} | `results/artifacts/fusion_ablation_summary.csv` |",
    f"| Streaming (debounce/cooldown) | alerts {raw['total_alerts']}->{tuned['total_alerts']}, apnea latency {tuned['apnea_detect_latency_s']}s | `results/STREAM_EVAL_RESULT.md` |",
    f"| Fusion robustness (noisy MC) | single FA {float(nsingle['false_alarm_rate']):.0%} -> fusion {float(nfusion['false_alarm_rate']):.0%} | `results/FUSION_NOISY_RESULT.md` |",
    "| Realistic radar DSP | naive vs range-bin+detrend+band-pass | `results/RADAR_REALISTIC_RESULT.md` |",
    f"| Sleep/wake state | classification accuracy {sleep_acc:.0%} | `results/SLEEP_STATE_RESULT.md` |",
    "| Personalization | per-household baseline adaptation | `results/PERSONALIZATION_RESULT.md` |",
    f"| Environment control | action match {env_acc:.0%} | `results/ENV_CONTROL_RESULT.md` |",
    "| Cry-context correlation | env/time association (no reason-classification) | `results/CRY_CONTEXT_RESULT.md` |",
    "| Sleep rhythm | multi-night routine insight | `results/SLEEP_RHYTHM_RESULT.md` |",
    "| Voice intent | keyword intent recognition | `results/VOICE_INTENT_RESULT.md` |",
    "| Cloud sleep report | overnight summary + personalized range | `results/SLEEP_REPORT.md` |",
    "| Dashboard | import smoke success | `results/logs/dashboard_import_smoke.log` |",
    "| Cry ML | limited reason-classifier (class imbalance) | `results/CRY_CLASSIFICATION_LIMITATION.md` |",
    "",
    "## Positioning",
    "",
    "- Main tasks: everyday non-contact monitoring (sleep/wake state, cry detection,",
    "  environment) + proactive control + personalization.",
    "- Apnea/breathing-abnormality is a rare safety backup, not the headline, and is not",
    "  a medical-grade claim.",
    "",
    "## Limits",
    "",
    "- All results are synthetic/controlled software validation, not real sensors/infants.",
    "- Cry reason classification stays experimental (class imbalance).",
    "- Environment control emits recommendations; real actuation is a 2nd-round hardware step.",
]
(results / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote results/RESULTS.md")
PY

echo "run_all complete"
