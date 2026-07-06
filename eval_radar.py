"""Evaluate synthetic raw-signal radar DSP.

This evaluation uses generated synthetic signals only. It is not a real radar
hardware test and must not be reported as such.
"""
import csv
from pathlib import Path

from radar_dsp import estimate_bpm_fft, detect_apnea, detect_motion, process_radar_buffer
from radar_sim_signal import generate_breathing_signal, generate_eval_cases


RESULTS = Path("results")
ART = RESULTS / "artifacts"


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def eval_bpm():
    rows = []
    for case in generate_eval_cases():
        _, sig, meta = generate_breathing_signal(**case)
        pred = estimate_bpm_fft(sig, meta["fs"])
        err = abs(pred - meta["true_bpm"])
        rows.append({
            "true_bpm": meta["true_bpm"],
            "pred_bpm": round(pred, 3),
            "abs_error": round(err, 3),
            "duration_s": meta["duration_s"],
            "fs": meta["fs"],
            "snr_db": meta["snr_db"],
        })
    mae = sum(r["abs_error"] for r in rows) / len(rows)
    return rows, mae


def eval_apnea():
    fs = 50
    duration_s = 12
    apnea_start = 6
    apnea_duration = 5
    _, sig, _ = generate_breathing_signal(
        duration_s=duration_s,
        fs=fs,
        bpm=30,
        snr_db=15,
        apnea_start=apnea_start,
        apnea_duration=apnea_duration,
        seed=100,
    )
    detected_at = None
    step = int(0.5 * fs)
    min_len = int(4.0 * fs)
    for end in range(min_len, len(sig) + 1, step):
        if detect_apnea(sig[:end], fs, window_s=4.0):
            detected_at = end / fs
            break
    delay = None if detected_at is None else max(0.0, detected_at - apnea_start)
    return {
        "detected": detected_at is not None,
        "detected_at_s": None if detected_at is None else round(detected_at, 3),
        "delay_s": None if delay is None else round(delay, 3),
        "apnea_start_s": apnea_start,
        "apnea_duration_s": apnea_duration,
    }


def eval_motion():
    fs = 50
    _, sig, _ = generate_breathing_signal(duration_s=12, fs=fs, bpm=30, snr_db=15, motion=True, seed=200)
    return {"detected": detect_motion(sig, fs, window_s=2.0)}


def write_markdown(bpm_rows, mae, apnea, motion):
    summary_rows = [{"metric": "bpm_mae", "value": round(mae, 3), "unit": "breaths_per_min"}]
    _write_csv(ART / "radar_bpm_estimation.csv", bpm_rows, ["true_bpm", "pred_bpm", "abs_error", "duration_s", "fs", "snr_db"])
    _write_csv(ART / "radar_bpm_error_summary.csv", summary_rows, ["metric", "value", "unit"])

    lines = [
        "# Radar DSP Result",
        "",
        "## Scope",
        "",
        "This result is based on synthetic raw radar-like signals only. It is not a real 60GHz radar hardware test.",
        "",
        "## BPM Estimation",
        "",
        "| true BPM | predicted BPM | abs error |",
        "|---:|---:|---:|",
    ]
    for row in bpm_rows:
        lines.append(f"| {row['true_bpm']} | {row['pred_bpm']} | {row['abs_error']} |")
    lines += [
        "",
        f"- MAE: {mae:.3f} breaths/min",
        "- Conditions: duration 30s, fs 50Hz, SNR 15dB, BPM cases 20/25/30/35/40/45",
        "- Method: FFT peak search within 15-70 BPM.",
        "",
        "## Apnea Smoke Test",
        "",
        f"- Synthetic apnea detected: {apnea['detected']}",
        f"- Apnea start: {apnea['apnea_start_s']}s",
        f"- Detection time: {apnea['detected_at_s']}s",
        f"- Detection delay: {apnea['delay_s']}s",
        "- Method: recent 4s RMS compared to earlier baseline RMS.",
        "",
        "## Motion Smoke Test",
        "",
        f"- Synthetic motion detected: {motion['detected']}",
        "- Method: recent 2s high-frequency energy compared to earlier baseline.",
        "",
        "## Report-Safe Sentence",
        "",
        f"Synthetic raw signal evaluation showed FFT-based breathing estimation MAE {mae:.3f} breaths/min under 30s, 50Hz, SNR 15dB conditions.",
        "",
        "## Do Not Claim",
        "",
        "- Do not claim real radar hardware performance.",
        "- Do not claim real infant apnea detection.",
        "- Do not claim real-world motion robustness.",
    ]
    (RESULTS / "RADAR_DSP_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RESULTS.mkdir(exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    bpm_rows, mae = eval_bpm()
    apnea = eval_apnea()
    motion = eval_motion()
    write_markdown(bpm_rows, mae, apnea, motion)

    print("BPM estimation (synthetic raw signal)")
    print("true_bpm,pred_bpm,abs_error")
    for row in bpm_rows:
        print(f"{row['true_bpm']},{row['pred_bpm']},{row['abs_error']}")
    print(f"MAE,{mae:.3f}")
    print(f"apnea_detected,{apnea['detected']},detected_at_s,{apnea['detected_at_s']},delay_s,{apnea['delay_s']}")
    print(f"motion_detected,{motion['detected']}")


if __name__ == "__main__":
    main()
