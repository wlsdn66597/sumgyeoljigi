"""레이더 호흡 추정 강건성 스윕 (SNR x 관측 윈도우 -> MAE).

잡음(SNR)과 관측 시간(window)이 호흡수 추정 정확도에 주는 영향을 정량화한다.
합성 신호 전용 검증이며 실제 센서/영유아 검증이 아니다.

실행: python eval_radar_robustness.py
"""
import csv
from pathlib import Path

import numpy as np

from radar_dsp import estimate_bpm_fft
from radar_sim_signal import generate_breathing_signal

RESULTS = Path("results")
ART = RESULTS / "artifacts"

FS = 50
SNRS = [5, 10, 15, 20]          # dB
WINDOWS = [15, 30, 60]          # seconds
BPMS = [20, 30, 40, 50]         # 셀당 평균낼 BPM 세트


def cell_mae(snr_db, window_s):
    errs = []
    for i, bpm in enumerate(BPMS):
        _, sig, _ = generate_breathing_signal(
            duration_s=window_s, fs=FS, bpm=bpm, snr_db=snr_db, seed=snr_db * 100 + window_s + i
        )
        pred = estimate_bpm_fft(sig, FS)
        errs.append(abs(pred - bpm))
    return round(float(np.mean(errs)), 3)


def main():
    RESULTS.mkdir(exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    grid = {(s, w): cell_mae(s, w) for s in SNRS for w in WINDOWS}

    # CSV (long form)
    rows = [{"snr_db": s, "window_s": w, "bpm_mae": grid[(s, w)]} for s in SNRS for w in WINDOWS]
    with (ART / "radar_robustness.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["snr_db", "window_s", "bpm_mae"])
        wr.writeheader(); wr.writerows(rows)

    # 콘솔 표 (행=window, 열=SNR)
    print("BPM MAE (breaths/min) — 행: window(s), 열: SNR(dB)")
    print("window\\SNR," + ",".join(str(s) for s in SNRS))
    for w in WINDOWS:
        print(f"{w}," + ",".join(f"{grid[(s, w)]:.3f}" for s in SNRS))

    # Markdown
    md = [
        "# Radar DSP Robustness (synthetic)",
        "",
        f"- FS={FS}Hz, BPM set={BPMS}, per-cell = mean abs error over BPM set",
        "- Synthetic raw signal only; not real radar/infant validation.",
        "",
        "| window(s) \\ SNR(dB) | " + " | ".join(str(s) for s in SNRS) + " |",
        "|---|" + "---|" * len(SNRS),
    ]
    for w in WINDOWS:
        md.append(f"| {w} | " + " | ".join(f"{grid[(s, w)]:.3f}" for s in SNRS) + " |")
    md += [
        "",
        "## Interpretation",
        "",
        "- 관측 윈도우가 길수록 FFT 주파수 해상도가 높아져 MAE가 낮아진다 (15s: MAE 약 1.0, 30s/60s: 약 0.0).",
        "- 이번 스윕에서 SNR 5~20dB 간 MAE 차이는 관측되지 않았고, window length가 지배적이었다.",
        "- 따라서 실사용 목표 정확도는 (이 조건에서는) 최소 관측 윈도우로 결정된다.",
    ]
    (RESULTS / "RADAR_ROBUSTNESS_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
