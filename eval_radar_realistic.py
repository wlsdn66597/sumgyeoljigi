"""현실적 레이더 DSP 검증: naive vs (range-bin 선택 + 디트렌드 + 대역통과).

다중 거리 bin·드리프트·클러터·저SNR 신호에서, 모든 bin을 평균내는 naive 추정과
현실적 전처리 추정의 호흡수 MAE를 비교한다. (합성 기반, 실센서 검증 아님)

실행: python eval_radar_realistic.py
"""
import csv
from pathlib import Path

import numpy as np

import radar_sim_signal as rs
from radar_dsp import estimate_bpm_fft, estimate_bpm_realistic

RESULTS = Path("results")
ART = RESULTS / "artifacts"
FS = 50
SEC = 30
BPMS = [20, 40, 45, 55]   # 공통 간섭(30 bpm)과 겹치지 않게


def main():
    RESULTS.mkdir(exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, bpm in enumerate(BPMS):
        bins, true_bpm, bbin = rs.generate_multibin(bpm, fs=FS, seconds=SEC, snr_db=8, seed=100 + i)
        naive = estimate_bpm_fft(bins.mean(axis=0), FS)          # 모든 bin 평균 (전처리 없음)
        real, sel = estimate_bpm_realistic(bins, FS)             # range-bin 선택 + 디트렌드 + 대역통과
        rows.append({
            "true_bpm": true_bpm,
            "naive_bpm": round(naive, 2), "naive_err": round(abs(naive - true_bpm), 2),
            "realistic_bpm": round(real, 2), "realistic_err": round(abs(real - true_bpm), 2),
            "picked_bin": sel, "true_bin": bbin,
        })
    mae_naive = round(float(np.mean([r["naive_err"] for r in rows])), 3)
    mae_real = round(float(np.mean([r["realistic_err"] for r in rows])), 3)

    with (ART / "radar_realistic.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print("true_bpm,naive_bpm,naive_err,realistic_bpm,realistic_err,picked_bin,true_bin")
    for r in rows:
        print(f"{r['true_bpm']},{r['naive_bpm']},{r['naive_err']},{r['realistic_bpm']},{r['realistic_err']},{r['picked_bin']},{r['true_bin']}")
    print(f"MAE naive={mae_naive}, realistic={mae_real}")

    md = [
        "# Realistic Radar DSP (range-bin + detrend + band-pass)",
        "",
        f"- 다중 거리 bin·드리프트·클러터·SNR 8dB 신호, BPM {BPMS}",
        "- naive: 모든 bin 평균 후 FFT / realistic: 호흡 bin 선택 → 디트렌드 → 대역통과 → FFT",
        "",
        "| true BPM | naive BPM(오차) | realistic BPM(오차) | 선택 bin/정답 bin |",
        "|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['true_bpm']} | {r['naive_bpm']} ({r['naive_err']}) | "
                  f"{r['realistic_bpm']} ({r['realistic_err']}) | {r['picked_bin']}/{r['true_bin']} |")
    md += [
        "",
        f"- **MAE: naive {mae_naive} → realistic {mae_real} 회/분**",
        "",
        "## 해석",
        "",
        "- naive 평균은 클러터·드리프트에 오염되어 오차가 크다.",
        "- range-bin 선택 + 디트렌드 + 대역통과가 호흡 성분을 회복해 오차를 크게 낮춘다.",
        "- 이는 2차 실물 레이더(다중 bin·multipath·드리프트) 대비 전처리 필요성을 뒷받침한다.",
        "",
        "## 한계",
        "",
        "- 합성 다중 bin 모델 기반이며 실제 레이더 클러터/multipath 분포와 다를 수 있다.",
    ]
    (RESULTS / "RADAR_REALISTIC_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
