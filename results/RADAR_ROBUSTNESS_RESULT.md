# Radar DSP Robustness (synthetic)

- FS=50Hz, BPM set=[20, 30, 40, 50], per-cell = mean abs error over BPM set
- Synthetic raw signal only; not real radar/infant validation.

| window(s) \ SNR(dB) | 5 | 10 | 15 | 20 |
|---|---|---|---|---|
| 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| 30 | 0.000 | 0.000 | 0.000 | 0.000 |
| 60 | 0.000 | 0.000 | 0.000 | 0.000 |

## Interpretation

- 관측 윈도우가 길수록 FFT 주파수 해상도가 높아져 MAE가 낮아진다 (15s: MAE 약 1.0, 30s/60s: 약 0.0).
- 이번 스윕에서 SNR 5~20dB 간 MAE 차이는 관측되지 않았고, window length가 지배적이었다.
- 따라서 실사용 목표 정확도는 (이 조건에서는) 최소 관측 윈도우로 결정된다.
