# Realistic Radar DSP (range-bin + detrend + band-pass)

- 다중 거리 bin·드리프트·클러터·SNR 8dB 신호, BPM [20, 40, 45, 55]
- naive: 모든 bin 평균 후 FFT / realistic: 호흡 bin 선택 → 디트렌드 → 대역통과 → FFT

| true BPM | naive BPM(오차) | realistic BPM(오차) | 선택 bin/정답 bin |
|---:|---:|---:|---:|
| 20 | 30.0 (10.0) | 20.0 (0.0) | 3/3 |
| 40 | 30.0 (10.0) | 40.0 (0.0) | 3/3 |
| 45 | 30.0 (15.0) | 44.0 (1.0) | 3/3 |
| 55 | 30.0 (25.0) | 54.0 (1.0) | 3/3 |

- **MAE: naive 15.0 → realistic 0.5 회/분**

## 해석

- naive 평균은 클러터·드리프트에 오염되어 오차가 크다.
- range-bin 선택 + 디트렌드 + 대역통과가 호흡 성분을 회복해 오차를 크게 낮춘다.
- 이는 2차 실물 레이더(다중 bin·multipath·드리프트) 대비 전처리 필요성을 뒷받침한다.

## 한계

- 합성 다중 bin 모델 기반이며 실제 레이더 클러터/multipath 분포와 다를 수 있다.
