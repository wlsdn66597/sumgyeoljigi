# Radar DSP Result

## Scope

This result is based on synthetic raw radar-like signals only. It is not a real 60GHz radar hardware test.

## BPM Estimation

| true BPM | predicted BPM | abs error |
|---:|---:|---:|
| 20 | 20.0 | 0.0 |
| 25 | 26.0 | 1.0 |
| 30 | 30.0 | 0.0 |
| 35 | 34.0 | 1.0 |
| 40 | 40.0 | 0.0 |
| 45 | 44.0 | 1.0 |

- MAE: 0.500 breaths/min
- Conditions: duration 30s, fs 50Hz, SNR 15dB, BPM cases 20/25/30/35/40/45
- Method: FFT peak search within 15-70 BPM.

## Apnea Smoke Test

- Synthetic apnea detected: True
- Apnea start: 6s
- Detection time: 10.0s
- Detection delay: 4.0s
- Method: recent 4s RMS compared to earlier baseline RMS.

## Motion Smoke Test

- Synthetic motion detected: True
- Method: recent 2s high-frequency energy compared to earlier baseline.

## Report-Safe Sentence

Synthetic raw signal evaluation showed FFT-based breathing estimation MAE 0.500 breaths/min under 30s, 50Hz, SNR 15dB conditions.

## Do Not Claim

- Do not claim real radar hardware performance.
- Do not claim real infant apnea detection.
- Do not claim real-world motion robustness.
