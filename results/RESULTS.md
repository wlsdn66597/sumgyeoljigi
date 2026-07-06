# NUNI Software Verification Results

## Environment

- Python: 3.9.7
- Git commit: `2f498af7ce6d9ab223557e35677d01b29c275cd9`
- Generated at: 2026-07-02 17:09:53

## Summary

| Area | Result | Artifact |
|---|---|---|
| Basic validation | compileall success | `results/logs/compileall.log` |
| Radar DSP | synthetic BPM MAE 0.5 breaths/min | `results/RADAR_DSP_RESULT.md` |
| Radar robustness | SNR x window MAE sweep | `results/RADAR_ROBUSTNESS_RESULT.md` |
| Realistic radar DSP | naive vs range-bin+detrend+band-pass | `results/RADAR_REALISTIC_RESULT.md` |
| Fusion | full 0/0 vs single 6/0 vs radar-only 0/2 | `results/FUSION_EVALUATION_RESULT.md` |
| Fusion robustness (noisy MC) | single FA 58% -> fusion 3% | `results/FUSION_NOISY_RESULT.md` |
| Streaming (debounce/cooldown) | alerts 4->1, apnea latency 1s | `results/STREAM_EVAL_RESULT.md` |
| Sleep/wake state | accuracy 100% | `results/SLEEP_STATE_RESULT.md` |
| Personalization | per-household baseline adaptation | `results/PERSONALIZATION_RESULT.md` |
| Environment control | action match 100% | `results/ENV_CONTROL_RESULT.md` |
| Cry-context correlation | env/time association (no reason-classification) | `results/CRY_CONTEXT_RESULT.md` |
| Sleep rhythm | multi-night routine insight | `results/SLEEP_RHYTHM_RESULT.md` |
| Voice intent | keyword intent recognition | `results/VOICE_INTENT_RESULT.md` |
| Cloud sleep report | overnight summary + cry-context + personalized range | `results/SLEEP_REPORT.md` |
| Cry ML | limited reason-classifier (class imbalance) | `results/CRY_CLASSIFICATION_LIMITATION.md` |

## Limits

- All results are synthetic/controlled software validation, not real sensors/infants.
- Fusion noisy-MC uses a synthetic noise model; real sensor noise will differ.
- Realistic radar DSP uses a synthetic multi-bin/clutter model.
- Cry-context/sleep-rhythm use synthetic correlated data; not causal proof on real infants.
