# Fusion Evaluation Result

- Controlled scenarios: 11
- Metrics are computed from this script at runtime.

## Policy Summary

| policy | false alarm | miss |
|---|---:|---:|
| single_sensor | 6 | 0 |
| radar_only | 0 | 2 |
| audio_only | 2 | 2 |
| env_only | 2 | 3 |
| full_fusion | 0 | 0 |

## Interpretation

Across these scenarios the full fusion policy is the only one with 0 false alarms AND 0 misses. Radar alone catches clear apnea but misses 'borderline' abnormal breathing that becomes actionable only when corroborated by cry or environment (cross-validation). Single-sensor policies over-alert on isolated signals. This shows fusion's value is both fewer false alarms than a naive single-signal policy and fewer misses than any single modality.

## Limits

- This is a controlled software scenario evaluation, not a real sensor or clinical test.
- Debounce/cooldown behavior is implemented in the live `Fusion` subscriber, not in this stateless scenario table.
