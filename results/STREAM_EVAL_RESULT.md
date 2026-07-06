# Streaming Fusion Evaluation (debounce/cooldown)

- Synthetic timeline: 100s, sustained apnea 30-45s, transient 1s blips at [60, 65, 70]

| config | total alerts | apnea detect latency (s) | transient false alerts |
|---|---:|---:|---:|
| raw (debounce=1, cooldown=0) | 4 | 0 | 3 |
| tuned (debounce=2, cooldown=10) | 1 | 1 | 0 |

## Interpretation

- Tuned debounce suppresses 1-second transient artifacts that the raw policy would alert on.
- Cooldown collapses repeated alerts of the same sustained event into fewer notifications.
- Sustained apnea is still detected with a small latency (debounce window).

## Limits

- Synthetic timeline only; not a real sensor or clinical validation.
