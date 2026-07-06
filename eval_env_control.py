"""선제 환경 제어 로직 평가 (환경 상황 → 기대 동작 일치율)."""
import csv
from pathlib import Path

from env_control import recommend

RESULTS = Path("results")
ART = RESULTS / "artifacts"

# (이름, env, sleep_state, 기대 action 집합)
SCENARIOS = [
    ("comfort", {"co2": 600, "humidity": 45, "temp": 22, "lux": 10}, "calm_sleep", set()),
    ("co2_high", {"co2": 1200, "humidity": 45, "temp": 22, "lux": 10}, "calm_sleep", {"ventilate"}),
    ("dry", {"co2": 600, "humidity": 28, "temp": 22, "lux": 10}, "calm_sleep", {"humidify"}),
    ("hot", {"co2": 600, "humidity": 45, "temp": 28, "lux": 10}, "awake", {"cool"}),
    ("cold", {"co2": 600, "humidity": 45, "temp": 16, "lux": 10}, "calm_sleep", {"heat"}),
    ("bright_sleep", {"co2": 600, "humidity": 45, "temp": 22, "lux": 80}, "calm_sleep", {"dim_light"}),
    ("bright_awake", {"co2": 600, "humidity": 45, "temp": 22, "lux": 80}, "awake", set()),
    ("multi", {"co2": 1100, "humidity": 30, "temp": 27, "lux": 90}, "restless",
     {"ventilate", "humidify", "cool", "dim_light"}),
]


def main():
    RESULTS.mkdir(exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    rows, correct = [], 0
    for name, env, ss, expected in SCENARIOS:
        got = {a for a, _ in recommend(env, ss)}
        ok = got == expected
        correct += ok
        rows.append({"scenario": name, "expected": "|".join(sorted(expected)) or "-",
                     "got": "|".join(sorted(got)) or "-", "ok": int(ok)})
    acc = correct / len(SCENARIOS)

    with (ART / "env_control_eval.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "expected", "got", "ok"])
        w.writeheader(); w.writerows(rows)

    print("scenario,expected,got,ok")
    for r in rows:
        print(f"{r['scenario']},{r['expected']},{r['got']},{r['ok']}")
    print(f"accuracy,{acc:.3f}")

    md = ["# Proactive Environment Control", "",
          f"- 통제 시나리오 {len(SCENARIOS)}개, 동작 일치율 {acc:.1%}",
          "- 입력: 환경 센서값 + 수면 상태. 울음 이유 분류에 의존하지 않음.", "",
          "| 시나리오 | 기대 동작 | 산출 동작 | 결과 |", "|---|---|---|---|"]
    md += [f"| {r['scenario']} | {r['expected']} | {r['got']} | {'O' if r['ok'] else 'X'} |" for r in rows]
    md += ["", "## 한계", "", "- 규칙 기반 권고이며 실제 가전 구동/실환경 검증은 2차 하드웨어 단계."]
    (RESULTS / "ENV_CONTROL_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
