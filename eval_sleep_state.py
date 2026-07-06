"""수면/각성 상태 분류 평가 (통제 시나리오 기준 정확도)."""
import csv
from pathlib import Path

from sleep_state import classify

RESULTS = Path("results")
ART = RESULTS / "artifacts"

# (이름, radar, cry, 기대 상태)
SCENARIOS = [
    ("calm_sleep", {"presence": True, "movement": 0.1}, {"is_crying": False}, "calm_sleep"),
    ("low_move", {"presence": True, "movement": 0.3}, {"is_crying": False}, "calm_sleep"),
    ("restless", {"presence": True, "movement": 0.5}, {"is_crying": False}, "restless"),
    ("restless_high", {"presence": True, "movement": 0.65}, {"is_crying": False}, "restless"),
    ("awake_move", {"presence": True, "movement": 0.85}, {"is_crying": False}, "awake"),
    ("awake_cry", {"presence": True, "movement": 0.2}, {"is_crying": True}, "awake"),
    ("absent", {"presence": False, "movement": 0.0}, {"is_crying": False}, "unknown"),
]


def main():
    RESULTS.mkdir(exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    rows, correct = [], 0
    for name, r, c, expected in SCENARIOS:
        pred, _ = classify(r, c)
        ok = pred == expected
        correct += ok
        rows.append({"scenario": name, "expected": expected, "pred": pred, "ok": int(ok)})
    acc = correct / len(SCENARIOS)

    with (ART / "sleep_state_eval.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "expected", "pred", "ok"])
        w.writeheader(); w.writerows(rows)

    print("scenario,expected,pred,ok")
    for r in rows:
        print(f"{r['scenario']},{r['expected']},{r['pred']},{r['ok']}")
    print(f"accuracy,{acc:.3f}")

    md = ["# Sleep/Wake State Classification", "",
          f"- 통제 시나리오 {len(SCENARIOS)}개, 정확도 {acc:.1%}",
          "- 입력: 레이더 움직임 + 울음 감지(is_crying). 울음 이유 분류에 의존하지 않음.", "",
          "| 시나리오 | 기대 | 예측 | 결과 |", "|---|---|---|---|"]
    md += [f"| {r['scenario']} | {r['expected']} | {r['pred']} | {'O' if r['ok'] else 'X'} |" for r in rows]
    md += ["", "## 한계", "", "- 합성/통제 시나리오 기준이며 실제 영유아 수면 검증이 아니다."]
    (RESULTS / "SLEEP_STATE_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
