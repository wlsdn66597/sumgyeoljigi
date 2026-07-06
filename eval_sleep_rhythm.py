"""수면 리듬 루틴 인사이트 검증.

새벽 특정 시간대에 자주 깨는 패턴을 심은 합성 5박 데이터에서 analyze()가 리듬
인사이트(자주 깨는 시간대 등)를 도출하는지 확인한다. (합성 기반)

실행: python eval_sleep_rhythm.py
"""
from pathlib import Path
import numpy as np

import sleep_rhythm

RESULTS = Path("results")


def make_night(seed, hours=8):
    rng = np.random.default_rng(seed)
    mins = hours * 60
    states = []
    for m in range(mins):
        h = m // 60
        p_awake = 0.01
        if h in (2, 3):          # 새벽 2~3시 자주 깸
            p_awake = 0.12
        r = rng.random()
        if r < p_awake:
            states.append("awake")
        elif r < p_awake + 0.10:
            states.append("restless")
        else:
            states.append("calm_sleep")
    return states


def main():
    RESULTS.mkdir(exist_ok=True)
    nights = [make_night(seed=i) for i in range(5)]
    out = sleep_rhythm.analyze(nights)

    print(f"nights={out['nights']}, avg_awakenings={out['avg_awakenings']}, "
          f"avg_longest_calm_min={out['avg_longest_calm_min']}")
    print("state_proportion:", out["state_proportion"])
    print("wake_hour_hist:", out["wake_hour_hist"])
    print("insights:")
    for s in out["insights"]:
        print("  -", s)

    md = ["# Sleep Rhythm → Routine Insight", "",
          f"- 합성 {out['nights']}박(새벽 2~3시 각성↑) 기준", "",
          f"- 밤 평균 각성: {out['avg_awakenings']}회",
          f"- 최장 안정수면 평균: {out['avg_longest_calm_min']}분",
          f"- 상태 비율: {out['state_proportion']}",
          f"- 자주 깨는 시간대 히스토그램: {out['wake_hour_hist']}",
          "", "## 도출된 루틴 인사이트", ""]
    md += [f"- {s}" for s in out["insights"]]
    md += ["", "## 의의", "",
           "- 개인화 baseline 위에, 여러 밤 데이터로 '이 아기의 수면 리듬'을 요약해 부모에게 "
           "루틴 코칭(자주 깨는 시간 전 대비 등)을 제공한다.",
           "", "## 한계", "", "- 합성 다일 데이터 기준이며 실제 장기 관찰 검증이 아니다."]
    (RESULTS / "SLEEP_RHYTHM_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
