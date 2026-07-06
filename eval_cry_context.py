"""울음-맥락 상관 분석 검증.

건조(낮은 습도)·특정 시간대와 울음이 상관되도록 만든 합성 야간 데이터에서
analyze()가 그 상관을 실제로 찾아내는지 확인한다. (합성 기반)

실행: python eval_cry_context.py
"""
from pathlib import Path
import numpy as np

import cry_context

RESULTS = Path("results")


def make_night(seed=0, hours=8):
    """습도가 밤새 낮아지고, 건조·새벽 시간대에 울음이 잦은 합성 야간."""
    rng = np.random.default_rng(seed)
    mins = hours * 60
    recs = []
    for m in range(mins):
        humidity = 52 - (m / mins) * 32 + rng.normal(0, 2)   # 52% → 20% 하강
        p_cry = 0.01
        if humidity < 35:
            p_cry += 0.28                                     # 건조하면 울음↑↑
        if 3 <= (m // 60) <= 4:
            p_cry += 0.08                                     # 새벽 3~4시 울음↑
        recs.append({
            "minute": m,
            "cry": int(rng.random() < p_cry),
            "humidity": round(float(humidity), 1),
            "co2": int(np.clip(rng.normal(680, 60), 400, 1500)),
            "temp": round(float(rng.normal(22, 0.4)), 1),
        })
    return recs


def main():
    RESULTS.mkdir(exist_ok=True)
    recs = make_night(seed=7)
    out = cry_context.analyze(recs)

    print(f"cry_minutes={out['cry_minutes']}")
    print("env(cry vs baseline):", out["env"])
    print("insights:")
    for s in out["insights"]:
        print("  -", s)

    md = ["# Cry-Context Correlation", "",
          f"- 합성 야간(습도 하강 + 새벽 울음↑) 기준, 울음 {out['cry_minutes']}분", "",
          "| 변수 | 울음 시 평균 | 평상시 평균 | 차이 |", "|---|---:|---:|---:|"]
    for k, v in out["env"].items():
        md.append(f"| {k} | {v['cry']} | {v['baseline']} | {v['diff']} |")
    md += ["", "## 도출된 인사이트", ""]
    md += [f"- {s}" for s in out["insights"]]
    md += ["", "## 의의", "",
           "- 울음 '이유'를 분류하지 않고도, 울음과 동반되는 환경·시간 맥락을 찾아 선제 대응"
           "(예: 건조 시 가습) 근거로 활용할 수 있다.",
           "", "## 한계", "", "- 합성 상관 데이터 기준이며 실제 아기 데이터의 인과를 증명하지 않는다."]
    (RESULTS / "CRY_CONTEXT_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
