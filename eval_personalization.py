"""개인화 적응 검증: 서로 다른 baseline의 두 '가구'에 각각 수렴하는지 확인."""
import csv
from pathlib import Path

import numpy as np

from personalization import Personalizer

RESULTS = Path("results")
ART = RESULTS / "artifacts"

# 두 아기(가구): 평균 호흡수가 다름 → 개인화가 각각 학습해야 함
HOUSEHOLDS = {"babyA_bpm40": 40.0, "babyB_bpm52": 52.0}


def run_one(true_bpm, seed):
    rng = np.random.default_rng(seed)
    p = Personalizer(alpha=0.05)
    for _ in range(300):
        p.update(round(float(rng.normal(true_bpm, 2)), 1), round(float(rng.uniform(0, 0.2)), 2))
    return p


def main():
    RESULTS.mkdir(exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, true_bpm) in enumerate(HOUSEHOLDS.items()):
        p = run_one(true_bpm, seed=1000 + i)   # 고정 seed → 재현 가능
        s = p.summary()
        err = abs(s["bpm_mean"] - true_bpm)
        rows.append({"household": name, "true_bpm": true_bpm,
                     "learned_bpm": s["bpm_mean"], "abs_error": round(err, 2),
                     "normal_range": s["bpm_normal_range"]})

    with (ART / "personalization_eval.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["household", "true_bpm", "learned_bpm", "abs_error", "normal_range"])
        w.writeheader(); w.writerows(rows)

    print("household,true_bpm,learned_bpm,abs_error,normal_range")
    for r in rows:
        print(f"{r['household']},{r['true_bpm']},{r['learned_bpm']},{r['abs_error']},{r['normal_range']}")

    md = ["# Personalization (adaptive baseline)", "",
          "- 서로 다른 baseline의 두 가구에 개인화가 각각 수렴하는지 검증(합성).", "",
          "| 가구 | 실제 평균 BPM | 학습 BPM | 오차 | 개인 정상범위 |",
          "|---|---:|---:|---:|---|"]
    md += [f"| {r['household']} | {r['true_bpm']} | {r['learned_bpm']} | {r['abs_error']} | {r['normal_range']} |" for r in rows]
    md += ["", "## 해석", "",
           "- 동일 로직이 가구별로 다른 baseline을 학습 → 고정 임계값 대신 개인 기준으로 보정 가능.",
           "", "## 한계", "", "- 합성 데이터 기준이며 실제 영유아 장기 데이터 검증이 아니다."]
    (RESULTS / "PERSONALIZATION_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
