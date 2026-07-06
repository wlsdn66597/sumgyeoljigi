"""수면 리듬 학습 → 루틴 인사이트.

여러 밤의 분단위 수면 상태(안정수면/뒤척임/각성)를 모아, 이 아기의 전형적 리듬
(각성 횟수, 최장 안정수면, 자주 깨는 시간대, 상태 비율)을 요약한다. 개인화의 상위
서비스로, 부모에게 '루틴 코칭'을 제공한다.
"""
import numpy as np


def _night_stats(states):
    """states: 분단위 상태 리스트(calm_sleep/restless/awake). 하룻밤 통계."""
    n = len(states)
    awakenings = 0
    prev = None
    best = cur = 0
    wake_hours = []
    for i, s in enumerate(states):
        if s == "awake" and prev != "awake":
            awakenings += 1
            wake_hours.append(i // 60)
        cur = cur + 1 if s == "calm_sleep" else 0
        best = max(best, cur)
        prev = s
    prop = {k: round(states.count(k) / n, 3) for k in ("calm_sleep", "restless", "awake")} if n else {}
    return {"awakenings": awakenings, "longest_calm_min": best, "wake_hours": wake_hours, "prop": prop}


def analyze(nights):
    """nights: [분단위 상태 리스트, ...] (여러 밤) → 루틴 인사이트."""
    per = [_night_stats(s) for s in nights if s]
    if not per:
        return {"nights": 0, "insights": ["데이터 없음"]}
    avg_awake = float(np.mean([p["awakenings"] for p in per]))
    avg_calm = float(np.mean([p["longest_calm_min"] for p in per]))
    hour_hist = {}
    for p in per:
        for h in p["wake_hours"]:
            hour_hist[h] = hour_hist.get(h, 0) + 1
    prop_avg = {k: round(float(np.mean([p["prop"].get(k, 0) for p in per])), 3)
                for k in ("calm_sleep", "restless", "awake")}

    insights = [f"밤 평균 각성 {avg_awake:.1f}회, 최장 안정수면 평균 {avg_calm:.0f}분",
                f"수면 상태 비율 — 안정수면 {prop_avg['calm_sleep']:.0%}, 뒤척임 "
                f"{prop_avg['restless']:.0%}, 각성 {prop_avg['awake']:.0%}"]
    if hour_hist:
        peak = max(hour_hist, key=hour_hist.get)
        insights.append(f"자주 깨는 시간대: {peak}시경 (관측 {len(nights)}박 중 {hour_hist[peak]}회) "
                        f"→ 해당 시간 전 수유/환경 점검 루틴 제안")
    return {"nights": len(per), "avg_awakenings": round(avg_awake, 1),
            "avg_longest_calm_min": round(avg_calm, 1), "wake_hour_hist": hour_hist,
            "state_proportion": prop_avg, "insights": insights}
