"""클라우드(비실시간) 스텁 + 아침 수면 리포트.

하이브리드 구조에서 클라우드 역할을 모사한다. 엣지가 올린 '비식별 이벤트/특징'만
받아(원시 음성·레이더 신호는 전송하지 않음) 밤새 추세를 요약한다.

실행: python cloud.py   → results/SLEEP_REPORT.md 생성
"""
import csv
from pathlib import Path
import numpy as np

from personalization import Personalizer
import cry_context

RESULTS = Path("results")
ART = RESULTS / "artifacts"


def simulate_night(hours=8, seed=7):
    """비식별 분단위 특징 레코드 생성 (엣지가 업로드했다고 가정)."""
    rng = np.random.default_rng(seed)
    mins = hours * 60

    cry_minutes = set()
    for onset in rng.choice(range(10, mins - 6), size=4, replace=False):
        for m in range(int(onset), int(onset) + int(rng.integers(2, 6))):
            cry_minutes.add(m)
    apnea_onset = int(rng.integers(90, mins - 2))
    apnea_minutes = {apnea_onset}

    recs = []
    for m in range(mins):
        cry = m in cry_minutes
        apnea = m in apnea_minutes
        recs.append({
            "minute": m,
            "breathing_rate": 2.0 if apnea else round(float(rng.normal(40, 2)), 1),
            "cry": int(cry),
            "apnea": int(apnea),
            "movement": round(float(rng.uniform(0.3, 0.8) if cry else rng.uniform(0, 0.2)), 2),
            "co2": int(np.clip(rng.normal(680 + m * 0.15, 60), 400, 1500)),
            "temp": round(float(rng.normal(22, 0.4)), 1),
            "humidity": round(float(np.clip(rng.normal(48 - m / mins * 12, 3), 20, 70)), 1),
        })
    return recs


def _events(recs, key):
    n = prev = 0
    for r in recs:
        if r[key] and not prev:
            n += 1
        prev = r[key]
    return n


def summarize(recs):
    brs = [r["breathing_rate"] for r in recs if not r["apnea"]]
    best = cur = 0
    for r in recs:
        calm = (not r["cry"]) and (not r["apnea"]) and r["movement"] < 0.3
        cur = cur + 1 if calm else 0
        best = max(best, cur)
    by_hour = {}
    for r in recs:
        if r["cry"]:
            h = r["minute"] // 60
            by_hour[h] = by_hour.get(h, 0) + 1
    personal = Personalizer(alpha=0.02)          # 야간 데이터로 개인 baseline 학습
    for r in recs:
        personal.update(r["breathing_rate"], r["movement"])
    return {
        "personal_bpm_range": personal.normal_bpm_range(),
        "monitored_min": len(recs),
        "cry_events": _events(recs, "cry"),
        "cry_minutes": sum(r["cry"] for r in recs),
        "apnea_events": _events(recs, "apnea"),
        "avg_bpm": round(float(np.mean(brs)), 1),
        "bpm_min": round(float(np.min(brs)), 1),
        "bpm_max": round(float(np.max(brs)), 1),
        "co2_min": min(r["co2"] for r in recs),
        "co2_max": max(r["co2"] for r in recs),
        "temp_min": min(r["temp"] for r in recs),
        "temp_max": max(r["temp"] for r in recs),
        "longest_calm_min": best,
        "cry_by_hour": by_hour,
    }


def write_report(recs, s):
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / "sleep_timeline.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader(); w.writerows(recs)

    hours = s["monitored_min"] / 60
    hour_lines = "\n".join(f"| {h}시간대 | {c}분 |" for h, c in sorted(s["cry_by_hour"].items())) or "| - | 0 |"
    cc = cry_context.analyze(recs)
    cc_lines = "\n".join(f"- {t}" for t in cc["insights"])
    md = f"""# 아침 수면 리포트 (클라우드 · 비실시간)

> 엣지가 업로드한 **비식별 이벤트/특징만** 사용한다. 원시 음성·레이더 신호는 전송하지 않는다.

## 야간 요약 ({hours:.1f}시간 모니터링)

| 항목 | 값 |
|---|---|
| 울음 이벤트 | {s['cry_events']}회 (총 {s['cry_minutes']}분) |
| 무호흡 의심 이벤트 | {s['apnea_events']}회 |
| 평균 호흡수 | {s['avg_bpm']} 회/분 (범위 {s['bpm_min']}~{s['bpm_max']}) |
| 최장 안정 수면 | {s['longest_calm_min']}분 연속 |
| CO₂ 범위 | {s['co2_min']}~{s['co2_max']} ppm |
| 실내 온도 범위 | {s['temp_min']}~{s['temp_max']} ℃ |

## 시간대별 울음 분포

| 시간대 | 울음 |
|---|---|
{hour_lines}

## 울음-맥락 상관

{cc_lines}

## 개인화(추세 기반) 제언

- 학습된 개인 정상 호흡 범위: {s['personal_bpm_range']} 회/분 (이 아기 기준으로 임계값 보정).
- 최장 안정 수면 {s['longest_calm_min']}분 기준으로 가구별 baseline을 갱신한다.
- CO₂가 {s['co2_max']}ppm까지 상승 → 취침 전 환기 또는 공기청정 세기 상향을 제안.

## 한계

- 본 리포트는 합성 야간 데이터 기반이며, 실제 영유아·임상 검증이 아니다.
"""
    (RESULTS / "SLEEP_REPORT.md").write_text(md, encoding="utf-8")
    _try_plot(recs)


def _try_plot(recs):
    """가능하면 야간 타임라인 PNG 저장 (matplotlib 없으면 건너뜀)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    mins = [r["minute"] for r in recs]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(mins, [r["breathing_rate"] for r in recs], lw=0.8, label="breathing (bpm)")
    for r in recs:
        if r["cry"]:
            ax.axvspan(r["minute"], r["minute"] + 1, color="orange", alpha=0.3)
        if r["apnea"]:
            ax.axvspan(r["minute"], r["minute"] + 1, color="red", alpha=0.6)
    ax.set_xlabel("minute"); ax.set_ylabel("bpm"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(ART / "sleep_timeline.png", dpi=150)


def main():
    RESULTS.mkdir(exist_ok=True)
    recs = simulate_night()
    s = summarize(recs)
    write_report(recs, s)
    print("아침 수면 리포트 생성 → results/SLEEP_REPORT.md")
    print(f"  울음 {s['cry_events']}회 · 무호흡 의심 {s['apnea_events']}회 · "
          f"평균 호흡 {s['avg_bpm']}회/분 · 최장 안정수면 {s['longest_calm_min']}분")


if __name__ == "__main__":
    main()
