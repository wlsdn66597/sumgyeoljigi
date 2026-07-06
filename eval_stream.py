"""스트리밍(시계열) 융합 평가 — 디바운스·쿨다운을 실제로 측정한다.

스냅샷 평가(eval_fusion)는 상태 전이/시간 동작을 측정하지 못한다. 여기서는
가짜 시계로 라이브 Fusion을 초 단위로 구동하여:
  - 지속 무호흡 이벤트의 검출 지연(초)
  - 1초짜리 일시적 이상(artifact)의 오경보 억제
  - 디바운스·쿨다운 유무에 따른 총 알림 수 변화
를 비교한다. (합성 시계열 기준, 실환경 검증 아님)

실행: python eval_stream.py
"""
import csv
from pathlib import Path

from fusion import Fusion
import topics

RESULTS = Path("results")
ART = RESULTS / "artifacts"

NORMAL_RADAR = {"presence": True, "breathing_rate": 40, "movement": 0.1, "apnea": False}
APNEA_RADAR = {"presence": True, "breathing_rate": 3, "movement": 0.05, "apnea": True}
NORMAL_CRY = {"is_crying": False, "cls": "none", "confidence": 0.1}
NORMAL_ENV = {"co2": 600, "temp": 22}

APNEA_START, APNEA_END = 30, 45     # 지속 무호흡 (참 이벤트)
TRANSIENT_SECONDS = {60, 65, 70}    # 1초짜리 일시적 무호흡 blip (오경보 유발원)
DURATION = 100


def timeline():
    """초별 (t, radar, is_true_event) 생성."""
    for t in range(DURATION):
        if APNEA_START <= t < APNEA_END:
            yield t, APNEA_RADAR, True
        elif t in TRANSIENT_SECONDS:
            yield t, APNEA_RADAR, False   # 1초만 이상 → 실제 위험 아님
        else:
            yield t, NORMAL_RADAR, False


def run(debounce_n, cooldown_s):
    clock = [0.0]
    fz = Fusion(debounce_n=debounce_n, cooldown_s=cooldown_s, now_fn=lambda: clock[0])
    alerts = []          # 알림이 발생한 초
    for t, radar, _ in timeline():
        clock[0] = float(t)
        _, _, emitted = fz.update(radar=radar, cry=NORMAL_CRY, env=NORMAL_ENV)
        if emitted:
            alerts.append(t)

    # 지속 무호흡 검출 지연
    ev_alerts = [t for t in alerts if APNEA_START <= t < APNEA_END]
    latency = (ev_alerts[0] - APNEA_START) if ev_alerts else None
    # 일시적 artifact로 인한 오경보
    transient_fa = sum(1 for t in alerts if t in TRANSIENT_SECONDS)
    return {
        "debounce_n": debounce_n,
        "cooldown_s": cooldown_s,
        "total_alerts": len(alerts),
        "apnea_detect_latency_s": latency,
        "transient_false_alerts": transient_fa,
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    raw = run(debounce_n=1, cooldown_s=0)      # 필터 없음(원시)
    tuned = run(debounce_n=2, cooldown_s=10)   # 디바운스+쿨다운

    rows = [raw, tuned]
    fields = ["debounce_n", "cooldown_s", "total_alerts", "apnea_detect_latency_s", "transient_false_alerts"]
    with (ART / "stream_eval_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    print("config,total_alerts,apnea_latency_s,transient_false_alerts")
    print(f"raw(debounce=1,cooldown=0),{raw['total_alerts']},{raw['apnea_detect_latency_s']},{raw['transient_false_alerts']}")
    print(f"tuned(debounce=2,cooldown=10),{tuned['total_alerts']},{tuned['apnea_detect_latency_s']},{tuned['transient_false_alerts']}")

    md = [
        "# Streaming Fusion Evaluation (debounce/cooldown)",
        "",
        f"- Synthetic timeline: {DURATION}s, sustained apnea {APNEA_START}-{APNEA_END}s, "
        f"transient 1s blips at {sorted(TRANSIENT_SECONDS)}",
        "",
        "| config | total alerts | apnea detect latency (s) | transient false alerts |",
        "|---|---:|---:|---:|",
        f"| raw (debounce=1, cooldown=0) | {raw['total_alerts']} | {raw['apnea_detect_latency_s']} | {raw['transient_false_alerts']} |",
        f"| tuned (debounce=2, cooldown=10) | {tuned['total_alerts']} | {tuned['apnea_detect_latency_s']} | {tuned['transient_false_alerts']} |",
        "",
        "## Interpretation",
        "",
        "- Tuned debounce suppresses 1-second transient artifacts that the raw policy would alert on.",
        "- Cooldown collapses repeated alerts of the same sustained event into fewer notifications.",
        "- Sustained apnea is still detected with a small latency (debounce window).",
        "",
        "## Limits",
        "",
        "- Synthetic timeline only; not a real sensor or clinical validation.",
    ]
    (RESULTS / "STREAM_EVAL_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
