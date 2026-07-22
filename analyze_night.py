"""야간 수집(JSONL) 분석 → 아침 리포트 생성.

recorder.py가 남긴 원시 로그에서 안정성·호흡·이벤트 지표를 뽑아
`results/NIGHT_REPORT_<날짜>.md` (+ 호흡 타임라인 PNG)로 저장한다.

지표: 수집 커버리지(공백), 재실 비율·이탈 에피소드, 수면 중 호흡수 분포,
움직임 에피소드(수면일지 대조용 타임스탬프), 경보 목록(야간 오경보 수),
수면상태 비율, 개인화 baseline 수렴.

사용: python analyze_night.py ~/nuni_data/nuni_20260712.jsonl [추가파일...]
"""
import json
import statistics
import sys
import time
from pathlib import Path

GAP_S = 5.0          # 이보다 긴 무수신 = 수집 공백
EXIT_MIN_S = 30.0    # 이보다 긴 부재 = 이탈 에피소드
MOVE_MIN_S = 3.0     # 움직임 에피소드 최소 길이
MOVE_MERGE_S = 10.0  # 움직임 에피소드 병합 간격
BR_LO, BR_HI = 5.0, 60.0   # 호흡 통계에 넣을 유효 범위

RESULTS = Path("results")


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
    rows.sort(key=lambda r: r.get("rx_ts", 0))
    return rows


def t(ts):
    return time.strftime("%H:%M:%S", time.localtime(ts))


def episodes(points, min_len=0.0, merge_gap=0.0):
    """[(ts, bool)] → True 구간 [(start, end)] (병합·최소길이 필터)."""
    eps, start, last = [], None, None
    for ts, flag in points:
        if flag:
            if start is None:
                start = ts
            last = ts
        elif start is not None:
            eps.append([start, last])
            start = None
    if start is not None:
        eps.append([start, last])
    if merge_gap:
        merged = []
        for s, e in eps:
            if merged and s - merged[-1][1] <= merge_gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        eps = merged
    return [(s, e) for s, e in eps if e - s >= min_len]


def main(paths):
    rows = load(paths)
    radar = [(r["rx_ts"], r["payload"]) for r in rows
             if r["topic"] == "sensor/radar" and isinstance(r["payload"], dict)]
    alerts = [(r["rx_ts"], r["payload"]) for r in rows
              if r["topic"] == "fusion/alert" and isinstance(r["payload"], dict)]
    fstates = [(r["rx_ts"], r["payload"]) for r in rows
               if r["topic"] == "fusion/state" and isinstance(r["payload"], dict)]
    if len(radar) < 10:
        print(f"레이더 메시지가 너무 적음({len(radar)}건) — 수집 확인 필요")
        return

    t0, t1 = radar[0][0], radar[-1][0]
    span = t1 - t0

    # 1) 수집 커버리지
    gaps = [(a, b) for (a, _), (b, __) in zip(radar, radar[1:]) if b - a > GAP_S]
    gap_total = sum(b - a for a, b in gaps)
    coverage = 1 - gap_total / span if span > 0 else 0

    # 2) 재실/이탈
    pres_pts = [(ts, bool(p.get("presence"))) for ts, p in radar]
    pres_ratio = sum(f for _, f in pres_pts) / len(pres_pts)
    exits = episodes([(ts, not f) for ts, f in pres_pts], min_len=EXIT_MIN_S)

    # 3) 수면 중 호흡수 분포 (재실·비움직임·유효범위)
    br = [p.get("breathing_rate", 0) for ts, p in radar
          if p.get("presence") and not p.get("motion")
          and BR_LO <= p.get("breathing_rate", 0) <= BR_HI]
    br_stat = (statistics.median(br), statistics.quantiles(br, n=10)[0],
               statistics.quantiles(br, n=10)[-1]) if len(br) >= 10 else None

    # 4) 움직임 에피소드 (수면일지 대조용)
    move_pts = [(ts, bool(p.get("motion")) or p.get("movement", 0) > 0.5)
                for ts, p in radar]
    moves = episodes(move_pts, min_len=MOVE_MIN_S, merge_gap=MOVE_MERGE_S)

    # 5) 수면상태 비율 (fusion/state.sleep)
    sleep_cnt = {}
    for _, p in fstates:
        s = (p.get("sleep") or {}).get("state")
        if s:
            sleep_cnt[s] = sleep_cnt.get(s, 0) + 1
    stotal = sum(sleep_cnt.values()) or 1

    # 6) 개인화 수렴 (처음/중간/마지막)
    personals = [(ts, p["personal"]) for ts, p in fstates
                 if p.get("personal", {}).get("bpm_normal_range")]
    conv = [personals[0], personals[len(personals) // 2], personals[-1]] if personals else []

    day = time.strftime("%Y%m%d", time.localtime(t0))
    md = [
        f"# 야간 수집 리포트 ({time.strftime('%Y-%m-%d', time.localtime(t0))})",
        "",
        f"- 구간: **{t(t0)} ~ {t(t1)}** ({span / 3600:.1f}시간), 레이더 {len(radar)}건",
        f"- 소스: {', '.join(sorted({p.get('source', '?') for _, p in radar}))}",
        "",
        "## 1. 안정성",
        f"- 수집 커버리지 **{coverage:.2%}** (공백 {len(gaps)}회, 총 {gap_total:.0f}초)",
        *[f"  - 공백: {t(a)} ~ {t(b)} ({b - a:.0f}초)" for a, b in gaps[:10]],
        "",
        "## 2. 재실",
        f"- 재실 비율 **{pres_ratio:.1%}**, 이탈(≥{EXIT_MIN_S:.0f}초) **{len(exits)}회**",
        *[f"  - 이탈: {t(s)} ~ {t(e)} ({(e - s) / 60:.1f}분)" for s, e in exits],
        "",
        "## 3. 수면 중 호흡수",
        (f"- 중앙값 **{br_stat[0]:.1f}회/분** (p10 {br_stat[1]:.1f} / p90 {br_stat[2]:.1f}, 표본 {len(br)})"
         if br_stat else f"- 유효 표본 부족({len(br)})"),
        "",
        "## 4. 움직임 에피소드 (수면일지와 대조할 것)",
        f"- 총 **{len(moves)}회**",
        *[f"  - {t(s)} ~ {t(e)} ({e - s:.0f}초)" for s, e in moves[:30]],
        "",
        "## 5. 경보 (야간 오경보 검토)",
        f"- fusion/alert **{len(alerts)}건**",
        *[f"  - {t(ts)} [{p.get('level')}] {p.get('reason')}" for ts, p in alerts[:20]],
        "",
        "## 6. 수면상태 비율",
        *[f"- {k}: {v / stotal:.1%}" for k, v in sorted(sleep_cnt.items())],
        "",
        "## 7. 개인화 수렴",
        *[f"- {t(ts)}: 정상범위 {pp.get('bpm_normal_range')} (표본 {pp.get('samples')})"
          for ts, pp in conv],
        "",
        "> 주의: 기능 검증용 요약이며 의료적 판정이 아님. 수면일지·수동 측정과 대조해 해석할 것.",
    ]
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"NIGHT_REPORT_{day}.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    # 타임라인 PNG (matplotlib 있으면)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [(ts - t0) / 3600 for ts, _ in radar]
        ys = [p.get("breathing_rate", 0) for _, p in radar]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(xs, ys, lw=0.6)
        for s, e in moves:
            ax.axvspan((s - t0) / 3600, (e - t0) / 3600, alpha=0.2, color="orange")
        for ts, _ in alerts:
            ax.axvline((ts - t0) / 3600, color="red", ls="--", lw=0.8)
        ax.set_xlabel("hours"); ax.set_ylabel("breaths/min")
        ax.set_title("Breathing timeline (orange=movement, red=alert)")
        png = RESULTS / f"NIGHT_TIMELINE_{day}.png"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"wrote {png}")
    except ImportError:
        print("(matplotlib 없음 — PNG 생략)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
