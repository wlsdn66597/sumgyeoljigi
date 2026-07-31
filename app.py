"""NUNI 데모 대시보드 (Streamlit).

실행: streamlit run app.py
사이드바 버튼으로 울음/무호흡 시나리오를 주입하면 융합 판단이 반응한다.
"""
import os
import time

import streamlit as st

import bus
import topics
import workers
from state_store import store

st.set_page_config(page_title="NUNI 데모", layout="wide")
# 분산 모드(NUNI_DASHBOARD_ONLY=1): 파이프라인은 nuni-edge가 상시 구동하고,
# 대시보드는 브로커 구독만 한다. 미설정 시 단일 프로세스로 자체 파이프라인을 돌린다.
workers.ensure_started(pipeline=os.getenv("NUNI_DASHBOARD_ONLY") != "1")

LEVEL_UI = {
    "normal": ("정상", "🟢", "success"),
    "attention": ("주의", "🟡", "warning"),
    "alert": ("경보", "🔴", "error"),
}

SLEEP_UI = {
    "calm_sleep": "😴 안정 수면",
    "restless": "🌀 뒤척임",
    "awake": "👀 각성",
    "unknown": "· 재실 없음",
    "hold": "⏳ 판정 보류",      # 사람은 있으나 호흡 판단 불가 — '재실 없음'과 구분
}

DISTRIBUTED = os.getenv("NUNI_DASHBOARD_ONLY") == "1"

with st.sidebar:
    if DISTRIBUTED:
        # 실물 모드에서는 시나리오 '주입'이 의미가 없다. 판단은 별도 프로세스(nuni-edge)가
        # 실제 센서 값으로 하므로, 대시보드 프로세스의 주입 플래그는 전달되지 않는다.
        # 자극은 소프트웨어가 아니라 물리적으로 준다(숨쉬는 인형·소리·입김).
        st.header("실물 데모")
        st.caption("모든 값이 실제 센서에서 옵니다. 자극은 물리적으로 줍니다.")
        st.markdown(
            "**호흡** — 숨쉬는 인형을 레이더 **정면 0.5~1m**, 가슴이 센서를 향하게\n\n"
            "**무호흡** — 인형 호흡을 멈추거나 센서 범위 밖으로\n\n"
            "**울음** — 울음소리 재생 (마이크 근처)\n\n"
            "**환경** — 센서에 입김 → CO₂ 상승 → 환기 권고"
        )
        with st.expander("서보 시뮬레이터 제어 (사용 시)"):
            st.caption("`breath-sim` 서비스 실행 중일 때만 동작합니다.")
            bpm = st.slider("호흡수(회/분)", 20, 60, 40, help="영아 정상 범위 30~60")
            if st.button("▶ 호흡수 적용", use_container_width=True):
                bus.publish("demo/breath", {"bpm": float(bpm)})
                st.toast(f"시뮬레이터 {bpm}회/분")
            if st.button("⚠️ 무호흡 주입 (10초)", use_container_width=True):
                bus.publish("demo/breath", {"apnea_s": 10})
                st.toast("서보 정지 — 무호흡 모사")
    else:
        st.header("데모 시나리오")
        if st.button("👶 울음 발생", use_container_width=True):
            store.set_inject("cry", 8)
        if st.button("⚠️ 무호흡 이벤트", use_container_width=True):
            store.set_inject("apnea", 8)
        if st.button("✅ 정상 복귀", use_container_width=True):
            store.clear_inject()
        st.caption("실물 센서 없이 시나리오를 재현합니다. (시뮬레이션)")

st.title("NUNI · 비접촉 영유아 케어 모니터")
holder = st.empty()

while True:
    snap = store.snapshot()
    lv = snap["fusion"]["level"]
    label, icon, kind = LEVEL_UI.get(lv, LEVEL_UI["normal"])
    r = snap["latest"].get(topics.RADAR, {})
    e = snap["latest"].get(topics.ENV, {})
    c = snap["latest"].get(topics.CRY, {})

    # 실물 레이더는 대상이 멀거나 비스듬하면 호흡을 못 잰다. 이때 호흡수 0은
    # '무호흡'이 아니라 '미측정'이므로, 0을 그대로 띄우면서 "정상"이라고 하면
    # 모순처럼 보인다. 측정 불가는 별도 상태로 구분해 보여준다.
    breath_ok = r.get("breath_valid", True)
    unmeasurable = bool(r.get("presence")) and not breath_ok

    with holder.container():
        if unmeasurable and lv == "normal":
            st.info("⚪  상태: 호흡 측정 대기  —  재실은 감지되나 호흡 신호를 얻지 못함 "
                    "(센서 정면 0.5~1m, 가슴을 향하게 두고 잠시 정지)")
        else:
            reason = ", ".join(snap["fusion"]["reasons"]) or "이상 없음"
            getattr(st, kind)(f"{icon}  상태: {label}  —  {reason}")

        ss = snap["sleep_state"]
        st.markdown(f"### 수면 상태: {SLEEP_UI.get(ss['state'], ss['state'])}")
        if ss.get("reason"):
            st.caption(ss["reason"])

        # 지금 화면의 값이 실물 센서에서 온 것인지 표시(데모 신뢰성)
        src = "실물 레이더(MR60BHA2)" if r.get("source") == "mr60bha2" else "시뮬레이션"
        st.caption(f"레이더 소스: {src}"
                   + (f" · 거리 {r['distance']}cm" if r.get("distance") is not None else "")
                   + (f" · 심박 {r['heart_rate']}bpm" if r.get("heart_rate") else ""))
        p = snap.get("personal", {})
        if p.get("bpm_normal_range"):
            st.caption(f"개인화 정상 호흡 범위 {p['bpm_normal_range']} 회/분 (학습 표본 {p.get('samples')})")

        c1 = st.columns(4)
        # 미측정일 때 0을 띄우면 '호흡 정지'로 오해된다 → 측정 불가로 표기
        c1[0].metric("호흡수(회/분)",
                     "측정 불가" if unmeasurable else r.get("breathing_rate", "-"))
        c1[1].metric("움직임", r.get("movement", "-"))
        c1[2].metric("울음", c.get("cls") if c.get("is_crying") else "없음")
        c1[3].metric("재실", "있음" if r.get("presence") else "-")

        c2 = st.columns(4)
        c2[0].metric("온도(℃)", e.get("temp", "-"))
        c2[1].metric("습도(%)", e.get("humidity", "-"))
        c2[2].metric("CO₂(ppm)", e.get("co2", "-"))
        c2[3].metric("조도(lux)", e.get("lux", "-"))

        acts = snap.get("actions", [])
        if acts:
            st.subheader("환경 제어 권고 (선제 케어)")
            for _a, _reason in acts:
                st.write(f"• {_reason}")

        st.subheader("호흡 파형 (최근 60초)")
        if snap["radar"]:
            st.line_chart({"호흡수": snap["radar"]})

        st.subheader("이벤트 로그")
        for ev in snap["events"]:
            st.text(ev)

    time.sleep(1)
