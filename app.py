"""NUNI 데모 대시보드 (Streamlit).

실행: streamlit run app.py
사이드바 버튼으로 울음/무호흡 시나리오를 주입하면 융합 판단이 반응한다.
"""
import os
import time

import streamlit as st

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
}

with st.sidebar:
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

    with holder.container():
        reason = ", ".join(snap["fusion"]["reasons"]) or "이상 없음"
        getattr(st, kind)(f"{icon}  상태: {label}  —  {reason}")

        ss = snap["sleep_state"]
        st.markdown(f"### 수면 상태: {SLEEP_UI.get(ss['state'], ss['state'])}")
        p = snap.get("personal", {})
        if p.get("bpm_normal_range"):
            st.caption(f"개인화 정상 호흡 범위 {p['bpm_normal_range']} 회/분 (학습 표본 {p.get('samples')})")

        c1 = st.columns(4)
        c1[0].metric("호흡수(회/분)", r.get("breathing_rate", "-"))
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
