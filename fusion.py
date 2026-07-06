"""멀티모달 융합 판단 엔진.

decide(): 레이더+음향+환경을 교차검증하는 규칙 (핵심 차별점)
single_sensor_decide(): 비교용 단일 센서 단순 정책 (오탐이 많음)
-> eval_fusion.py 에서 둘을 비교해 "융합이 오탐을 줄인다"를 정량화.
"""
import topics
import bus
from state_store import store
import time

import sleep_state
import env_control
from personalization import Personalizer

_ORDER = {"normal": 0, "attention": 1, "alert": 2}


def _max(a, b):
    return a if _ORDER[a] >= _ORDER[b] else b


def decide(radar, cry, env):
    """(level, reasons) 반환. level: normal/attention/alert"""
    level, reasons = "normal", []

    apnea = bool(radar and radar.get("apnea"))
    br = radar.get("breathing_rate", radar.get("breathing_bpm", 99)) if radar else 99
    if radar and radar.get("presence") and (apnea or br < 8):
        level = _max(level, "alert")
        reasons.append("호흡 미검출·무호흡 의심")

    cry_conf = cry.get("confidence", cry.get("cry_score", 0)) if cry else 0
    cry_flag = bool(cry and cry.get("is_crying") and cry_conf >= 0.7)
    if cry_flag:
        level = _max(level, "attention")
        reasons.append(f"울음 감지({cry.get('cls')})")

    # 교차검증: 울음 + (호흡 이상 or 큰 움직임)이 동시일 때만 경보로 상향
    if cry_flag and radar and (radar.get("movement", 0) > 0.6 or apnea or br < 8):
        level = _max(level, "alert")
        reasons.append("다중 신호 동시 이상")

    # 경계 신호 교차검증: 단독으론 임계 미만인 '경계 호흡'이 다른 모달과 겹치면 경보로 상향.
    #  -> 레이더 단독은 놓치지만(정상 판정) 융합은 잡는 케이스 = 융합의 고유 가치.
    irregular = bool(radar and radar.get("presence") and not apnea and (8 <= br <= 15 or br > 55))
    bad_env = bool(env and env.get("co2", 0) > 1000)
    if irregular and (cry_flag or bad_env):
        level = _max(level, "alert")
        reasons.append("경계 호흡+보강신호 교차검증")

    if env:
        if env.get("co2", 0) > 1000:
            level = _max(level, "attention")
            reasons.append("CO₂ 높음")
        t = env.get("temp")
        if t is not None and (t < 18 or t > 26):
            level = _max(level, "attention")
            reasons.append("실내 온도 이상")

    return level, reasons


def single_sensor_decide(radar, cry, env):
    """비교용: 개별 신호 하나만으로 경보 -> 오탐/미탐 많음."""
    if cry and cry.get("is_crying"):
        return "alert", ["울음(단일)"]
    if radar and radar.get("apnea"):
        return "alert", ["무호흡(단일)"]
    if radar and radar.get("movement", 0) > 0.5:
        return "alert", ["움직임(단일)"]
    if env and env.get("co2", 0) > 800:
        return "alert", ["CO₂(단일)"]
    return "normal", []


class Fusion:
    """라이브 구독자: 최신 신호를 모아 판단하고 상태/경보를 발행.

    now_fn: 시간 소스 주입(기본 time.time). 스트리밍 평가에서 가짜 시계로 대체해
            디바운스·쿨다운을 실시간 대기 없이 검증할 수 있다.
    """

    def __init__(self, debounce_n=2, cooldown_s=10, now_fn=time.time):
        self.radar = self.cry = self.env = None
        self.last = "normal"
        self.debounce_n = debounce_n
        self.cooldown_s = cooldown_s
        self._now = now_fn
        self._candidate = "normal"
        self._candidate_count = 0
        self._last_alert_key = None
        self._last_alert_ts = 0.0
        self.personalizer = Personalizer()   # 개인화 baseline 학습

    def _debounce(self, level):
        if level == "normal":
            self._candidate = "normal"
            self._candidate_count = 0
            return level
        if level == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = level
            self._candidate_count = 1
        return level if self._candidate_count >= self.debounce_n else "normal"

    def _cooldown_allows(self, level, reasons):
        key = (level, "|".join(reasons))
        now = self._now()
        if key == self._last_alert_key and now - self._last_alert_ts < self.cooldown_s:
            return False
        self._last_alert_key = key
        self._last_alert_ts = now
        return True

    def _process(self):
        """현재 신호로 판단·발행. (level, reasons, alert_emitted) 반환."""
        raw_level, reasons = decide(self.radar, self.cry, self.env)
        level = self._debounce(raw_level)
        out_reasons = reasons if level != "normal" else []
        store.set_fusion(level, out_reasons)

        # --- 일상 모니터링(헤드라인): 수면상태 · 개인화 · 선제 환경제어 ---
        ss, ss_reason = sleep_state.classify(self.radar, self.cry)
        store.set_sleep_state(ss, ss_reason)
        personal = {}
        if self.radar:
            self.personalizer.update(self.radar.get("breathing_rate", 0),
                                     self.radar.get("movement", 0.0))
            personal = self.personalizer.summary()
            store.set_personal(personal)
        actions = env_control.recommend(self.env, ss)
        store.set_actions(actions)
        if actions:
            bus.publish(topics.CONTROL, {"actions": [a for a, _ in actions], "ts": topics.now()})

        # 대시보드가 MQTT 구독만으로 전체 상태를 복원하도록 통합 상태를 발행한다.
        # (분산 모드에서 대시보드는 파이프라인을 돌리지 않고 이 메시지로 화면을 채운다.)
        bus.publish(topics.FUSION_STATE, {
            "level": level, "reasons": out_reasons, "raw_level": raw_level,
            "sleep": {"state": ss, "reason": ss_reason},
            "personal": personal,
            "actions": [[a, r] for a, r in actions],
            "ts": topics.now(),
        })

        emitted = False
        if level != "normal" and level != self.last and self._cooldown_allows(level, out_reasons):
            bus.publish(topics.ALERT, {"level": level, "reason": ", ".join(out_reasons), "ts": topics.now()})
            emitted = True

        self.last = level
        return level, out_reasons, emitted

    def on_message(self, topic, payload):
        if topic == topics.RADAR:
            self.radar = payload
        elif topic == topics.CRY:
            self.cry = payload
        elif topic == topics.ENV:
            self.env = payload
        return self._process()

    def update(self, radar=None, cry=None, env=None):
        """한 틱에 세 모달을 한꺼번에 갱신하고 1회 판단 (시계열 평가용)."""
        if radar is not None:
            self.radar = radar
        if cry is not None:
            self.cry = cry
        if env is not None:
            self.env = env
        return self._process()
