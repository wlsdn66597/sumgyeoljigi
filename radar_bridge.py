"""MR60BHA2(XIAO ESP32C6, ESPHome) → sensor/radar 브리지.

레이더 키트의 ESPHome 펌웨어(deploy/mr60bha2.yaml)가 상태를 MQTT로 발행하면,
이 브리지가 구독해 숨결지기 표준 스키마(sensor/radar)로 1초 주기 재발행한다.
상위 계층(fusion·대시보드)은 무수정.

유도 처리(MR60BHA2가 직접 안 주는 값):
- movement: 컴포넌트가 움직임 값을 안 주므로 distance 변동폭으로 유도한다
  (최근 창의 max-min 이 클수록 큰 움직임 = 뒤척임).
- apnea: 재실 중 호흡수 < APNEA_BPM_MIN 이 APNEA_HOLD_S 이상 지속되면 True.

안정성:
- ESPHome 수신이 STALE_S 이상 끊기거나 호흡수가 NaN이면 발행을 멈춘다
  (오래된 값으로 '정상'을 주장하지 않기 위함; 공백은 recorder 커버리지에 드러남).
- 전환 절차: 이 브리지를 켜면 nuni-edge 의 가상 레이더는 NUNI_SIM_RADAR=0 으로 끈다.

실행: python radar_bridge.py   (systemd: nuni-radar-bridge — 레이더 도착 후 enable)
"""
import collections
import json
import math
import os
import time

import paho.mqtt.client as mqtt

import topics

PREFIX = os.getenv("NUNI_ESPHOME_PREFIX", "mr60bha2")
# ESPHome state_topic(YAML에서 명시) → 내부 키
TOPICS = {
    f"{PREFIX}/breath_rate": "breath",     # 회/분
    f"{PREFIX}/heart_rate": "heart",       # 참고 기록용
    f"{PREFIX}/presence": "presence",      # ON/OFF (has_target)
    f"{PREFIX}/distance": "distance",      # cm (움직임 유도용)
}

APNEA_BPM_MIN = float(os.getenv("NUNI_APNEA_BPM_MIN", "6"))
APNEA_HOLD_S = float(os.getenv("NUNI_APNEA_HOLD_S", "4"))
STALE_S = float(os.getenv("NUNI_RADAR_STALE_S", "10"))
MOVE_WINDOW_S = float(os.getenv("NUNI_MOVE_WINDOW_S", "2.0"))
MOVE_SCALE_CM = float(os.getenv("NUNI_MOVE_SCALE_CM", "30"))   # 이만큼 변동 = 움직임 1.0
HOST = os.getenv("NUNI_MQTT_HOST", "localhost")

_state = {"breath": None, "heart": None, "presence": None,
          "distance": None, "last_rx": 0.0}
_low_since = [None]                         # 호흡 저하 시작 시각(무호흡 유도)
_dist_hist = collections.deque()            # (ts, distance_cm)


def on_esphome(topic, raw):
    """ESPHome 상태값(평문) 수신 → 내부 상태 갱신."""
    key = TOPICS.get(topic)
    if key is None:
        return
    now = time.time()
    if key == "presence":
        _state[key] = raw.strip().upper() in ("ON", "TRUE", "1")
    else:
        try:
            val = float(raw)
        except ValueError:
            return
        _state[key] = val
        if key == "distance" and not math.isnan(val):
            _dist_hist.append((now, val))
    _state["last_rx"] = now


def _movement(now):
    """최근 창의 거리 변동폭으로 움직임(0~1)을 유도."""
    while _dist_hist and now - _dist_hist[0][0] > MOVE_WINDOW_S:
        _dist_hist.popleft()
    if len(_dist_hist) < 2:
        return 0.1
    ds = [d for _, d in _dist_hist]
    return max(0.0, min(1.0, (max(ds) - min(ds)) / MOVE_SCALE_CM))


def derive_apnea(presence, bpm, now):
    """재실 중 호흡수가 임계 미만으로 APNEA_HOLD_S 지속되면 무호흡 의심."""
    if not presence or bpm is None:
        _low_since[0] = None
        return False
    if bpm < APNEA_BPM_MIN:
        if _low_since[0] is None:
            _low_since[0] = now
        return (now - _low_since[0]) >= APNEA_HOLD_S
    _low_since[0] = None
    return False


def step(now=None):
    """현재 수신 상태 → sensor/radar 메시지. 데이터 없거나 stale/NaN이면 None."""
    now = now or time.time()
    b = _state["breath"]
    if b is None or math.isnan(b) or (now - _state["last_rx"]) > STALE_S:
        return None
    presence = _state["presence"] if _state["presence"] is not None else True
    move = _movement(now)
    apnea = derive_apnea(presence, b, now)
    msg = topics.radar_msg(round(0.0 if apnea else b, 1), round(move, 2), bool(presence))
    msg.update({"apnea": apnea, "motion": move > 0.5, "source": "mr60bha2",
                "heart_rate": _state["heart"], "distance": _state["distance"]})
    return msg


def main():
    _cbv = getattr(mqtt, "CallbackAPIVersion", None)
    c = mqtt.Client(_cbv.VERSION1) if _cbv else mqtt.Client()
    c.on_message = lambda cl, ud, m: on_esphome(m.topic, m.payload.decode("utf-8", "replace"))
    c.connect(HOST, 1883, 60)
    for t in TOPICS:
        c.subscribe(t)
    c.loop_start()
    print(f"[radar-bridge] ESPHome '{PREFIX}/*' -> {topics.RADAR} (stale {STALE_S}s)")
    warned = False
    while True:
        m = step()
        if m is not None:
            c.publish(topics.RADAR, json.dumps(m))
            warned = False
        elif not warned:
            print("[radar-bridge] ESPHome 수신 대기 중 (레이더 미연결/stale)")
            warned = True
        time.sleep(1)


if __name__ == "__main__":
    main()
