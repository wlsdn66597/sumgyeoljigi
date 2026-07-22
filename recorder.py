"""MQTT 전 토픽 레코더 — 야간 실험용 원시 데이터 기록(JSONL).

브로커의 모든 토픽을 `{rx_ts, topic, payload}` 한 줄 JSON으로
`~/nuni_data/nuni_YYYYMMDD.jsonl`에 기록한다(날짜 바뀌면 파일 교체).
대시보드/융합과 무관하게 원본을 남겨 사후 재분석·재현을 가능하게 하는 것이
목적. ESPHome 등 비JSON 페이로드는 문자열 그대로 저장한다.

실행: python recorder.py   (systemd: nuni-recorder, 상시 구동)
분석: python analyze_night.py ~/nuni_data/nuni_YYYYMMDD.jsonl
"""
import json
import os
import time
from pathlib import Path

import paho.mqtt.client as mqtt

DATA_DIR = Path(os.getenv("NUNI_DATA_DIR", str(Path.home() / "nuni_data")))
HOST = os.getenv("NUNI_MQTT_HOST", "localhost")

_cur = {"day": None, "fh": None}


def _file_for(now):
    day = time.strftime("%Y%m%d", time.localtime(now))
    if day != _cur["day"]:
        if _cur["fh"]:
            _cur["fh"].close()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _cur["fh"] = open(DATA_DIR / f"nuni_{day}.jsonl", "a", encoding="utf-8")
        _cur["day"] = day
    return _cur["fh"]


def _on_message(client, userdata, msg):
    now = time.time()
    raw = msg.payload.decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = raw                      # ESPHome 상태값 등 비JSON
    line = json.dumps({"rx_ts": round(now, 3), "topic": msg.topic,
                       "payload": payload}, ensure_ascii=False)
    fh = _file_for(now)
    fh.write(line + "\n")
    fh.flush()


def main():
    _cbv = getattr(mqtt, "CallbackAPIVersion", None)
    c = mqtt.Client(_cbv.VERSION1) if _cbv else mqtt.Client()
    c.on_message = _on_message
    c.connect(HOST, 1883, 60)
    c.subscribe("#")
    print(f"[nuni-recorder] recording all topics -> {DATA_DIR}")
    c.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()
