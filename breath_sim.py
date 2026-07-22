"""인형 호흡 시뮬레이터 — SG90 서보로 영아 호흡(가슴 오르내림)을 모사.

서보 혼에 막대를 달아 인형 가슴/이불 밑에 두면, 설정한 호흡수(BPM)로
몇 mm씩 사인파처럼 부드럽게 오르내린다. 레이더 검증의 ground truth:
BPM을 40으로 설정하고 레이더가 40을 읽는지 비교한다(다이얼 데모).

배선(RPi 물리핀): 갈색→GND(9번), 빨강→5V(2번), 주황→GPIO18(12번)
사전 준비: .venv/bin/pip install lgpio   (Debian 13 표준 GPIO 라이브러리)

실행: python breath_sim.py --bpm 40
원격 제어(MQTT, demo/breath 토픽):
  mosquitto_pub -t demo/breath -m '{"bpm": 55}'      # 호흡수 변경
  mosquitto_pub -t demo/breath -m '{"apnea_s": 10}'  # 10초 무호흡(정지) 주입
"""
import argparse
import json
import math
import os
import time

import lgpio

GPIO = int(os.getenv("NUNI_SERVO_GPIO", "18"))
CENTER_US = 1500      # 중립 펄스폭(us)
AMP_US = 120          # 진폭(us) ≈ ±11도 — 가슴 몇 mm 변위 수준
TICK = 0.02           # 50Hz 갱신

state = {"bpm": 40.0, "apnea_until": 0.0}


def _mqtt_listen():
    """demo/breath 토픽으로 bpm 변경·무호흡 주입을 받는다(브로커 없으면 생략)."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("[breath-sim] paho 없음 — MQTT 원격 제어 비활성")
        return

    def on_msg(c, u, m):
        try:
            d = json.loads(m.payload.decode())
        except (ValueError, UnicodeDecodeError):
            return
        if "bpm" in d:
            state["bpm"] = max(5.0, min(90.0, float(d["bpm"])))
            print(f"[breath-sim] bpm -> {state['bpm']}")
        if "apnea_s" in d:
            state["apnea_until"] = time.time() + float(d["apnea_s"])
            print(f"[breath-sim] apnea {d['apnea_s']}s 주입")

    _cbv = getattr(mqtt, "CallbackAPIVersion", None)
    c = mqtt.Client(_cbv.VERSION1) if _cbv else mqtt.Client()
    c.on_message = on_msg
    try:
        c.connect(os.getenv("NUNI_MQTT_HOST", "localhost"), 1883, 60)
        c.subscribe("demo/breath")
        c.loop_start()
        print("[breath-sim] MQTT 원격 제어 대기 (demo/breath)")
    except OSError:
        print("[breath-sim] 브로커 연결 실패 — CLI 설정값으로만 동작")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpm", type=float, default=40.0, help="호흡수(회/분), 영아 30~60")
    args = ap.parse_args()
    state["bpm"] = args.bpm

    h = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(h, GPIO)

    _mqtt_listen()
    print(f"[breath-sim] GPIO{GPIO}, {state['bpm']}회/분 시작 (Ctrl+C 종료)")
    phase = 0.0
    try:
        while True:
            now = time.time()
            if now < state["apnea_until"]:
                pulse = CENTER_US - AMP_US          # 호기 위치에서 정지(숨 멈춤)
            else:
                phase += 2 * math.pi * (state["bpm"] / 60.0) * TICK
                pulse = CENTER_US + AMP_US * math.sin(phase)
            # 서보 펄스: 50Hz 주기, 듀티 = pulse/20000
            lgpio.tx_servo(h, GPIO, int(pulse), 50)
            time.sleep(TICK)
    except KeyboardInterrupt:
        pass
    finally:
        lgpio.tx_servo(h, GPIO, 0, 50)              # 서보 릴리즈
        lgpio.gpiochip_close(h)


if __name__ == "__main__":
    main()
