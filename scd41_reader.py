"""SCD41 (DFRobot Gravity SEN0536) I2C → sensor/env 발행.

파이가 I2C로 SCD41(주소 0x62)에서 CO₂·온도·습도를 읽어 표준 sensor/env 로 발행한다.
상위 계층(fusion·env_control·대시보드)은 무수정.

배선(RPi 물리핀): VCC→3.3V(1) · GND→6 · SDA→GPIO2(3) · SCL→GPIO3(5)
사전 준비: I2C 활성화(완료) + .venv/bin/pip install smbus2
확인: i2cdetect -y 1  → 0x62 보이면 정상

참고: SCD41엔 조도(lux) 센서가 없다. lux는 임시 기본값으로 채우고,
      실제 조도는 이후 별도 센서/브리지로 대체한다(env_control의 dim_light용).
전환: 이 리더를 켜면 nuni-edge 의 가상 환경을 NUNI_SIM_ENV=0 으로 끈다.

실행: python scd41_reader.py   (systemd: nuni-scd41 — 센서 연결 후 enable)
"""
import json
import os
import time

from smbus2 import SMBus, i2c_msg
import paho.mqtt.client as mqtt

import topics

ADDR = 0x62
I2C_BUS = int(os.getenv("NUNI_I2C_BUS", "1"))
HOST = os.getenv("NUNI_MQTT_HOST", "localhost")
PERIOD_S = float(os.getenv("NUNI_SCD41_PERIOD_S", "5"))      # SCD41 주기측정 = 5초
DEFAULT_LUX = float(os.getenv("NUNI_DEFAULT_LUX", "15"))      # SCD41엔 조도 없음(임시)

# SCD4x 명령어
CMD_START = 0x21B1      # start_periodic_measurement
CMD_STOP = 0x3F86       # stop_periodic_measurement
CMD_READ = 0xEC05       # read_measurement
CMD_DATA_READY = 0xE4B8  # get_data_ready_status


def _crc(b0, b1):
    """Sensirion CRC-8 (poly 0x31, init 0xFF)."""
    crc = 0xFF
    for b in (b0, b1):
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def _write_cmd(bus, cmd):
    bus.i2c_rdwr(i2c_msg.write(ADDR, [(cmd >> 8) & 0xFF, cmd & 0xFF]))


def data_ready(bus):
    """측정 준비 여부. data_ready 명령 직후 곧바로 읽지 않으면 센서가
    clock-stretching으로 응답을 지연시켜 read_measurement 호출이 멈춘다."""
    _write_cmd(bus, CMD_DATA_READY)
    time.sleep(0.005)
    r = i2c_msg.read(ADDR, 3)
    bus.i2c_rdwr(r)
    d = list(r)
    if _crc(d[0], d[1]) != d[2]:
        return False
    return ((d[0] << 8) | d[1]) & 0x07FF != 0


def read_measurement(bus):
    """→ (co2_ppm, temp_c, rh_pct) 또는 준비 안 됐거나 CRC 불량 시 None."""
    if not data_ready(bus):
        return None
    _write_cmd(bus, CMD_READ)
    time.sleep(0.005)
    r = i2c_msg.read(ADDR, 9)
    bus.i2c_rdwr(r)
    d = list(r)
    for i in range(0, 9, 3):                 # 3워드 각각 CRC 검증
        if _crc(d[i], d[i + 1]) != d[i + 2]:
            return None
    co2 = (d[0] << 8) | d[1]
    temp = -45 + 175 * ((d[3] << 8) | d[4]) / 65535
    rh = 100 * ((d[6] << 8) | d[7]) / 65535
    return co2, round(temp, 1), round(rh)


def main():
    bus = SMBus(I2C_BUS)
    # 이전 측정이 돌고 있을 수 있으니 정지 후 재시작(재실행에 안전)
    _write_cmd(bus, CMD_STOP)
    time.sleep(0.5)
    _write_cmd(bus, CMD_START)
    time.sleep(5)                            # 첫 측정 준비 대기

    _cbv = getattr(mqtt, "CallbackAPIVersion", None)
    c = mqtt.Client(_cbv.VERSION1) if _cbv else mqtt.Client()
    c.connect(HOST, 1883, 60)
    c.loop_start()
    print(f"[scd41] I2C bus {I2C_BUS} @0x{ADDR:02x} -> {topics.ENV}")

    while True:
        m = read_measurement(bus)
        if m is not None:
            co2, temp, rh = m
            c.publish(topics.ENV, json.dumps(topics.env_msg(temp, rh, co2, DEFAULT_LUX)))
        time.sleep(PERIOD_S)


if __name__ == "__main__":
    main()
