"""가상 센서 발행자 (레이더 + 환경).

레이더: '현재 주입 상태'(정상/무호흡/움직임)를 반영하는 원신호 1초를 롤링 버퍼에
        누적하고 radar_dsp.process_radar_buffer 로 호흡수/무호흡/움직임을 추출한다.
        (합성 신호 + 실제 DSP. 무호흡은 최근 창 진폭이 이전 대비 급감할 때 감지됨)
실물 교체 지점: _push_chunk() 대신 60GHz 모듈의 원신호 1초를 `_buf.extend(...)`.
"""
import random
import time
import collections

import numpy as np

import topics
import bus
from state_store import store
from radar_dsp import process_radar_buffer

_FS = 50
_WINDOW_S = 12
_buf = collections.deque(maxlen=_FS * _WINDOW_S)   # 최근 12초 원신호
_phase = [0.0]                                      # 호흡 위상 연속성 유지
_rng = np.random.default_rng()


def _push_chunk(seconds, apnea, motion, bpm=40.0, snr_db=15):
    """현재 상태를 반영한 원신호 1초를 롤링 버퍼에 추가."""
    n = int(_FS * seconds)
    t = np.arange(n) / _FS
    f = bpm / 60.0
    amp = 0.03 if apnea else 1.0                    # 무호흡 = 흉부 변위 급감
    sig = amp * np.sin(2 * np.pi * f * t + _phase[0])
    _phase[0] = (_phase[0] + 2 * np.pi * f * seconds) % (2 * np.pi)
    if motion:
        sig = sig + 1.5 * np.sin(2 * np.pi * 4.0 * t)   # 움직임 = 고주파 성분
    power = max(float(np.mean(sig ** 2)), 1e-9)
    sig = sig + _rng.normal(0.0, np.sqrt(power / (10 ** (snr_db / 10))), n)
    _buf.extend(sig.tolist())


def radar_step() -> dict:
    inj = store.active_injects()
    apnea = "apnea" in inj
    motion = ("motion" in inj) or (random.random() < 0.03)   # 가끔 뒤척임
    _push_chunk(1.0, apnea, motion)
    d = process_radar_buffer(list(_buf), _FS)                # 실제 DSP
    msg = topics.radar_msg(round(d["breathing_rate"], 1), round(d["movement"], 2), True)
    msg.update({"apnea": d["apnea"], "motion": d["motion"], "source": "synthetic_raw_dsp"})
    return msg


def env_step() -> dict:
    co2 = max(400, int(random.gauss(650, 80)))
    return topics.env_msg(
        round(random.gauss(22, 0.5), 1),     # 온도
        round(random.gauss(45, 3)),          # 습도
        co2,                                 # CO2
        round(random.uniform(5, 30), 1),     # 조도(야간)
    )


def run():
    """분산(MQTT) 모드에서 독립 프로세스로 실행할 때 사용."""
    while True:
        bus.publish(topics.RADAR, radar_step())
        bus.publish(topics.ENV, env_step())
        time.sleep(1)


if __name__ == "__main__":
    run()
