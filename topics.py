"""MQTT 토픽 규격서 (코드로 관리하는 계약서).

모든 모듈은 이 토픽 이름과 메시지 빌더만 사용한다.
가상 센서든 실물 센서든 동일한 토픽/포맷으로 발행하므로,
발행자만 교체하면 상위 계층(AI·융합·대시보드)은 그대로 동작한다.
"""
import time

# --- 토픽 이름 ---
RADAR = "sensor/radar"        # 60GHz 레이더 (호흡·움직임·재실)
ENV = "sensor/env"            # 환경 센서 (온습도·CO2·조도)
CRY = "audio/cry"             # 울음 분류 결과
VOICE = "audio/voice"         # 부모 음성 인텐트
FUSION_STATE = "fusion/state"  # 융합 판단 현재 상태
ALERT = "fusion/alert"        # 경보 이벤트
CONTROL = "control/action"    # 선제 환경 제어 권고
TOF = "sensor/tof"            # ToF 거리 센서(VL53L1X, ESP32) — 침대 난간 높이에서 수평으로 본다


def now() -> float:
    return time.time()


# --- 메시지 빌더 (스키마) ---
def radar_msg(breathing_rate: float, movement: float, presence: bool) -> dict:
    """breathing_rate: 회/분, movement: 0~1, presence: 재실 여부"""
    return {"breathing_rate": breathing_rate, "movement": movement,
            "presence": presence, "ts": now()}


def env_msg(temp: float, humidity: float, co2: int, lux: float) -> dict:
    return {"temp": temp, "humidity": humidity, "co2": co2, "lux": lux, "ts": now()}


def cry_msg(is_crying: bool, cls: str, confidence: float) -> dict:
    """cls: hungry/sleepy/discomfort/none, confidence: 0~1"""
    return {"is_crying": is_crying, "cls": cls, "confidence": confidence, "ts": now()}


def tof_msg(distance_mm: int, detected: bool) -> dict:
    """distance_mm: 측정 거리, detected: 난간 높이 빔에 무언가 있음(아기가 일어섬). 판단은 ESP32가 한다."""
    return {"distance_mm": distance_mm, "detected": detected, "ts": now()}


def voice_msg(intent: str) -> dict:
    return {"intent": intent, "ts": now()}
