"""울음 분류기.

기본(REAL=False): 시뮬레이션. 하드웨어/모델 없이 파이프라인 검증용.
실물(REAL=True):  마이크 → YAMNet(tflite) 내장 클래스로 울음 유무 직접 판정.
                  커스텀 이유 분류 헤드는 학습 데이터가 없어 사용하지 않는다
                  (이유 분류는 실험적이라는 프로젝트 방침과 일치 — cls는 항상
                  "none"이며, 이유 대신 cry_context.py의 맥락 상관을 사용).
"""
import os
import random
import time

import topics
import bus
from state_store import store

CLASSES = ["hungry", "sleepy", "discomfort"]      # REAL=False 시뮬레이션 전용
REAL = os.getenv("NUNI_CRY_REAL", "0") == "1"
CRY_THRESHOLD = float(os.getenv("NUNI_CRY_THRESHOLD", "0.3"))   # YAMNet cry_score
MIC_WINDOW_S = 1.5

_model = None
_mic = None


def step() -> dict:
    if REAL:
        return classify_yamnet()

    inj = store.active_injects()
    if "cry" in inj:                                   # 울음 시나리오
        return topics.cry_msg(True, random.choice(CLASSES),
                              round(random.uniform(0.7, 0.95), 2))
    # 평상시: 낮은 확률로만 저신뢰 오검출 발생 -> 융합 규칙이 걸러내야 함
    if random.random() < 0.03:
        return topics.cry_msg(True, random.choice(CLASSES),
                              round(random.uniform(0.55, 0.65), 2))
    return topics.cry_msg(False, "none", round(random.uniform(0, 0.3), 2))


MIC_NATIVE_SR = int(os.getenv("NUNI_MIC_SR", "48000"))   # USB 마이크가 16kHz 직캡처 미지원(하드웨어 기본이 다름)


def classify_yamnet() -> dict:
    """마이크 최근 1.5초(마이크 고유 레이트) → 16kHz 리샘플 → YAMNet 내장
    클래스로 울음 유무 판정 (이유 분류 없음)."""
    global _model, _mic
    if _model is None:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cry_model"))
        from infer import Mic                  # sounddevice 링버퍼만 재사용(텐서플로우 불요)
        from yamnet_cry import YamnetCry, resample_linear
        _model = YamnetCry.load()
        _mic = Mic(sr=MIC_NATIVE_SR); _mic.start()
        globals()["_resample"] = resample_linear

    raw = _mic.get_last(MIC_WINDOW_S)
    x16 = _resample(raw, MIC_NATIVE_SR)
    cry_score, label = _model.predict(x16)
    is_crying = cry_score > CRY_THRESHOLD
    return topics.cry_msg(is_crying, "none", round(cry_score, 2))


def run():
    while True:
        bus.publish(topics.CRY, step())
        time.sleep(1)


if __name__ == "__main__":
    run()
