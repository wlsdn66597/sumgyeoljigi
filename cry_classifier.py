"""울음 분류기.

기본(REAL=False): 시뮬레이션. 하드웨어/모델 없이 파이프라인 검증용.
실물(REAL=True):  마이크 → YAMNet 임베딩 → 커스텀 헤드 추론 (아래 TODO 구현)
"""
import os
import sys
import random
import time

import topics
import bus
from state_store import store

CLASSES = ["hungry", "sleepy", "discomfort"]
REAL = False              # True: 학습된 YAMNet 모델 + 마이크로 실제 분류
CRY_THRESHOLD = 0.5       # 울음 감지 임계값 (YAMNet cry_score)

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


def classify_yamnet() -> dict:
    """마이크 최근 1.5초 → YAMNet 울음 감지 + 이유 분류."""
    global _model, _mic
    if _model is None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cry_model"))
        from infer import CryModel, Mic
        _model = CryModel.load(os.path.join(os.path.dirname(__file__), "cry_model", "artifacts"))
        _mic = Mic(); _mic.start()

    cry_score, reason, conf = _model.predict(_mic.get_last(1.5))
    is_crying = cry_score > CRY_THRESHOLD
    return topics.cry_msg(is_crying, reason if is_crying else "none",
                          round(conf if is_crying else cry_score, 2))


def run():
    while True:
        bus.publish(topics.CRY, step())
        time.sleep(1)


if __name__ == "__main__":
    run()
