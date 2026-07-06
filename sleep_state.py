"""수면/각성 상태 분류 (일상 모니터링 헤드라인).

레이더 움직임 + 울음 '감지'(is_crying)로 일상 상태를 도출한다.
울음 '이유' 분류에는 의존하지 않는다(감지만 사용).

state: calm_sleep(안정 수면) / restless(뒤척임) / awake(각성) / unknown(재실 없음)
"""
MOVE_RESTLESS = 0.4   # 이 이상이면 뒤척임
MOVE_AWAKE = 0.7      # 이 이상이면 각성

KO_LABEL = {
    "calm_sleep": "안정 수면",
    "restless": "뒤척임",
    "awake": "각성",
    "unknown": "재실 없음",
}


def classify(radar, cry):
    """→ (state, reason)"""
    if not radar or not radar.get("presence"):
        return "unknown", "재실 신호 없음"

    crying = bool(cry and cry.get("is_crying"))
    movement = radar.get("movement", 0.0)

    if crying or movement >= MOVE_AWAKE:
        return "awake", "울음 또는 큰 움직임"
    if movement >= MOVE_RESTLESS:
        return "restless", "움직임 증가"
    return "calm_sleep", "규칙적 호흡·낮은 움직임"
