"""부모 음성 인텐트 인식.

기본(REAL=False): 키워드 기반 인텐트 분류 (제한된 명령 집합).
실물(REAL=True):  Vosk 오프라인 STT로 마이크 → 텍스트 → classify_text.

제한된 인텐트 집합으로 좁혀 경량 규칙으로도 높은 정확도를 확보하고,
인식 실패(unknown) 시 대시보드 터치 입력으로 대체하도록 설계.
"""
import bus
import topics
from state_store import store

REAL = False

INTENTS = {
    "baby_status_query": ["상태", "어때", "괜찮", "자고", "잘 자", "호흡", "울"],
    "humidity_control": ["건조", "습도", "가습", "촉촉"],
    "temperature_control": ["더워", "더운", "추워", "추운", "온도", "에어컨", "시원", "따뜻"],
    "light_control": ["불", "조명", "어두", "밝"],
    "air_control": ["공기", "환기", "답답", "청정"],
}

# 인텐트 → 제안 동작 (대시보드/제어 연동용)
ACTIONS = {
    "baby_status_query": "현재 아기 상태 요약 응답",
    "humidity_control": "가습기 동작",
    "temperature_control": "냉난방 조정",
    "light_control": "조명 조정",
    "air_control": "공기청정기 동작",
}


def classify_text(text):
    """텍스트 → (intent, 매칭 점수). 매칭 없으면 unknown."""
    text = (text or "").lower()
    best, best_hits = "unknown", 0
    for intent, kws in INTENTS.items():
        hits = sum(1 for k in kws if k in text)
        if hits > best_hits:
            best, best_hits = intent, hits
    return best, best_hits


def recognize_from_mic(seconds=3):
    """실물 경로: Vosk STT (미설치 시 안내)."""
    # TODO(실물): vosk KaldiRecognizer + sounddevice 로 텍스트 추출 후 classify_text
    raise NotImplementedError("Vosk STT는 여기에 구현")


def step():
    """데모용: store 에 주입된 발화를 분류해 발행 (없으면 none)."""
    phrase = None
    inj = store.active_injects()
    for kind in inj:
        if kind.startswith("voice:"):
            phrase = kind.split("voice:", 1)[1]
            break
    if not phrase:
        return topics.voice_msg("none")
    intent, _ = classify_text(phrase)
    return topics.voice_msg(intent)


def run():
    while True:
        bus.publish(topics.VOICE, step())
        import time
        time.sleep(1)


# --- 평가 ---
LABELED = [
    ("아기 지금 상태 어때?", "baby_status_query"),
    ("아기 잘 자고 있어?", "baby_status_query"),
    ("호흡 괜찮아?", "baby_status_query"),
    ("방이 너무 건조한가?", "humidity_control"),
    ("가습기 켜줘", "humidity_control"),
    ("좀 더운 것 같아", "temperature_control"),
    ("에어컨 틀어줘", "temperature_control"),
    ("불 좀 꺼줘", "light_control"),
    ("너무 어두워", "light_control"),
    ("공기가 답답해", "air_control"),
    ("환기 좀 하자", "air_control"),
]


def evaluate():
    correct = 0
    rows = []
    for text, expected in LABELED:
        pred, _ = classify_text(text)
        ok = pred == expected
        correct += ok
        rows.append((text, expected, pred, ok))
    acc = correct / len(LABELED)
    return acc, rows


def main():
    from pathlib import Path
    acc, rows = evaluate()
    print(f"음성 인텐트 인식 정확도: {acc:.1%} ({sum(r[3] for r in rows)}/{len(rows)})")
    for text, exp, pred, ok in rows:
        print(f"  [{'O' if ok else 'X'}] '{text}' → {pred} (정답 {exp})")

    results = Path("results")
    results.mkdir(exist_ok=True)
    md = [
        "# Voice Intent Result",
        "",
        f"- 제한 명령 집합 {len(INTENTS)}종, 평가 문장 {len(LABELED)}개",
        f"- 인식 정확도: {acc:.1%}",
        "- 방식: 키워드 매칭(경량). 인식 실패(unknown) 시 대시보드 입력으로 대체.",
        "",
        "| 문장 | 정답 | 예측 | 결과 |",
        "|---|---|---|---|",
    ]
    md += [f"| {t} | {e} | {p} | {'O' if ok else 'X'} |" for t, e, p, ok in rows]
    md += ["", "## 한계", "", "- 키워드 기반이라 표현이 크게 벗어나면 unknown 처리된다. 실물은 Vosk STT + 동일 규칙."]
    (results / "VOICE_INTENT_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
