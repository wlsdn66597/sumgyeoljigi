"""아기방 소리 종류 분류 — YAMNet 내장 클래스를 우리 버킷으로 묶는다(학습 없음).

YAMNet 한 번의 추론이 AudioSet 521개 클래스 점수를 모두 내므로, 울음 감지와 같은
추론 결과에서 다른 소리 점수만 더 읽는다. 추가 연산·추가 모델이 없다.

버킷 (시스템에서 쓰는 곳)
  cry      울음            → 알림·달래기
  whimper  칭얼거림        → 울기 전 전조, 미리 달래기   (검증 데이터 없음)
  laugh    웃음·옹알이     → '깨서 노는 중', 알림 없음
  cough    기침·재채기·쌕쌕 → 밤 기침 추세 리포트
  other    그 외            → 무시 (오경보 방지)
"""
import numpy as np

BUCKETS = {
    "cry": ["Baby cry, infant cry", "Crying, sobbing"],
    "whimper": ["Whimper"],
    "laugh": ["Baby laughter", "Laughter", "Giggle", "Babbling"],
    "cough": ["Cough", "Sneeze", "Wheeze"],
}
EVENT_BUCKETS = list(BUCKETS)          # other 제외
# 버킷별 임계값. 울음은 실시간 울음 임계값과 같은 0.3. 기침은 숨소리·코골이·물 마시는
# 소리를 기침으로 잡는 오경보가 많아 0.5로 올렸다(eval_sound_events: 정밀도 0.61→0.72,
# 재현율 0.89→0.85). 기침 임계값은 평가셋을 보고 정한 값이라 약간 낙관적일 수 있다.
THRESHOLDS = {"cry": 0.3, "whimper": 0.3, "laugh": 0.3, "cough": 0.5}


def bucket_index(class_names):
    """버킷 → YAMNet 클래스 인덱스 목록."""
    idx = {n: i for i, n in enumerate(class_names)}
    missing = [n for names in BUCKETS.values() for n in names if n not in idx]
    if missing:
        raise ValueError(f"YAMNet 클래스 목록에 없음: {missing}")
    return {b: [idx[n] for n in names] for b, names in BUCKETS.items()}


def frame_bucket_scores(scores, bidx):
    """프레임 점수[T,521] → 버킷 점수[T,n_bucket] (버킷 안 클래스 중 최댓값)."""
    return np.stack([scores[:, bidx[b]].max(axis=1) for b in EVENT_BUCKETS], axis=1)


def clip_bucket_scores(scores, bidx):
    """클립 단위 버킷 점수: 프레임(≈1초) 중 최댓값. 기침처럼 짧은 소리도 놓치지 않는다."""
    if len(scores) == 0:
        return np.zeros(len(EVENT_BUCKETS), np.float32)
    return frame_bucket_scores(scores, bidx).max(axis=0)


def decide(bucket_scores, thresholds=THRESHOLDS):
    """버킷 점수 → 하나의 라벨.

    울음 우선: 울음 점수가 임계값을 넘으면 다른 소리보다 먼저 울음으로 판정한다.
    우는 아기를 '웃음(노는 중)'으로 보면 알림이 막히므로, 오류의 방향이 비대칭이다.
    (평가: 울음→웃음 오판 43→13, 울음 재현율 0.82→0.89)
    그 외에는 임계값을 넘은 버킷 중 최고 점수, 하나도 없으면 other.
    """
    s = dict(zip(EVENT_BUCKETS, bucket_scores))
    if s["cry"] >= thresholds["cry"]:
        return "cry"
    passed = {b: v for b, v in s.items() if v >= thresholds[b]}
    return max(passed, key=passed.get) if passed else "other"
