"""YAMNet 로딩 + 프레임/세그먼트 임베딩 공통 모듈.

- 울음 '감지'  : YAMNet 내장 AudioSet 'Baby cry, infant cry' 계열 점수 (프레임별)
- 울음 '이유'  : 세그먼트 임베딩(연속 SEG_FRAMES 프레임 평균, 1024-d) → 3클래스 헤드

학습과 추론의 입력 길이를 맞추는 게 핵심이다. 실시간 추론은 마이크 1.5초 창
(= YAMNet 프레임 2개)을 보므로, 학습도 클립 전체 평균이 아니라 같은 길이의
세그먼트 단위로 만든다. 울음 점수가 낮은 세그먼트(무음·잡음)는 버린다.
"""
import csv
import hashlib
import os

import numpy as np

import config

_yamnet = None
_classes = None

CRY_NAMES = ["Baby cry, infant cry", "Crying, sobbing"]
_CACHE_VER = "v2"


def load_yamnet():
    """YAMNet 모델과 AudioSet 클래스 이름 목록을 로드(캐시)."""
    global _yamnet, _classes
    if _yamnet is None:
        import tensorflow_hub as hub
        _yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        cmap = _yamnet.class_map_path().numpy().decode()
        with open(cmap) as f:
            _classes = [row["display_name"] for row in csv.DictReader(f)]
    return _yamnet, _classes


def frames(wav16k):
    """16kHz 모노 파형 → (프레임별 울음 점수[T], 프레임 임베딩[T,1024])."""
    y, classes = load_yamnet()
    scores, emb, _ = y(np.asarray(wav16k, dtype=np.float32))
    scores, emb = scores.numpy(), emb.numpy()
    idx = [classes.index(n) for n in CRY_NAMES if n in classes]
    cry = scores[:, idx].max(axis=1) if idx else np.zeros(len(scores), np.float32)
    return cry.astype(np.float32), emb.astype(np.float32)


def segments(cry, emb, seg=config.SEG_FRAMES, cry_min=config.CRY_MIN):
    """연속 seg프레임 슬라이딩(간격 1프레임) 평균 → (세그먼트 임베딩[n,1024], 울음 점수[n]).
    cry_min 미만 세그먼트는 제외(None이면 전부 유지)."""
    T = len(emb)
    if T == 0:
        return np.zeros((0, emb.shape[1] if emb.ndim == 2 else 1024), np.float32), np.zeros(0, np.float32)
    if T < seg:
        X, c = emb.mean(axis=0, keepdims=True), np.array([cry.mean()])
    else:
        X = np.stack([emb[i:i + seg].mean(axis=0) for i in range(T - seg + 1)])
        c = np.array([cry[i:i + seg].mean() for i in range(T - seg + 1)])
    if cry_min is not None:
        keep = c >= cry_min
        X, c = X[keep], c[keep]
    return X.astype(np.float32), c.astype(np.float32)


def _cache_path(path, tag=""):
    st = os.stat(path)
    key = hashlib.md5(f"{_CACHE_VER}|{path}|{st.st_mtime}|{st.st_size}|{tag}".encode()).hexdigest()
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, key + ".npz")


def load_wav(path):
    import librosa
    w, _ = librosa.load(path, sr=config.SR, mono=True)
    return w


def file_frames(path, aug=None, tag=""):
    """파일의 프레임 (울음 점수, 임베딩)을 디스크 캐시와 함께 반환.
    aug(wav)->wav 를 주면 증강본을 계산한다(tag로 캐시 구분)."""
    cached = _cache_path(path, tag)
    if os.path.exists(cached):
        z = np.load(cached)
        return z["cry"], z["emb"]
    w = load_wav(path)
    if aug is not None:
        w = aug(w)
    cry, emb = frames(w)
    np.savez(cached, cry=cry, emb=emb)
    return cry, emb


def file_scores(path):
    """파일의 YAMNet 전체 클래스 점수[T,521](float16)와 평균 임베딩[1024]을 캐시와 함께 반환.
    소리 종류 분류(sound_events) 평가용."""
    cached = _cache_path(path, "scores")
    if os.path.exists(cached):
        z = np.load(cached)
        return z["scores"].astype(np.float32), z["emb_mean"]
    y, _ = load_yamnet()
    scores, emb, _ = y(load_wav(path).astype(np.float32))
    scores = scores.numpy().astype(np.float16)
    emb_mean = emb.numpy().mean(axis=0).astype(np.float32)
    np.savez(cached, scores=scores, emb_mean=emb_mean)
    return scores.astype(np.float32), emb_mean


# --- 하위 호환 (기존 코드용) ------------------------------------------------
def analyze(wav16k):
    """16kHz 모노 파형 → (울음 점수, 평균 임베딩 1024-d)."""
    cry, emb = frames(wav16k)
    return float(cry.mean()) if len(cry) else 0.0, emb.mean(axis=0)


def embed(wav16k):
    return analyze(wav16k)[1]
