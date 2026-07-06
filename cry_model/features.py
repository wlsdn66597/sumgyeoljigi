"""YAMNet 로딩 + 임베딩/울음 감지 공통 모듈.

- 울음 '감지(is_crying)'  : YAMNet 내장 AudioSet 'Baby cry, infant cry' 점수
- 울음 '이유 분류'        : YAMNet 임베딩(1024-d) → 커스텀 헤드 (train.py)
"""
import os
import csv
import hashlib
import numpy as np

import config

_yamnet = None
_classes = None

CRY_NAMES = ["Baby cry, infant cry", "Crying, sobbing"]


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


def analyze(wav16k):
    """16kHz 모노 파형 → (울음 점수, 평균 임베딩 1024-d)."""
    y, classes = load_yamnet()
    scores, emb, _ = y(wav16k.astype("float32"))
    scores, emb = scores.numpy(), emb.numpy()
    idx = [classes.index(n) for n in CRY_NAMES if n in classes]
    cry_score = float(scores[:, idx].mean(axis=0).max()) if idx else 0.0
    return cry_score, emb.mean(axis=0)


def embed(wav16k):
    return analyze(wav16k)[1]


def embed_file(path, cache_dir=None):
    """파일의 (clean) 임베딩을 디스크 캐시와 함께 반환. 재학습 시 추출 생략."""
    cache_dir = cache_dir or config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    st = os.stat(path)
    key = hashlib.md5(f"{path}{st.st_mtime}{st.st_size}".encode()).hexdigest()
    cached = os.path.join(cache_dir, key + ".npy")
    if os.path.exists(cached):
        return np.load(cached)
    import librosa
    w, _ = librosa.load(path, sr=config.SR, mono=True)
    e = embed(w)
    np.save(cached, e)
    return e
