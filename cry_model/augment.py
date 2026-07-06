"""파형 레벨 데이터 증강 (도메인 갭 보완).

깨끗한 공개 데이터 ↔ 실제 침실(원거리·잔향·생활소음) 차이를 모사한다.
train 데이터에만 적용한다.
"""
import os
import glob
import numpy as np
import librosa

import config

SR = 16000

_noise_files = None
_rir_files = None


def _list(dir_, cache_attr):
    globals()[cache_attr] = globals().get(cache_attr)
    if globals()[cache_attr] is None:
        globals()[cache_attr] = glob.glob(os.path.join(dir_, "**", "*.wav"), recursive=True) if os.path.isdir(dir_) else []
    return globals()[cache_attr]


def add_real_noise(w, snr_db=None):
    """ESC-50/MUSAN 등 실제 소음을 SNR 맞춰 섞음 (없으면 백색소음 fallback)."""
    files = _list(config.NOISE_DIR, "_noise_files")
    if not files:
        return add_noise(w, snr_db)
    n, _ = librosa.load(np.random.choice(files), sr=SR, mono=True)
    if len(n) < len(w):
        n = np.tile(n, int(np.ceil(len(w) / len(n))))
    n = n[: len(w)]
    snr_db = snr_db if snr_db is not None else np.random.uniform(5, 20)
    sig_p = np.mean(w ** 2) + 1e-9
    noise_p = np.mean(n ** 2) + 1e-9
    k = np.sqrt(sig_p / (10 ** (snr_db / 10)) / noise_p)
    return w + k * n


def real_reverb(w):
    """실측 RIR 컨볼루션 (없으면 합성 reverb fallback)."""
    files = _list(config.RIR_DIR, "_rir_files")
    if not files:
        return reverb(w)
    ir, _ = librosa.load(np.random.choice(files), sr=SR, mono=True)
    out = np.convolve(w, ir)[: len(w)]
    return out / (np.max(np.abs(out)) + 1e-9)


def add_noise(w, snr_db=None):
    snr_db = snr_db if snr_db is not None else np.random.uniform(5, 20)
    sig_p = np.mean(w ** 2) + 1e-9
    noise = np.random.randn(len(w))
    noise_p = np.mean(noise ** 2) + 1e-9
    k = np.sqrt(sig_p / (10 ** (snr_db / 10)) / noise_p)
    return w + k * noise


def reverb(w, decay=None):
    decay = decay if decay is not None else np.random.uniform(0.2, 0.5)
    ir_len = int(SR * decay)
    ir = np.exp(-np.linspace(0, 6, ir_len)) * (np.random.rand(ir_len) - 0.5)
    ir[0] = 1.0
    out = np.convolve(w, ir)[: len(w)]
    return out / (np.max(np.abs(out)) + 1e-9)


def distance(w):
    """원거리 모사: 게인 감쇠 + 간이 저역통과."""
    gain = np.random.uniform(0.3, 0.7)
    k = np.random.randint(3, 9)
    lp = np.convolve(w, np.ones(k) / k, mode="same")
    return gain * lp


def time_stretch(w):
    return librosa.effects.time_stretch(w, rate=np.random.uniform(0.9, 1.1))


def pitch_shift(w):
    return librosa.effects.pitch_shift(w, sr=SR, n_steps=np.random.uniform(-2, 2))


def random_augment(w, sr=SR):
    """무작위로 몇 가지 증강을 조합해 적용.

    noise/ · rir/ 폴더에 실제 파일이 있으면 실측 소음·잔향을,
    없으면 합성 버전을 자동으로 사용한다.
    """
    ops = [add_real_noise, real_reverb, distance, time_stretch, pitch_shift]
    np.random.shuffle(ops)
    out = w.astype(np.float32)
    for op in ops:
        if np.random.rand() < 0.5:
            try:
                out = op(out)
            except Exception:
                pass
    return out.astype(np.float32)
