"""파형 레벨 데이터 증강 (도메인 갭 보완).

깨끗한 공개 데이터 ↔ 실제 침실(원거리·잔향·생활소음) 차이를 모사한다.
train 데이터에만 적용한다.

소음: ESC-50 (★ crying_baby 클래스 제외 — 다른 아기 울음을 '소음'으로 섞으면
      라벨이 오염된다) + OpenSLR 28 등방성 배경소음
잔향: OpenSLR 28 실측 RIR + 소/중형 방 시뮬레이션 RIR (첫 채널만 사용)
파일이 없으면 합성 소음·잔향으로 대체한다.
"""
import glob
import os

import numpy as np

import config

SR = config.SR
ESC50_EXCLUDE_TARGETS = {"20"}      # ESC-50 target 20 = crying_baby

_cache = {}


def _slr28_list(name):
    """OpenSLR 28 공식 목록(rir_list / noise_list)의 wav 경로. 각 줄 마지막 토큰이
    'RIRS_NOISES/…/파일.wav' 형태의 상대 경로다. 목록이 없으면 빈 리스트."""
    lst = os.path.join(config.DATA_ROOT, "RIRS_NOISES", "real_rirs_isotropic_noises", name)
    if not os.path.exists(lst):
        return []
    out = []
    with open(lst, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = os.path.join(config.DATA_ROOT, *line.split()[-1].split("/"))
                if os.path.exists(p):
                    out.append(p)
    return out


def _noise_files():
    if "noise" not in _cache:
        files = []
        esc = os.path.join(config.DATA_ROOT, "ESC-50-master", "audio")
        for p in glob.glob(os.path.join(esc, "*.wav")):
            target = os.path.splitext(os.path.basename(p))[0].split("-")[-1]
            if target not in ESC50_EXCLUDE_TARGETS:
                files.append(p)
        files += _slr28_list("noise_list")              # 실측 등방성 배경소음 (92개)
        _cache["noise"] = sorted(files)
    return _cache["noise"]


def _rir_sets():
    """(실측 RIR, 시뮬레이션 RIR). 개수 차이(325 vs 40,000)가 커서 따로 두고 반반 뽑는다."""
    if "rir" not in _cache:
        root = os.path.join(config.DATA_ROOT, "RIRS_NOISES")
        real = _slr28_list("rir_list")                  # 실측 RIR (RWCP·REVERB·AIR, 325개)
        sim = []
        for room in ("smallroom", "mediumroom"):     # 아기방 규모
            sim += glob.glob(os.path.join(root, "simulated_rirs", room, "**", "*.wav"), recursive=True)
        _cache["rir"] = (sorted(real), sorted(sim))
    return _cache["rir"]


def _rir_files():
    real, sim = _rir_sets()
    return real + sim


def _load_first_channel(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def _mix_at_snr(w, n, snr_db):
    sig_p = np.mean(w ** 2) + 1e-9
    noise_p = np.mean(n ** 2) + 1e-9
    return w + np.sqrt(sig_p / (10 ** (snr_db / 10)) / noise_p) * n


def add_noise(w, rng):
    snr_db = rng.uniform(5, 20)
    files = _noise_files()
    if not files:
        return _mix_at_snr(w, rng.standard_normal(len(w)), snr_db)
    n = _load_first_channel(files[rng.integers(len(files))])
    if len(n) < len(w):
        n = np.tile(n, int(np.ceil(len(w) / max(len(n), 1))))
    start = rng.integers(0, len(n) - len(w) + 1)
    return _mix_at_snr(w, n[start:start + len(w)], snr_db)


def reverb(w, rng):
    real, sim = _rir_sets()
    files = real if (real and (not sim or rng.random() < 0.5)) else sim
    if files:
        ir = _load_first_channel(files[rng.integers(len(files))])
        ir = ir[np.argmax(np.abs(ir)):]              # 직접음 도달 전 지연 제거
    else:
        ir_len = int(SR * rng.uniform(0.2, 0.5))
        ir = np.exp(-np.linspace(0, 6, ir_len)) * (rng.random(ir_len) - 0.5)
        ir[0] = 1.0
    out = np.convolve(w, ir)[: len(w)]
    return out * (np.max(np.abs(w)) + 1e-9) / (np.max(np.abs(out)) + 1e-9)   # 원 음량 유지


def distance(w, rng):
    """원거리 모사: 게인 감쇠 + 간이 저역통과."""
    k = int(rng.integers(3, 9))
    return rng.uniform(0.3, 0.7) * np.convolve(w, np.ones(k) / k, mode="same")


def time_stretch(w, rng):
    import librosa
    return librosa.effects.time_stretch(w, rate=rng.uniform(0.9, 1.1))


def pitch_shift(w, rng):
    """주의: 울음 음높이(F0)는 이유 단서일 수 있다(통증 울음은 높음). --no-pitch로 끌 수 있다."""
    import librosa
    return librosa.effects.pitch_shift(w, sr=SR, n_steps=rng.uniform(-1, 1))


def random_augment(w, rng, allow_pitch=True):
    """무작위로 몇 가지 증강을 조합해 적용 (rng: np.random.Generator — 재현성)."""
    ops = [add_noise, reverb, distance, time_stretch] + ([pitch_shift] if allow_pitch else [])
    out = np.asarray(w, dtype=np.float32)
    for i in rng.permutation(len(ops)):
        if rng.random() < 0.5:
            out = ops[i](out, rng)
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)
