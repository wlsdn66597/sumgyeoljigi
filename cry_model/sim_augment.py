"""증강을 넣어 가상 행동 라벨 시뮬레이션(sim_synthetic.py, 7차)을 다시 돌리고, 소리를 비교하는 공간을 바꿔 본다(8차).

    python sim_augment.py prep --workers 12   # 울음 2,521개 × (원본 + 방 조건 변형 4개), 앞 20초 → 음향 요약 + 파이식 YAMNet 임베딩
    python sim_augment.py prep --workers 12 --ops stretch,pitch --tag mild   # 울음 변형(속도·음높이만) 4개
    python sim_augment.py run                 # → DATA/runs/synthetic/AUGMENT_REPORT.md

  GPU 서버: NUNI_CRY_DATA=~/cry/data CRY_JOINT=~/cry/data/joint2 (GPU는 안 쓰고 CPU 병렬, 각 2~3분)

증강: augment.py와 같은 변형을 JOINT/aug(ESC-50 아기 울음 제외 + OpenSLR 28 잡음·실측/시뮬레이션 잔향)로 만든다.
  방 조건 증강: 잡음 SNR 5~20dB / 잔향(실측·시뮬레이션 반반) / 거리감(작게 + 고역 감쇠) / 속도 0.9~1.1배 / 음높이 ±1반음
  울음 변형 증강: 속도 / 음높이만.  각 50% 확률, 무작위 순서(하나도 안 걸리면 하나를 고른다).
  변형은 원래 울음의 라벨을 물려받는다. 부모 답은 '그 울음'에 대한 것이라 기기에서 증강해도 같은 답을 붙이기 때문이다.
  시험 울음(일반 모델의 시험 그룹, 개인화의 아기별 뒤쪽 절반)의 변형은 학습에 쓰지 않는다.

1) 일반 모델(7차와 같은 그룹 단위 학습 60 / 검증 20 / 시험 20): 학습·검증 울음에 변형을 더하면 오르나.
   시험은 원본 그대로와 '조건이 바뀐' 시험(시험 울음마다 변형 1개) 두 가지.
2) 개인화(Enes 아기 24명): 아기마다 시간 앞 절반 = 부모 답을 받은 울음, 뒤 절반 = 처음 듣는 울음.
   이전 답 최근 5개 / 10개 / 앞 절반 전부. 방법: proto(행동별 평균과 코사인), knn(가장 비슷한 이전 울음 투표, 파이 모듈과 같음).
   비교 공간
     YAMNet 16차원(파이 기본) — 소리 '종류'를 가르도록 학습된 임베딩을 공개 울음으로 줄인 것
     음향 요약 예측 — YAMNet 임베딩 → 음향 요약 6개 선형 변환을 공개 데이터로 배운 것(원인 라벨 안 씀). 원본만 / 증강까지
     YAMNet 16 + 음향 요약 예측(증강)
     실제 음향 요약 6개 — 가상 라벨을 만든 바로 그 공간이라 가상 세계에서는 상한(순환)
   라벨: 가상(② 공통 규칙, ③ 아기마다 다른 규칙) / 실제 Enes 라벨(울음을 멈추게 한 행동; 아기 안 라벨 섞기 귀무와 비교).
"""
import argparse
import os
import signal
import sys
import threading
import time

if len(sys.argv) > 1 and sys.argv[1] == "prep":            # 작업자 하나에 코어 하나
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expanduser(os.environ.get("NUNI_CRY_DATA", "D:/datasets/cry"))
JOINT = os.path.expanduser(os.environ.get("CRY_JOINT", os.path.join(DATA, "joint")))
ART = os.path.join(HERE, "artifacts")
OUT = os.path.join(DATA, "augsim")
RUNS = os.path.join(DATA, "runs", "synthetic")
SR, WIN, HOP, CRY_THR = 16000, 15600, 16000, 0.3
CLIP_S, N_AUG = 20, 4
OPS = ["noise", "reverb", "distance", "stretch", "pitch"]
OPS_KO = ["잡음", "잔향", "거리감", "속도", "음높이"]
ACOUSTIC_KO = ["음높이", "음높이 변동", "크기", "크기 변동", "소리 나는 비율", "음색(밝기)"]


# --- 1. 울음 목록 --------------------------------------------------------------------
def items():
    """sim_synthetic.py pool과 같은 범위: Enes(조각을 울음 단위로 묶음) + 공개 데이터."""
    e = pd.read_csv(os.path.join(DATA, "enes3", "meta.csv"), dtype={"date": str, "time": str})
    dt = (e.date.str.zfill(8) + e.time.str.zfill(4)).where(e.date.str.len() <= 8, e.date + e.time)
    ts = pd.to_datetime(dt.str[:12], format="%d%m%Y%H%M", errors="coerce")
    e["ts"] = (ts - pd.Timestamp(0)).dt.total_seconds() + e.t0
    rows = []
    for b, g in e.groupby("bout"):
        g = g.sort_values("ep")
        rows.append({"source": "enes", "group": f"enes:{g.baby.iloc[0]}", "baby": g.baby.iloc[0], "t": g.ts.min(),
                     "stop": g.stop.iloc[0], "bout": b,
                     "paths": "|".join(os.path.join(DATA, "enes3", "wav", i + ".wav") for i in g.id)})
    j = pd.read_csv(os.path.join(JOINT, "meta.csv"))
    u = pd.read_csv(os.path.join(JOINT, "meta_unlab.csv"))
    for r in pd.concat([j[j.dataset != "enes"], u[u.dataset != "enes_other"]]).itertuples():
        rows.append({"source": r.dataset, "group": r.group, "baby": r.group if r.dataset == "corvin" else "",
                     "t": np.nan, "stop": "", "bout": -1, "paths": os.path.join(JOINT, r.path)})
    return pd.DataFrame(rows)


def load_clip(paths):
    xs, n = [], 0
    for p in paths.split("|"):
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        xs.append(x)
        n += len(x)
        if n >= CLIP_S * SR:
            break
    return np.concatenate(xs)[: CLIP_S * SR]


def acoustic(x):
    """sim_synthetic.acoustic과 같은 음향 요약(배열 입력): 음높이(반음) 중앙값·사분위 범위, 크기(dB)·변동,
    소리 나는 비율, 음색(스펙트럼 중심)."""
    import librosa
    if len(x) < SR:
        x = np.pad(x, (0, SR - len(x)))
    r = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
    f = librosa.yin(x, fmin=150, fmax=1000, sr=SR, frame_length=1024, hop_length=256)
    c = librosa.feature.spectral_centroid(y=x, sr=SR, n_fft=1024, hop_length=256)[0]
    k = min(len(r), len(f), len(c))
    r, f, c = r[:k], f[:k], c[:k]
    db = 20 * np.log10(r + 1e-9)
    act = db > db.max() - 30
    a = act if act.any() else np.ones(k, bool)
    semi = 12 * np.log2(f[a] / 100)
    return np.nan_to_num([np.median(semi), np.subtract(*np.percentile(semi, [75, 25])), np.mean(db[a]),
                          np.std(r[a]) / (np.mean(r[a]) + 1e-9), act.mean(), np.log(np.median(c[a]) + 1)])


class Augmenter:
    """augment.py와 같은 변형. 재료는 joint_build.py가 만든 JOINT/aug/{noise,rir_real,rir_sim}."""

    def __init__(self, ops=OPS):
        import glob
        d = os.path.join(JOINT, "aug")
        self.noise = sorted(glob.glob(os.path.join(d, "noise", "*.wav")))
        self.rir = [sorted(glob.glob(os.path.join(d, k, "*.wav"))) for k in ("rir_real", "rir_sim")]
        assert self.noise and all(self.rir), f"증강 재료 없음: {d}"
        self.allowed = np.array([o in ops for o in OPS])

    @staticmethod
    def _read(p):
        return sf.read(p, dtype="float32", always_2d=True)[0][:, 0]

    def __call__(self, x, rng):
        """반환: (변형 파형, 쓴 변형 비트마스크 — OPS 순서)."""
        import librosa
        from scipy.signal import fftconvolve
        order = rng.permutation(len(OPS))
        pick = (rng.random(len(OPS)) < 0.5) & self.allowed
        if not pick.any():
            cand = np.flatnonzero(self.allowed)
            pick[cand[rng.integers(len(cand))]] = True
        out, used = np.asarray(x, np.float32), 0
        for i in order:
            if not pick[i]:
                continue
            used |= 1 << i
            if OPS[i] == "noise":
                n = self._read(self.noise[rng.integers(len(self.noise))])
                if len(n) < len(out):
                    n = np.tile(n, int(np.ceil(len(out) / max(len(n), 1))))
                s = rng.integers(0, len(n) - len(out) + 1)
                n = n[s:s + len(out)]
                snr = rng.uniform(5, 20)
                out = out + np.sqrt((np.mean(out ** 2) + 1e-9) / 10 ** (snr / 10) / (np.mean(n ** 2) + 1e-9)) * n
            elif OPS[i] == "reverb":
                pool = self.rir[int(rng.random() < 0.5)]
                h = self._read(pool[rng.integers(len(pool))])
                h = h[np.argmax(np.abs(h)):][:SR]                          # 직접음 도달 전 지연 제거, 1초까지
                y = fftconvolve(out, h)[: len(out)]
                out = y * (np.abs(out).max() + 1e-9) / (np.abs(y).max() + 1e-9)   # 원 음량 유지
            elif OPS[i] == "distance":
                k = int(rng.integers(3, 9))
                out = rng.uniform(0.3, 0.7) * np.convolve(out, np.ones(k) / k, mode="same")
            elif OPS[i] == "stretch":
                out = librosa.effects.time_stretch(out, rate=rng.uniform(0.9, 1.1))
            else:
                out = librosa.effects.pitch_shift(out, sr=SR, n_steps=rng.uniform(-1, 1))
        peak = np.abs(out).max() + 1e-9
        return (out / peak if peak > 1 else out).astype(np.float32), used


def _interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    return Interpreter(model_content=open(path, "rb").read(), num_threads=1)   # 한글 경로 대비 model_content


class Yam:
    """파이와 같은 YAMNet(TFLite, 0.975초 창 1초 간격). 울음 판정 창(울음 점수 0.3 이상)의 평균 임베딩."""

    def __init__(self):
        it = _interpreter(os.path.join(ART, "yamnet_emb.tflite"))
        it.allocate_tensors()
        self.it, self.inp = it, it.get_input_details()[0]["index"]
        self.out = {int(d["shape"][-1]): d["index"] for d in it.get_output_details()}

    def clip(self, x):
        if len(x) < WIN:
            x = np.pad(x, (0, WIN - len(x)))
        embs, cry = [], []
        for s in range(0, len(x) - WIN + 1, HOP):
            self.it.set_tensor(self.inp, x[s:s + WIN].astype(np.float32))
            self.it.invoke()
            embs.append(self.it.get_tensor(self.out[1024]).reshape(-1).copy())
            sc = self.it.get_tensor(self.out[521]).reshape(-1)
            cry.append(max(sc[19], sc[20]))
        embs, g = np.array(embs), np.array(cry) >= CRY_THR
        return (embs[g] if g.any() else embs).mean(0), float(g.mean())


_W = {}


def _init(ops, seed0):
    _W["aug"], _W["yam"], _W["seed0"] = Augmenter(ops), Yam(), seed0


def _prep_one(job):
    k, paths = job
    rng = np.random.default_rng(_W["seed0"] + k)
    x = load_clip(paths)
    versions, ops = [x], [0]
    for _ in range(N_AUG):
        v, used = _W["aug"](x, rng)
        versions.append(v)
        ops.append(used)
    emb = [_W["yam"].clip(v) for v in versions]
    return (k, np.array([acoustic(v) for v in versions], np.float32), np.array([e for e, _ in emb], np.float32),
            np.array([g for _, g in emb], np.float32), np.array(ops, np.int8))


def mem_guard(min_gb=3.0):
    """공용 서버 보호: 서버 전체의 남은 메모리가 min_gb 아래면 이 작업(프로세스 그룹)만 끝낸다. 리눅스 전용."""
    if not os.path.exists("/proc/meminfo"):
        return

    def watch():
        while True:
            with open("/proc/meminfo") as f:
                avail = next(int(ln.split()[1]) for ln in f if ln.startswith("MemAvailable")) / 2 ** 20
            if avail < min_gb:
                print(f"남은 메모리 {avail:.1f}GB — 중단", flush=True)
                os.killpg(os.getpgid(0), signal.SIGKILL)
            time.sleep(2)
    threading.Thread(target=watch, daemon=True).start()


def cmd_prep(a):
    from multiprocessing import Pool
    m = items()
    if a.limit:
        m = pd.concat([m[m.source == "enes"].head(a.limit // 2), m[m.source != "enes"].head(a.limit - a.limit // 2)])
    N, V = len(m), N_AUG + 1
    A = np.zeros((N, V, 6), np.float32)
    E = np.zeros((N, V, 1024), np.float32)
    G = np.zeros((N, V), np.float32)
    O = np.zeros((N, V), np.int8)
    mem_guard()
    t0 = time.time()
    ops = a.ops.split(",")
    with Pool(a.workers, initializer=_init, initargs=(ops, 1000 if not a.tag else 101000)) as pool:
        for n, (k, Ak, Ek, Gk, Ok) in enumerate(pool.imap_unordered(_prep_one, list(enumerate(m.paths)), chunksize=2)):
            A[k], E[k], G[k], O[k] = Ak, Ek, Gk, Ok
            if (n + 1) % 100 == 0:
                print(f"  {n + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    if a.limit:
        print(f"시험 {N}개 {time.time() - t0:.1f}s (저장 안 함)\n음향 요약 원본:\n{A[:, 0].round(2)}\n"
              f"변형 비트: {O.tolist()}\n울음 판정 창 비율: {G.round(2).tolist()}")
        return
    os.makedirs(OUT, exist_ok=True)
    m.drop(columns="paths").to_csv(os.path.join(OUT, "items.csv"), index=False)
    np.savez(os.path.join(OUT, f"prep_{a.tag}.npz" if a.tag else "prep.npz"), A=A, E=E, G=G, ops=O)
    print(f"저장: 울음 {N}개 × (원본 + 변형 {N_AUG}, {'·'.join(ops)}) {time.time() - t0:.0f}s", flush=True)


# --- 2. 라벨·모델 ---------------------------------------------------------------------
def make_labels(m, A0, world, eps, seed):
    """sim_synthetic.make_labels와 같다(원본 음향 요약 기준). 0 수유 / 1 기저귀·자세 / 2 안아주기."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    km = lambda Z: KMeans(3, n_init=10, random_state=seed).fit_predict(Z)  # noqa: E731
    y = rng.integers(0, 3, len(m))
    if world == "global":
        Z = np.zeros_like(A0)
        for _, idx in m.groupby("source").indices.items():      # 출처(녹음 환경)마다 기준을 맞춘 뒤 나눈다
            Z[idx] = (A0[idx] - A0[idx].mean(0)) / (A0[idx].std(0) + 1e-9)
        rule = rng.permutation(3)[km(Z)]
    else:
        rule = np.full(len(m), -1)
        for b, idx in m.groupby("baby").indices.items():
            if not b or len(idx) < 15:
                continue
            Zb = (A0[idx] - A0[idx].mean(0)) / (A0[idx].std(0) + 1e-9)
            rule[idx] = rng.permutation(3)[km(Zb)]               # 아기마다 다른 짝
    keep = (rng.random(len(m)) >= eps) & (rule >= 0)
    y[keep] = rule[keep]
    return y


def general_model(E, y, groups, seed, vers, view):
    """7차(sim_synthetic.general_model)와 같은 그룹 분할·PCA 32·로지스틱 회귀.
    vers = 학습·검증에 쓰는 판(0 원본, 나머지 변형). 시험은 원본 또는 변형 1개(view)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.metrics import balanced_accuracy_score as BA
    rng = np.random.default_rng(seed)
    ug = rng.permutation(np.unique(groups))
    n = len(ug)
    tr_g, va_g = set(ug[: int(n * 0.6)]), set(ug[int(n * 0.6): int(n * 0.8)])
    tr = np.array([g in tr_g for g in groups])
    va = np.array([g in va_g for g in groups])
    te = ~(tr | va)
    stack = lambda mask: (E[mask][:, vers].reshape(-1, E.shape[-1]), np.repeat(y[mask], len(vers)))  # noqa: E731
    Xtr, ytr = stack(tr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    pca = PCA(32, random_state=seed).fit((Xtr - mu) / sd)
    f = lambda X: pca.transform((X - mu) / sd)  # noqa: E731
    Ztr, Zva = f(Xtr), f(E[va, 0])
    best = max((0.01, 0.1, 1.0), key=lambda C: BA(
        y[va], LR(C=C, max_iter=2000, class_weight="balanced").fit(Ztr, ytr).predict(Zva)))
    Xtv, ytv = stack(tr | va)
    clf = LR(C=best, max_iter=2000, class_weight="balanced").fit(f(Xtv), ytv)
    ti = np.where(te)[0]
    return {"clean": BA(y[ti], clf.predict(f(E[ti, 0]))), "changed": BA(y[ti], clf.predict(f(E[ti, view[ti]])))}


def r2(P, T):
    return 1 - ((P - T) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)


def spaces(m, A, E, view):
    """비교 공간들. 반환: ({이름: [N, V, d]}, 음향 요약 예측의 R²)."""
    from sklearn.linear_model import Ridge
    b = np.load(os.path.join(ART, "cry_personal_basis.npz"))
    Y16 = ((E - b["mean"]) / b["std"]) @ b["pca"]
    pub = (m.source != "enes").values
    am, asd = A[pub, 0].mean(0), A[pub, 0].std(0) + 1e-9
    xm, xsd = E[pub, 0].mean(0), E[pub, 0].std(0) + 1e-6
    X, T = (E - xm) / xsd, (A - am) / asd                     # T: 실제 음향 요약(공개 데이터 기준 표준화)
    ug = np.random.default_rng(0).permutation(np.unique(m.group.values[pub]))
    hold = pub & m.group.isin(ug[: len(ug) // 5]).values      # 공개 그룹 20%로 규제 세기를 고른다
    fit, hi = pub & ~hold, np.where(hold)[0]
    S, R2 = {}, {}
    for name, V in (("원본", 1), ("증강", N_AUG + 1)):
        Xf, Tf = X[fit, :V].reshape(-1, X.shape[-1]), np.repeat(T[fit, 0], V, axis=0)   # 목표는 늘 원본의 음향 요약

        def held(al):
            g = Ridge(alpha=al).fit(Xf, Tf)
            return g, r2(g.predict(X[hi, 0]), T[hi, 0]), r2(g.predict(X[hi, view[hi]]), T[hi, 0])
        al = max(np.logspace(0, 5, 11), key=lambda al: sum(v.mean() for v in held(al)[1:]))
        _, rc, rv = held(al)
        R2[name] = {"clean": rc, "changed": rv, "alpha": al}
        Xa, Ta = X[pub, :V].reshape(-1, X.shape[-1]), np.repeat(T[pub, 0], V, axis=0)
        S[f"음향 요약 예측({name}으로 배움)"] = Ridge(alpha=al).fit(Xa, Ta).predict(
            X.reshape(-1, X.shape[-1])).reshape(*X.shape[:2], 6)
    def spread(Z):
        """Enes 아기 안에서 원본 울음이 퍼진 정도(라벨 안 씀). 공개 데이터는 YAMNet 16차원에 극단값이 있어
        (퍼짐 6,580 vs 아기 안 수십) 기준으로 쓰면 YAMNet 쪽 비중이 0이 된다."""
        en = ~pub
        d = np.concatenate([Z[ix, 0] - Z[ix, 0].mean(0) for ix in
                            (np.where(en & (m.baby.values == b))[0] for b in np.unique(m.baby.values[en]))])
        return np.sqrt((d ** 2).sum(1).mean())
    D = S["음향 요약 예측(증강으로 배움)"]
    return {"YAMNet 16차원(파이 기본)": Y16, **S,
            "YAMNet 16 + 음향 요약 예측(증강)": np.concatenate([Y16 / spread(Y16), D / spread(D)], -1),  # 같은 비중으로
            "실제 음향 요약 6개": T}, R2


def personal(m, S, y, babies, method, vers, n_past, changed, view, seed):
    """아기마다 시간 앞 절반(부모 답을 받은 울음, 최근 n_past개)으로 뒤 절반을 맞힌다. 반환: [(아기, 정답, 예측)]."""
    rng = np.random.default_rng(seed)
    rec = []
    for b in babies:
        idx = np.where(m.baby.values == b)[0]
        idx = idx[np.lexsort((m.bout.values[idx], m.t.values[idx]))]
        h = len(idx) // 2
        past, test = idx[:h], idx[h:]
        if n_past:
            past = past[-n_past:]
        if method == "history":                                  # 이전 답 빈도만(동률은 무작위)
            pred = int(np.argmax(np.bincount(y[past], minlength=3) + rng.random(3) * 1e-3))
            rec += [(b, y[i], pred) for i in test]
            continue
        mu = S[past, 0].mean(0)                                  # 이 아기 기준점(원본 울음 평균)
        P = (S[past][:, vers] - mu).reshape(-1, S.shape[-1])
        yl = np.repeat(y[past], len(vers))
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        if method == "proto":
            C = np.array([P[yl == c].mean(0) if (yl == c).any() else np.full(P.shape[1], np.nan) for c in range(3)])
            Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        for i in test:
            q = S[i, view[i] if changed else 0] - mu
            q = q / (np.linalg.norm(q) + 1e-9)
            if method == "proto":
                pred = int(np.argmax(np.nan_to_num(Cn @ q, nan=-9.0)))
            else:                                                # 파이 모듈 knn: 이웃 3개(증강하면 원본당 5개라 15개), exp(3·유사도) 가중
                sims = Pn @ q
                top = np.argsort(-sims)[: 3 * len(vers)]
                pred = int(np.argmax(np.bincount(yl[top], weights=np.exp(3 * sims[top]), minlength=3)))
            rec.append((b, y[i], pred))
    return rec


def bal(rec):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score([r[1] for r in rec], [r[2] for r in rec])


def boot_diff(ra, rb, n=2000, seed=0):
    """아기 단위 재표집으로 bal(rb) - bal(ra)의 95% 구간(두 기록은 같은 울음 순서)."""
    babies = np.array([r[0] for r in ra])
    ya, pa, pb = (np.array([r[k] for r in rr]) for rr, k in ((ra, 1), (ra, 2), (rb, 2)))
    by = {b: np.where(babies == b)[0] for b in np.unique(babies)}
    ub = list(by)
    rng = np.random.default_rng(seed)
    from sklearn.metrics import balanced_accuracy_score as BA
    d = []
    for _ in range(n):
        s = np.concatenate([by[b] for b in rng.choice(ub, len(ub))])
        d.append(BA(ya[s], pb[s]) - BA(ya[s], pa[s]))
    return np.percentile(d, [2.5, 97.5])


# --- 3. 진단 -----------------------------------------------------------------------
def diagnostics(m, A, E, G, O, vers):
    """변형(vers 판)이 얼마나 바꾸는지: 변형 종류별로 ③ 규칙상 같은 유형으로 남는 비율, YAMNet 16차원 유사도."""
    from sklearn.cluster import KMeans
    b = np.load(os.path.join(ART, "cry_personal_basis.npz"))
    Y16 = ((E - b["mean"]) / b["std"]) @ b["pca"]
    keep, sim, other, sim_ac, other_ac, ops = [], [], [], [], [], []
    for bb, idx in m.groupby("baby").indices.items():
        if not bb or m.source.values[idx[0]] != "enes":
            continue
        A0 = A[idx, 0]
        mu, sd = A0.mean(0), A0.std(0) + 1e-9
        Zc = (A[idx] - mu) / sd
        Yc = Y16[idx] - Y16[idx, 0].mean(0)
        nrm = lambda Z: Z / (np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-9)  # noqa: E731
        Yn, Zn = nrm(Yc), nrm(Zc)
        cy, cz = Yn[:, 0] @ Yn[:, 0].T, Zn[:, 0] @ Zn[:, 0].T
        off = ~np.eye(len(idx), dtype=bool)
        other.append(cy[off].mean())
        other_ac.append(cz[off].mean())
        km = KMeans(3, n_init=10, random_state=0).fit(Zc[:, 0]) if len(idx) >= 15 else None
        for v in vers:
            sim += list((Yn[:, 0] * Yn[:, v]).sum(1))
            sim_ac += list((Zn[:, 0] * Zn[:, v]).sum(1))
            keep += list(km.predict(Zc[:, v]) == km.labels_) if km is not None else [np.nan] * len(idx)
            ops += list(O[idx, v])
    keep, sim, sim_ac, ops = np.array(keep, float), np.array(sim), np.array(sim_ac), np.array(ops)
    rows = []
    for i, name in enumerate(OPS_KO):
        on = (ops & (1 << i)) > 0
        if not on.any():
            continue
        rows.append({"op": name, "freq": on.mean(), "keep_on": np.nanmean(keep[on]), "keep_off": np.nanmean(keep[~on]),
                     "sim_on": sim[on].mean(), "sim_off": sim[~on].mean()})
    en = (m.source == "enes").values
    return {"ops": rows, "keep": np.nanmean(keep), "sim": sim.mean(), "other": float(np.mean(other)),
            "sim_ac": sim_ac.mean(), "other_ac": float(np.mean(other_ac)),
            "gate": {"enes": (G[en, 0].mean(), G[en][:, vers].mean()), "pub": (G[~en, 0].mean(), G[~en][:, vers].mean())}}


# --- 4. 실행 -----------------------------------------------------------------------
def cmd_run(a):
    import warnings
    warnings.filterwarnings("ignore")
    m = pd.read_csv(os.path.join(OUT, "items.csv"))
    m["baby"], m["stop"] = m.baby.fillna("").astype(str), m.stop.fillna("").astype(str)
    z = np.load(os.path.join(OUT, "prep.npz"))
    A, E, G, O = (z[k] for k in ("A", "E", "G", "ops"))
    VAR = {"방 조건": list(range(1, N_AUG + 1))}                    # 판 번호: 0 원본, 1~4 방 조건 변형
    mild = os.path.join(OUT, "prep_mild.npz")
    if os.path.exists(mild):                                     # 5~8 울음 변형(속도·음높이만)
        z2 = np.load(mild)
        assert np.abs(z2["E"][:, 0] - E[:, 0]).max() < 1e-3, "원본 순서가 다르다"
        A, E, G, O = (np.concatenate([x, z2[k][:, 1:]], 1) for x, k in ((A, "A"), (E, "E"), (G, "G"), (O, "ops")))
        VAR["울음 변형"] = list(range(N_AUG + 1, 2 * N_AUG + 1))
    A, E, O = A.astype(np.float64), E.astype(np.float64), O.astype(int)
    MODES = {"없음": [0], **{k: [0] + v for k, v in VAR.items()}}
    N = len(m)
    view = np.random.default_rng(99).integers(1, N_AUG + 1, N)     # '조건이 바뀐' 시험에 쓰는 변형(설정 간 공통)
    enes = (m.source == "enes").values
    t0 = time.time()
    S, R2 = spaces(m, A, E, view)
    diag = {k: diagnostics(m, A, E, G, O, v) for k, v in VAR.items()}
    print(f"  공간·진단 {time.time() - t0:.0f}s", flush=True)

    gen = []
    for world, eps in (("global", 0.0), ("global", 0.3), ("baby", 0.0), ("baby", 0.3)):
        for seed in range(a.seeds):
            y = make_labels(m, A[:, 0], world, eps, seed)
            sel = np.ones(N, bool) if world == "global" else enes     # 7차와 같게: 아기별 규칙은 Enes에서
            for mode, vers in MODES.items():
                r = general_model(E[sel], y[sel], m.group.values[sel], seed, vers, view[sel])
                gen.append({"world": world, "eps": eps, "seed": seed, "aug": mode, **r})
    print(f"  일반 모델 {time.time() - t0:.0f}s", flush=True)

    all_b = sorted(m.baby[enes].unique())
    rule_b = sorted(b for b, n in m.baby[enes].value_counts().items() if n >= 15)
    cfgs = [(s, me, ag) for s in S for me in ("proto", "knn") for ag in MODES] + [("이력만(답 빈도)", "history", "없음")]
    pers, recs = [], {}

    def run_cfgs(label, y, babies, seed, n_pasts=(5, 10, 0), views=(False, True)):
        for s, me, ag in cfgs:
            for npast in n_pasts:
                for ch in views:
                    if me == "history" and ch:
                        continue
                    rec = personal(m, S.get(s), y, babies, me, MODES[ag], npast, ch, view, seed)
                    pers.append({"label": label, "seed": seed, "space": s, "method": me, "aug": ag, "n_past": npast,
                                 "changed": ch, "bal": bal(rec), "n": len(rec)})
                    if label == "real" and npast == 0 and not ch:
                        recs[(s, me, ag)] = rec

    for world, eps in (("global", 0.0), ("baby", 0.0), ("baby", 0.3)):
        for seed in range(a.seeds):
            run_cfgs(f"{world}:{eps}", make_labels(m, A[:, 0], world, eps, seed), all_b if world == "global" else rule_b, seed)
    print(f"  개인화(가상) {time.time() - t0:.0f}s", flush=True)
    real = np.array(["FIS".index(s) if s in ("F", "I", "S") else 0 for s in m.stop])
    run_cfgs("real", real, all_b, 0)
    rng = np.random.default_rng(7)
    for r_ in range(a.perms):
        yp = real.copy()
        for b in all_b:
            idx = np.where(m.baby.values == b)[0]
            yp[idx] = rng.permutation(yp[idx])
        run_cfgs("perm", yp, all_b, r_, n_pasts=(0,), views=(False,))
    print(f"  개인화(실제·귀무) {time.time() - t0:.0f}s", flush=True)
    ci = {k: boot_diff(recs[(k[0], k[1], "없음")], recs[k]) for k in recs if k[2] != "없음" and k[1] != "history"}

    os.makedirs(RUNS, exist_ok=True)
    GN, PS = pd.DataFrame(gen), pd.DataFrame(pers)
    GN.to_csv(os.path.join(RUNS, "augment_general.csv"), index=False)
    PS.to_csv(os.path.join(RUNS, "augment_personal.csv"), index=False)
    report(m, R2, diag, GN, PS, ci, len(rule_b), list(MODES))


def report(m, R2, diag, GN, PS, ci, n_rule, modes):
    en = (m.source == "enes").values
    slash = " / ".join(modes)
    L = ["# 8차: 증강 + 비교 공간 바꾸기 (가상 행동 라벨 시뮬레이션 재실행)", "",
         f"- 울음 {len(m)}개(Enes {en.sum()} + 공개 {(~en).sum()})의 앞 {CLIP_S}초. 원본마다 변형 {N_AUG}개씩, "
         "변형은 원래 울음의 라벨을 물려받는다.",
         "  - 방 조건 증강: 잡음 / 잔향 / 거리감 / 속도 / 음높이를 각 50% 확률로 조합(augment.py와 같은 세기)",
         "  - 울음 변형 증강: 속도(0.9~1.1배) / 음높이(±1반음)만 각 50% — 같은 방에서 울음만 조금 다른 경우",
         "- 시험 울음의 변형은 학습에 쓰지 않는다. '조건 바뀐 시험' = 시험 울음 대신 그 울음의 방 조건 변형 1개를 들려준다.",
         "- 균형 정확도, 무작위 0.333. 가상 라벨은 시드 5개 평균.", "",
         "## 0. 변형이 소리를 얼마나 바꾸나 (Enes)", ""]
    for k, d in diag.items():
        g = d["gate"]
        L += [f"**{k} 증강** — 울음 판정 창 비율 원본 → 변형: Enes {g['enes'][0]:.2f} → {g['enes'][1]:.2f}, "
              f"공개 {g['pub'][0]:.2f} → {g['pub'][1]:.2f}. 코사인 유사도(아기 기준점에서) 원본 ↔ 자기 변형 {d['sim']:.2f} "
              f"vs 같은 아기의 다른 울음끼리 {d['other']:.2f} (YAMNet 16차원), 음향 요약 6개에서는 {d['sim_ac']:.2f} vs "
              f"{d['other_ac']:.2f}. ③ 규칙상 원본과 같은 유형으로 남는 변형 {d['keep']:.0%}.", "",
              "| 변형 | 들어간 비율 | 같은 유형 유지(들어감 / 안 들어감) | YAMNet 원본↔변형 유사도(들어감 / 안 들어감) |",
              "|---|---:|---:|---:|"]
        for r in d["ops"]:
            L.append(f"| {r['op']} | {r['freq']:.0%} | {r['keep_on']:.0%} / {r['keep_off']:.0%} | "
                     f"{r['sim_on']:.2f} / {r['sim_off']:.2f} |")
        L.append("")
    L += ["YAMNet 임베딩 → 음향 요약 선형 변환의 설명력 R² (공개 데이터 중 처음 보는 그룹 20%, 증강 = 방 조건 증강):", "",
          "| 학습 | 입력 | " + " | ".join(ACOUSTIC_KO) + " |", "|---|---|" + "---:|" * 6]
    for name, r in R2.items():
        for k, lab in (("clean", "원본"), ("changed", "조건 바뀜")):
            L.append(f"| {name}으로 배움 | {lab} | " + " | ".join(f"{v:.2f}" for v in r[k]) + " |")
    L += ["", "## 1. 일반 모델 (그룹 단위 학습 60 / 검증 20 / 시험 20)", "",
          f"| 가상 세계 | 라벨 잡음 | 원본 시험 ({slash}) | 조건 바뀐 시험 ({slash}) |", "|---|---|---:|---:|"]
    wname = {"global": "② 공통 규칙", "baby": "③ 아기마다 다른 규칙"}
    for (w, e), gg in GN.groupby(["world", "eps"], sort=False):
        v = gg.groupby("aug")[["clean", "changed"]].mean()
        L.append(f"| {wname[w]} | {e:.0%} | " + " / ".join(f"{v.clean[k]:.3f}" for k in modes) + " | "
                 + " / ".join(f"{v.changed[k]:.3f}" for k in modes) + " |")
    lname = {"global:0.0": "② 공통 규칙, 라벨 잡음 0%", "baby:0.0": f"③ 아기마다 다른 규칙, 라벨 잡음 0% (규칙이 있는 아기 {n_rule}명)",
             "baby:0.3": f"③ 아기마다 다른 규칙, 라벨 잡음 30% (아기 {n_rule}명)", "real": "실제 Enes 라벨 (아기 24명)"}
    L += ["", "## 2. 개인화: 아기마다 앞 절반의 답으로 뒤 절반 맞히기", "",
          f"칸 = 증강 {slash}. '답 5개' = 앞 절반 중 최근 5개만 쓴 경우."]
    nul = PS[PS.label == "perm"].groupby(["space", "method", "aug"]).bal.quantile(0.95)
    for lab in ("global:0.0", "baby:0.0", "baby:0.3", "real"):
        d = PS[PS.label == lab]
        cols = ["답 5개", "답 10개", "앞 절반 전부", "조건 바뀐 시험(앞 절반 전부)"]
        extra = " | 귀무 95% | 증강 효과 95% 구간 |" if lab == "real" else " |"
        L += ["", f"### {lname[lab]}", "", "| 비교 공간 | 방법 | " + " | ".join(cols) + extra,
              "|---|---|" + "---:|" * (len(cols) + (2 if lab == "real" else 0))]
        for (s, me), gg in d.groupby(["space", "method"], sort=False):
            v = gg.groupby(["aug", "n_past", "changed"]).bal.mean()
            cell = lambda npast, ch: (f"{v[('없음', npast, ch)]:.3f}" if me == "history" else  # noqa: E731
                                      " / ".join(f"{v[(k, npast, ch)]:.3f}" for k in modes))
            cells = [cell(5, False), cell(10, False), cell(0, False), "-" if me == "history" else cell(0, True)]
            row = f"| {s} | {'-' if me == 'history' else me} | " + " | ".join(cells)
            if lab == "real":
                nn = (f"{nul[(s, me, '없음')]:.3f}" if me == "history" else
                      " / ".join(f"{nul[(s, me, k)]:.3f}" for k in modes))
                cc = [f"{k} {ci[(s, me, k)][0]:+.3f}~{ci[(s, me, k)][1]:+.3f}" for k in modes[1:] if (s, me, k) in ci]
                row += f" | {nn} | " + ("; ".join(cc) or "-")
            L.append(row + " |")
        if lab == "real":
            L += ["", "귀무 = 아기 안에서 라벨을 섞어 같은 절차(앞 절반 전부, 원본 시험)를 20번. 섞으면 시간에 따른 답 비율 변화가 "
                  "사라져서 '이력만'은 귀무가 실제보다 높다. 증강 효과 구간 = 아기 단위 재표집 2,000번(그 증강 − 증강 없음)."]
    txt = "\n".join(L) + "\n"
    open(os.path.join(RUNS, "AUGMENT_REPORT.md"), "w", encoding="utf-8").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="앞쪽 몇 개만 시험(저장 안 함)")
    p.add_argument("--ops", default=",".join(OPS), help="쓸 변형(쉼표). 약한 증강: stretch,pitch")
    p.add_argument("--tag", default="", help="저장 이름 prep_<tag>.npz (run은 prep_mild.npz가 있으면 같이 쓴다)")
    p = sp.add_parser("run")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--perms", type=int, default=20)
    a = ap.parse_args()
    {"prep": cmd_prep, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
