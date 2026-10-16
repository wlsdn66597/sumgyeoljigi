"""행동 개수(3·4·6개)와 MFCC 블록을 파이 모듈(../cry_personal.py) 그대로 재생해 비교한다(10차).

    python sim_kclass.py feats --workers 12   # 울음 2,521개: 파이식 MFCC(울음 판정 창만) + 스펙트럼 모양 + (Enes) 리듬
    python sim_kclass.py export               # artifacts/cry_personal_acoustic.npz에 MFCC 멜 필터·표준화·블록 퍼짐을 더한다
    python sim_kclass.py run --workers 12     # → DATA/runs/synthetic/KCLASS_REPORT.md

정답 세계: ③ 아기마다 다른 규칙을 특징 6묶음으로 만든다(아기 안 표준화 → k-평균 → 유형 크기를 정한 비율로 배정).
  요약 6 / eGeMAPS / MFCC / YAMNet / 스펙트럼 모양(대비·평탄도·롤오프·대역폭·영교차) / 리듬(음절 속도·쉼·유성 비율·길이)
  스펙트럼 모양과 리듬은 어떤 비교 공간도 직접 쓰지 않아 가장 공정하다. 비교 공간과 같은 특징으로 만든 세계는 순환이라 뺀다.
행동 개수: 3·4·6개(유형 크기 균등)와 6개 실사용 비율(가정: 수유 35 / 기저귀 20 / 재우기 15 / 안아주기 15 / 트림 10 / 온도 5%).
평가: 아기마다 시간순 재생(힌트를 먼저 내고 답을 받음), 답 10개 이상 쌓인 뒤 1순위 균형 정확도와 2순위 안 적중률(행동별 평균).
"""
import argparse
import os
import sys
import time

if len(sys.argv) > 1 and sys.argv[1] in ("feats", "run"):
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]                   # 서버(같은 폴더) / 로컬(저장소 루트)의 cry_personal
import cry_personal as cp  # noqa: E402
import sim_augment as sa  # noqa: E402
import sim_space as ss  # noqa: E402

FEATS2 = os.path.join(sa.OUT, "feats_k.npz")
ACOUSTIC_NPZ = os.path.join(sa.ART, "cry_personal_acoustic.npz")
REAL6 = [0.35, 0.20, 0.15, 0.15, 0.10, 0.05]
GRID = [("3개 균등", 3, None), ("4개 균등", 4, None), ("6개 균등", 6, None), ("6개 실사용 비율", 6, REAL6)]
CFG = {"이력만": dict(w_sound=0.0, blocks=["acoustic"]),
       "YAMNet 16": dict(blocks=["yamnet16"], gate=False),
       "음향 변환": dict(blocks=["acoustic"], gate=False),
       "MFCC만": dict(blocks=["mfcc"], gate=False),
       "음향 변환 + MFCC": dict(blocks=["acoustic", "mfcc"], gate=False),
       "YAMNet 16 + MFCC": dict(blocks=["yamnet16", "mfcc"], gate=False),
       "음향 변환 + MFCC, 자가 검증": dict(blocks=["acoustic", "mfcc"], gate=True)}
FAMILY = {"yamnet16": {"YAMNet"}, "acoustic": {"요약 6", "eGeMAPS", "MFCC"}, "mfcc": {"MFCC"}}
_W = {}


# --- 1. 특징 -----------------------------------------------------------------------
def _init_feats():
    import librosa
    _W["yam"], _W["fb"] = sa.Yam(), librosa.filters.mel(sr=sa.SR, n_fft=cp.N_FFT)


def spectral_shape(x):
    """스펙트럼 모양 22개: 대비(7대역) 평균·표준편차, 평탄도·롤오프·대역폭·영교차 평균·표준편차(소리 나는 프레임)."""
    import librosa
    kw = dict(n_fft=1024, hop_length=256)
    con = librosa.feature.spectral_contrast(y=x, sr=sa.SR, **kw)
    flat = librosa.feature.spectral_flatness(y=x, **kw)
    roll = np.log(librosa.feature.spectral_rolloff(y=x, sr=sa.SR, **kw) + 1)
    bw = np.log(librosa.feature.spectral_bandwidth(y=x, sr=sa.SR, **kw) + 1)
    zcr = librosa.feature.zero_crossing_rate(x, frame_length=1024, hop_length=256)
    r = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
    F = np.vstack([con, np.log(flat + 1e-6), roll, bw, zcr])[:, : len(r)]
    db = 20 * np.log10(r[: F.shape[1]] + 1e-9)
    act = db > db.max() - 30
    F = F[:, act] if act.sum() >= 3 else F
    return np.r_[F.mean(1), F.std(1)]


def _feat_one(job):
    """파이처럼 1초 창마다 YAMNet 울음 점수로 거르고(0.3 이상, 없으면 전부), 그 창들의 MFCC 프레임으로 통계를 낸다."""
    k, paths = job
    x = sa.load_clip(paths)
    xw = np.pad(x, (0, max(0, sa.WIN - len(x))))
    wins = [xw[s:s + sa.WIN] for s in range(0, len(xw) - sa.WIN + 1, sa.HOP)]
    cry = []
    for w in wins:
        _W["yam"].it.set_tensor(_W["yam"].inp, w.astype(np.float32))
        _W["yam"].it.invoke()
        sc = _W["yam"].it.get_tensor(_W["yam"].out[521]).reshape(-1)
        cry.append(max(sc[19], sc[20]) >= sa.CRY_THR)
    use = [w for w, c in zip(wins, cry) if c] or wins
    fr = [cp.mfcc_frames(w, _W["fb"]) for w in use]
    ms = cp.mfcc_stats(np.concatenate([f[0] for f in fr]), np.concatenate([f[1] for f in fr]))
    xs = np.pad(x, (0, max(0, sa.SR - len(x))))
    return k, ms.astype(np.float32), spectral_shape(xs).astype(np.float32)


def rhythm(m):
    """Enes 울음(조각 묶음)마다 리듬 5개: 음절 속도, 음절 사이 쉼, 유성 비율, 울음 길이(로그), 조각 수. 공개 데이터는 NaN."""
    e = pd.read_csv(os.path.join(sa.DATA, "enes3", "meta.csv"), dtype={"date": str, "time": str})
    R = np.full((len(m), 5), np.nan)
    by = {b: g for b, g in e.groupby("bout")}
    for i in np.where(m.source.values == "enes")[0]:
        g = by[m.bout.values[i]]
        d = g.dur.sum()
        R[i] = [g.n_syl.sum() / d, np.average(g.gap_med.fillna(g.gap_med.median()), weights=g.dur), g.voiced.sum() / d,
                np.log(d), len(g)]
    return R


def cmd_feats(a):
    from multiprocessing import Pool
    m = sa.items()
    MS = np.zeros((len(m), 40), np.float32)
    SP = np.zeros((len(m), 22), np.float32)
    t0 = time.time()
    with Pool(a.workers, initializer=_init_feats) as pool:
        for n, (k, ms, sp) in enumerate(pool.imap_unordered(_feat_one, list(enumerate(m.paths)), chunksize=4)):
            MS[k], SP[k] = ms, sp
            if (n + 1) % 500 == 0:
                print(f"  {n + 1}/{len(m)} {time.time() - t0:.0f}s", flush=True)
    m2 = pd.read_csv(os.path.join(sa.OUT, "items.csv"))
    np.savez(FEATS2, mfcc_dev=MS, spec=np.nan_to_num(SP), rhythm=rhythm(m2))
    print(f"저장 {FEATS2} {time.time() - t0:.0f}s", flush=True)


# --- 2. 파이 배포 파일 --------------------------------------------------------------
def enes_order(m, min_n=15):
    en = (m.source == "enes").values
    order = {}
    for b in sorted(bb for bb, n in m.baby[en].value_counts().items() if n >= min_n):
        ix = np.where(m.baby.values == b)[0]
        order[b] = ix[np.lexsort((m.bout.values[ix], m.t.values[ix]))]
    return order


def cmd_export(_):
    """MFCC 멜 필터(librosa와 같은 값)·공개 데이터 기준 표준화, 그리고 블록마다 'Enes 아기 안 퍼짐'(라벨 안 씀)을 더한다."""
    import librosa
    m, A, E, G, O, VAR, MODES, view = sa.load_all()
    f = np.load(FEATS2)
    z = dict(np.load(ACOUSTIC_NPZ))
    pub = (m.source != "enes").values
    MS = f["mfcc_dev"].astype(float)
    z["mel_fb"] = librosa.filters.mel(sr=sa.SR, n_fft=cp.N_FFT).astype(np.float32)
    z["mfcc_mean"], z["mfcc_std"] = MS[pub].mean(0).astype(np.float32), (MS[pub].std(0) + 1e-6).astype(np.float32)
    for k in ("spread_yamnet16", "spread_acoustic", "spread_mfcc"):
        z.pop(k, None)
    b = np.load(os.path.join(sa.ART, "cry_personal_basis.npz"))
    dev = cp.CryPersonal({k: b[k] for k in ("mean", "std", "pca")}, None, acoustic_path=z,
                         blocks=["yamnet16", "acoustic", "mfcc"])
    order = enes_order(m, min_n=2)
    for blk in ("yamnet16", "acoustic", "mfcc"):
        dev.blocks, dev.w[blk] = [blk], 1.0
        Z = np.array([dev._project(E[i, 0], MS[i]) for i in range(len(m))])
        d = np.concatenate([Z[ix] - Z[ix].mean(0) for ix in order.values()])
        z[f"spread_{blk}"] = np.float32(np.sqrt((d ** 2).sum(1).mean()))
    np.savez(ACOUSTIC_NPZ, **z)
    print("저장", ACOUSTIC_NPZ, {k: round(float(z[k]), 3) for k in z if k.startswith("spread_")})


# --- 3. 재생 ------------------------------------------------------------------------
def baby_rule(m, F, order, K, freqs, seed):
    """아기 안에서 표준화(차원이 많으면 PCA 10) → k-평균 K유형 → 유형 크기를 freqs 비율로 배정.
    균등이면 유형 → 행동 짝을 아기마다 무작위로, 실사용 비율이면 비율 순서가 곧 행동(수유가 가장 흔함)이다."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    fr = np.ones(K) / K if freqs is None else np.asarray(freqs, float)
    y = np.zeros(len(m), int)
    for ix in order.values():
        Z = (F[ix] - F[ix].mean(0)) / (F[ix].std(0) + 1e-9)
        if Z.shape[1] > 10:
            Z = PCA(10, random_state=seed).fit_transform(Z)
        C = KMeans(K, n_init=10, random_state=seed).fit(Z).cluster_centers_
        slot = rng.permutation(K)                               # 유형 j → 비율 칸 slot[j](= 행동 번호)
        raw = fr * len(ix)
        cap = np.floor(raw).astype(int)
        cap[np.argsort(-(raw - cap))[: len(ix) - cap.sum()]] += 1
        cols = np.concatenate([[j] * cap[slot[j]] for j in range(K)])
        D = ((Z[:, None] - C[None]) ** 2).sum(-1)[:, cols]
        r, c = linear_sum_assignment(D)
        y[ix[r]] = slot[cols[c]]
    return y


def _init_run():
    m, A, E, G, O, VAR, MODES, view = sa.load_all()
    f, f2 = np.load(ss.FEATS), np.load(FEATS2)
    b = np.load(os.path.join(sa.ART, "cry_personal_basis.npz"))
    _W.update(m=m, E0=E[:, 0].copy(), MS=f2["mfcc_dev"].astype(float), basis={k: b[k] for k in ("mean", "std", "pca")},
              ac=dict(np.load(ACOUSTIC_NPZ)), order=enes_order(m),
              worlds={"요약 6": A[:, 0].copy(), "eGeMAPS": f["egemaps"].astype(float), "MFCC": f2["mfcc_dev"].astype(float),
                      "YAMNet": E[:, 0].copy(), "스펙트럼 모양": f2["spec"].astype(float), "리듬": f2["rhythm"]})
    del E


def replay(y, actions, order):
    """설정마다 모듈을 아기별로 새로 만들어 시간순 재생. 반환 {설정: 지표}."""
    m, E0, MS = _W["m"], _W["E0"], _W["MS"]
    out = {}
    for name, kw in CFG.items():
        rec = []
        for ix in order.values():
            dev = cp.CryPersonal(_W["basis"], None, acoustic_path=_W["ac"], actions=actions, hint_seconds=10, **kw)
            for i in ix:
                t = float(m.t.values[i])
                h = dev.hint(t, embs=[E0[i]], mfcc=MS[i])
                if h["n_feedback"] >= 10:
                    rec.append((y[i], actions.index(h["top"]), actions[y[i]] in h["top2"], h["n_feedback"], h["sound_on"]))
                dev.start(t)
                dev.add_window(E0[i])
                dev.feedback(dev.end(t + 60, mfcc=(MS[i], MS[i])), actions[y[i]])
        R = np.array(rec, float)
        yt, yp, hit, nf, on = R.T
        yt, yp = yt.astype(int), yp.astype(int)

        def top2(sel):
            return float(np.mean([hit[sel & (yt == c)].mean() for c in np.unique(yt[sel])]))
        out[name] = {"top1": sa.bal_np(yt, yp), "top2": top2(nf >= 10), "top1_20": sa.bal_np(yt[nf >= 20], yp[nf >= 20]),
                     "top2_20": top2(nf >= 20), "on": float(on.mean())}
    return out


def _task(job):
    kind, world, gname, K, freqs, seed = job
    m, order = _W["m"], _W["order"]
    if kind == "sim":
        y = baby_rule(m, _W["worlds"][world], order, K, freqs, seed)
        actions = cp.ACTIONS6[:K] if K == 6 else [f"a{k}" for k in range(K)]
        res = replay(y, actions, order)
    else:                                                      # 실제 Enes 라벨(3개), seed > 0이면 아기 안 섞기 귀무
        order = enes_order(m, min_n=2)
        y = np.array(["FIS".index(s) if s in ("F", "I", "S") else 0 for s in m.stop.astype(str)])
        if seed:
            rng = np.random.default_rng(seed)
            for ix in order.values():
                y[ix] = rng.permutation(y[ix])
        res = replay(y, cp.ACTIONS3, order)
    return [{"kind": kind, "world": world, "grid": gname, "K": K, "seed": seed, "cfg": c, **v} for c, v in res.items()]


def cmd_run(a):
    import warnings
    from multiprocessing import Pool
    warnings.filterwarnings("ignore")
    jobs = [("sim", w, g, K, fr, s) for w in ("요약 6", "eGeMAPS", "MFCC", "YAMNet", "스펙트럼 모양", "리듬")
            for g, K, fr in GRID for s in range(a.seeds)]
    jobs += [("real", "실제 Enes", "3개(실제)", 3, None, s) for s in range(a.perms + 1)]
    rows, t0 = [], time.time()
    with Pool(a.workers, initializer=_init_run) as pool:
        for n, r in enumerate(pool.imap_unordered(_task, jobs)):
            rows += r
            if (n + 1) % 20 == 0:
                print(f"  {n + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(sa.RUNS, "kclass.csv"), index=False)
    report(D)


def report(D):
    S = D[D.kind == "sim"]
    worlds = list(dict.fromkeys(S.world))
    fair_of = {c: [w for w in worlds if w not in set().union(*[FAMILY[b] for b in kw.get("blocks", [])])]
               if c != "이력만" else worlds for c, kw in CFG.items()}
    L = ["# 10차: 행동 개수와 MFCC (파이 모듈 그대로 재생)", "",
         "- 가상 ③ 아기마다 다른 규칙, Enes 아기 22명, 답 10개 이상 쌓인 뒤. 시드 3개 평균.",
         "- 1순위 = 균형 정확도(무작위 1/K), 2순위 안 = 정답이 1·2순위에 든 비율(행동별 평균, 무작위 2/K).",
         "- 공정 평균 = 그 비교 공간과 같은 특징으로 만든 세계를 뺀 평균. 스펙트럼 모양·리듬 세계는 모든 공간에 공정하다.",
         "- 맥락 모델은 쓰지 않았다(이력 × 소리).", "",
         "## 1. 행동 개수별 (공정 평균: 1순위 / 2순위 안)", "",
         "| 비교 공간 | " + " | ".join(g for g, _, _ in GRID) + " |", "|---|" + "---:|" * len(GRID),
         "| 무작위 | " + " | ".join(f"{1 / K:.2f} / {2 / K:.2f}" for _, K, _ in GRID) + " |"]
    for c in CFG:
        cells = []
        for g, _, _ in GRID:
            x = S[(S.cfg == c) & (S.grid == g) & S.world.isin(fair_of[c])]
            cells.append(f"{x.top1.mean():.2f} / {x.top2.mean():.2f}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## 2. 가장 공정한 두 세계(스펙트럼 모양 · 리듬)만: 1순위 / 2순위 안", "",
          "| 비교 공간 | " + " | ".join(g for g, _, _ in GRID) + " |", "|---|" + "---:|" * len(GRID)]
    for c in CFG:
        cells = []
        for g, _, _ in GRID:
            x = S[(S.cfg == c) & (S.grid == g) & S.world.isin(["스펙트럼 모양", "리듬"])]
            cells.append(f"{x.top1.mean():.2f} / {x.top2.mean():.2f}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## 3. 세계별 1순위 (6개 실사용 비율). **굵게** = 순환", "",
          "| 비교 공간 | " + " | ".join(worlds) + " |", "|---|" + "---:|" * len(worlds)]
    for c in CFG:
        x = S[(S.cfg == c) & (S.grid == "6개 실사용 비율")].groupby("world").top1.mean()
        L.append(f"| {c} | " + " | ".join(f"**{x[w]:.2f}**" if w not in fair_of[c] else f"{x[w]:.2f}" for w in worlds) + " |")
    L += ["", "## 4. 답이 더 쌓이면 (공정 평균, 답 20개 이상: 1순위 / 2순위 안, 소리 켜진 비율은 자가 검증만)", "",
          "| 비교 공간 | " + " | ".join(g for g, _, _ in GRID) + " |", "|---|" + "---:|" * len(GRID)]
    for c in CFG:
        cells = []
        for g, _, _ in GRID:
            x = S[(S.cfg == c) & (S.grid == g) & S.world.isin(fair_of[c])]
            on = f" ({x.on.mean():.0%})" if "자가" in c else ""
            cells.append(f"{x.top1_20.mean():.2f} / {x.top2_20.mean():.2f}{on}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    R = D[D.kind == "real"]
    L += ["", "## 5. 실제 Enes 라벨 (행동 3개, 아기 24명): 1순위 / 2순위 안 (귀무 = 아기 안 라벨 섞기 20번의 95%)", "",
          "| 비교 공간 | 실제 | 귀무 95% | 소리 켜진 비율 |", "|---|---:|---:|---:|"]
    for c in CFG:
        x, n = R[(R.cfg == c) & (R.seed == 0)].iloc[0], R[(R.cfg == c) & (R.seed > 0)]
        L.append(f"| {c} | {x.top1:.3f} / {x.top2:.3f} | {n.top1.quantile(.95):.3f} / {n.top2.quantile(.95):.3f} | {x.on:.0%} |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(sa.RUNS, "KCLASS_REPORT.md"), "w", encoding="utf-8").write(txt)
    print(txt)


# --- 4. 긴 사용 기간: 아기 여럿을 이어 붙인 가상 아기 ----------------------------------------------
LONG_CFG = ["이력만", "YAMNet 16", "음향 변환 + MFCC", "음향 변환 + MFCC, 자가 검증"]
LONG_BUCKETS = [(10, 19), (20, 39), (40, 59), (60, 79), (80, 999)]
LONG_BUCKETS_BIG = [(10, 19), (20, 39), (40, 79), (80, 119), (120, 159), (160, 199), (200, 999)]


def _long_task(job):
    """Enes 아기 gsize명씩 묶어 가상 아기를 만든다(4명이면 울음 약 100개, 11명이면 약 280개). 아기마다 자기 평균을 빼(목소리 차이 제거) 같은
    아기 안의 차이만 남기고, 각자의 시간 순서를 지키며 번갈아 넣는다. 반환: 울음마다 (설정, 정답, 1순위, 2순위 적중, 답 수)."""
    world, K, seed, gsize = job
    m, E0, MS = _W["m"], _W["E0"], _W["MS"]
    order = _W["order"]
    rng = np.random.default_rng(seed)
    babies = list(order)
    rng.shuffle(babies)
    groups = [babies[i:i + gsize] for i in range(0, len(babies) - len(babies) % gsize, gsize)]
    En, Mn, Fn = E0.copy(), MS.copy(), _W["worlds"][world].copy()
    gE, gM = E0[np.concatenate(list(order.values()))].mean(0), MS[np.concatenate(list(order.values()))].mean(0)
    for ix in order.values():                                 # 아기마다 평균을 빼고 전체 평균을 더한다
        En[ix] += gE - E0[ix].mean(0)
        Mn[ix] += gM - MS[ix].mean(0)
        Fn[ix] = (Fn[ix] - Fn[ix].mean(0)) / (Fn[ix].std(0) + 1e-9)
    vorder = {}
    for gi, grp in enumerate(groups):
        pos = np.concatenate([np.arange(len(order[b])) / len(order[b]) for b in grp])
        ix = np.concatenate([order[b] for b in grp])
        vorder[f"v{gi}"] = ix[np.argsort(pos + rng.random(len(pos)) * 1e-6)]
    y = baby_rule(m, Fn, vorder, K, None, seed)
    actions = cp.ACTIONS3 if K == 3 else [f"a{k}" for k in range(K)]
    rec = []
    for name in LONG_CFG:
        kw = CFG[name]
        for ix in vorder.values():
            dev = cp.CryPersonal(_W["basis"], None, acoustic_path=_W["ac"], actions=actions, hint_seconds=10, **kw)
            for n_, i in enumerate(ix):
                t = 1.7e9 + n_ * 3600.0
                h = dev.hint(t, embs=[En[i]], mfcc=Mn[i])
                if h["n_feedback"] >= 10:
                    rec.append((world, K, seed, name, y[i], actions.index(h["top"]), actions[y[i]] in h["top2"],
                                h["n_feedback"], h["sound_on"]))
                dev.start(t)
                dev.add_window(En[i])
                dev.feedback(dev.end(t + 60, mfcc=(Mn[i], Mn[i])), actions[y[i]])
    return rec


def cmd_long(a):
    import warnings
    from multiprocessing import Pool
    warnings.filterwarnings("ignore")
    jobs = [(w, K, s, a.group) for w in ("요약 6", "eGeMAPS", "MFCC", "YAMNet", "스펙트럼 모양", "리듬") for K in (3, 4)
            for s in range(a.seeds)]
    buckets = LONG_BUCKETS if a.group <= 4 else LONG_BUCKETS_BIG
    tag = "" if a.group == 4 else f"_g{a.group}"
    rows, t0 = [], time.time()
    with Pool(a.workers, initializer=_init_run) as pool:
        for n, r in enumerate(pool.imap_unordered(_long_task, jobs)):
            rows += r
            if (n + 1) % 12 == 0:
                print(f"  {n + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows, columns=["world", "K", "seed", "cfg", "y", "pred", "hit2", "n", "on"])
    D.to_csv(os.path.join(sa.RUNS, f"kclass_long{tag}.csv"), index=False)
    nb = 22 // a.group
    fair = {c: [w for w in dict.fromkeys(D.world) if w not in set().union(*[FAMILY[b] for b in CFG[c].get("blocks", [])])]
            if c != "이력만" else list(dict.fromkeys(D.world)) for c in LONG_CFG}
    lab = lambda lo, hi: f"{lo}~{hi}" if hi < 999 else f"{lo}+"  # noqa: E731
    L = [f"# 긴 사용 기간: 아기 {a.group}명을 이어 붙인 가상 아기(울음 약 {25 * a.group}개)로 답 수별 정확도", "",
         "- 아기마다 자기 평균을 빼 목소리 차이를 지우고, 각자의 시간 순서를 지키며 번갈아 넣었다(가정: 아기들의 울음 변화 폭이 비슷).",
         f"- 가상 아기 {nb}명(울음 15개 이상인 Enes 아기 22명 중 {nb * a.group}명을 시드마다 다르게 {a.group}명씩) × 시드 {a.seeds}개, "
         "정답 세계 6가지 중 공정한 세계 평균.",
         "- 칸 = 1순위 균형 정확도 / 정답이 1·2순위 안에 든 비율(행동별 평균). 괄호 = 그 구간 울음 수(시드·세계 합).", ""]
    for K in (3, 4):
        L += [f"## 행동 {K}개 (무작위 {1 / K:.2f} / {2 / K:.2f})", "",
              "| 설정 | " + " | ".join(f"답 {lab(lo, hi)}개" for lo, hi in buckets) + " |", "|---|" + "---:|" * len(buckets)]
        for c in LONG_CFG:
            g = D[(D.K == K) & (D.cfg == c) & D.world.isin(fair[c])]
            cells = []
            for lo, hi in buckets:
                x = g[(g.n >= lo) & (g.n <= hi)]
                if len(x) < 50:
                    cells.append("-")
                    continue
                t1 = np.mean([sa.bal_np(xx.y, xx.pred) for _, xx in x.groupby(["world", "seed"])])
                t2 = np.mean([np.mean([xx.hit2[xx.y == k].mean() for k in xx.y.unique()]) for _, xx in x.groupby(["world", "seed"])])
                on = f", 소리 {x.on.mean():.0%}" if "자가" in c else ""
                cells.append(f"{t1:.2f} / {t2:.2f} ({len(x)}{on})")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L.append("")
    txt = "\n".join(L) + "\n"
    open(os.path.join(sa.RUNS, f"KCLASS_LONG_REPORT{tag}.md"), "w", encoding="utf-8").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("feats")
    p.add_argument("--workers", type=int, default=4)
    sp.add_parser("export")
    p = sp.add_parser("run")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--perms", type=int, default=20)
    p = sp.add_parser("long")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--group", type=int, default=4, help="가상 아기 하나에 묶을 Enes 아기 수")
    a = ap.parse_args()
    {"feats": cmd_feats, "export": cmd_export, "run": cmd_run, "long": cmd_long}[a.cmd](a)


if __name__ == "__main__":
    main()
