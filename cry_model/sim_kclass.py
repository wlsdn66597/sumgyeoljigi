"""행동 개수(3·4·6개)와 MFCC 블록을 파이 모듈(../cry_personal.py) 그대로 재생해 비교한다(10차).

    python sim_kclass.py feats --workers 12   # 울음 2,521개: 파이식 MFCC(울음 판정 창만) + 스펙트럼 모양 + (Enes) 리듬
    python sim_kclass.py export               # artifacts/cry_personal_acoustic.npz에 MFCC 멜 필터·표준화·블록 퍼짐을 더한다
    python sim_kclass.py run --workers 12     # → DATA/runs/synthetic/KCLASS_REPORT.md
    python sim_kclass.py long13 --workers 8   # 13차 기본값으로 긴 사용 기간(아기 11명 → 울음 약 275개) → KCLASS_LONG13_REPORT_g11.md

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
FAMILY = {"yamnet16": {"YAMNet"}, "acoustic": {"요약 6", "eGeMAPS", "MFCC"}, "mfcc": {"MFCC"}, "rhythm": {"리듬"},
          "voice": {"목소리"}, "spectral": {"스펙트럼 모양"}}


def _family_pca(worlds, enes):
    """혼합 세계용: 특징 묶음마다 Enes 울음으로 표준화 → 주성분 5개."""
    out = {}
    for k, F in worlds.items():
        X = np.nan_to_num(F[enes].astype(float))
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        P = np.zeros((len(F), 5))
        P[enes] = Z @ Vt[:5].T
        out[k] = P
    return out


def mixed_world(seed):
    """혼합 세계: 모든 특징 묶음의 주성분 5개를 묶음마다 무작위 비율·무작위 방향으로 섞은 10차원.
    원인이 여러 특징에 조금씩 섞여 드러나는 경우."""
    rng = np.random.default_rng(1000 + seed)
    fams = _W["fam5"]
    w = rng.dirichlet(np.ones(len(fams)))
    return sum(w[i] * (F @ rng.standard_normal((5, 10))) for i, F in enumerate(fams.values()))
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
    cry, embs = [], []
    for w in wins:
        _W["yam"].it.set_tensor(_W["yam"].inp, w.astype(np.float32))
        _W["yam"].it.invoke()
        sc = _W["yam"].it.get_tensor(_W["yam"].out[521]).reshape(-1)
        embs.append(_W["yam"].it.get_tensor(_W["yam"].out[1024]).reshape(-1).copy())
        cry.append(max(sc[19], sc[20]) >= sa.CRY_THR)
    use = [w for w, c in zip(wins, cry) if c] or wins
    emb = np.mean([e for e, c in zip(embs, cry) if c] or embs, 0)          # sim_augment.Yam.clip과 같은 울음 창 평균
    fr = [cp.frame_features(w, _W["fb"]) for w in use]
    ms = cp.audio_stats(*(np.concatenate([f[j] for f in fr]) for j in range(5)))   # MFCC 40 + 리듬 8 + 목소리 14 + 스펙트럼 20
    xs = np.pad(x, (0, max(0, sa.SR - len(x))))
    return k, ms.astype(np.float32), spectral_shape(xs).astype(np.float32), emb.astype(np.float32)


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
    MS = np.zeros((len(m), 82), np.float32)
    SP = np.zeros((len(m), 22), np.float32)
    EM = np.zeros((len(m), 1024), np.float32)
    t0 = time.time()
    with Pool(a.workers, initializer=_init_feats) as pool:
        for n, (k, ms, sp, em) in enumerate(pool.imap_unordered(_feat_one, list(enumerate(m.paths)), chunksize=4)):
            MS[k], SP[k], EM[k] = ms, sp, em
            if (n + 1) % a.every == 0:
                print(f"  {n + 1}/{len(m)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(sa.OUT, exist_ok=True)
    m.drop(columns="paths").to_csv(os.path.join(sa.OUT, "items.csv"), index=False)
    MS = np.nan_to_num(MS)
    np.savez(FEATS2, mfcc_dev=MS[:, :40], rhythm_dev=MS[:, 40:48], audio_dev=MS, spec=np.nan_to_num(SP),
             rhythm=rhythm(m), emb=EM)
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
    f = np.load(FEATS2)
    if os.path.exists(os.path.join(sa.OUT, "prep.npz")):
        m, A, E, G, O, VAR, MODES, view = sa.load_all()
    else:                                                     # 서버 중간 파일 없이(노트북): feats의 울음 임베딩을 쓴다
        m = pd.read_csv(os.path.join(sa.OUT, "items.csv"))
        m["baby"] = m.baby.fillna("").astype(str)
        E = f["emb"].astype(float)[:, None, :]
    z = dict(np.load(ACOUSTIC_NPZ))
    pub = (m.source != "enes").values
    MS = f["audio_dev"].astype(float)
    for k, (lo, hi) in cp.AUDIO_SLICES.items():
        z[f"{k}_mean"] = MS[pub, lo:hi].mean(0).astype(np.float32)
        z[f"{k}_std"] = (MS[pub, lo:hi].std(0) + 1e-6).astype(np.float32)
    z["mel_fb"] = librosa.filters.mel(sr=sa.SR, n_fft=cp.N_FFT).astype(np.float32)
    for k in ("spread_yamnet16", "spread_acoustic", *(f"spread_{b}" for b in cp.AUDIO_SLICES)):
        z.pop(k, None)
    b = np.load(os.path.join(sa.ART, "cry_personal_basis.npz"))
    dev = cp.CryPersonal({k: b[k] for k in ("mean", "std", "pca")}, None, acoustic_path=z,
                         blocks=["yamnet16", "acoustic", *cp.AUDIO_SLICES])
    order = enes_order(m, min_n=2)
    for blk in ("yamnet16", "acoustic", *cp.AUDIO_SLICES):
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
    MS = f2["audio_dev"] if "audio_dev" in f2 else (np.c_[f2["mfcc_dev"], f2["rhythm_dev"]] if "rhythm_dev" in f2
                                                      else f2["mfcc_dev"])
    _W.update(m=m, E0=E[:, 0].copy(), MS=MS.astype(float), basis={k: b[k] for k in ("mean", "std", "pca")},
              ac=dict(np.load(ACOUSTIC_NPZ)), order=enes_order(m),
              worlds={"요약 6": A[:, 0].copy(), "eGeMAPS": f["egemaps"].astype(float), "MFCC": f2["mfcc_dev"].astype(float),
                      "YAMNet": E[:, 0].copy(), "스펙트럼 모양": f2["spec"].astype(float), "리듬": f2["rhythm"]})
    if "audio_dev" in f2:
        _W["worlds"]["목소리"] = f2["audio_dev"][:, 48:62].astype(float)
    _W["fam5"] = _family_pca(_W["worlds"], (m.source == "enes").values)
    del E


def replay(y, actions, order):
    """설정마다 모듈을 아기별로 새로 만들어 시간순 재생. 반환 {설정: 지표}."""
    m, E0, MS = _W["m"], _W["E0"], _W["MS"]
    out = {}
    for name, kw in CFG.items():
        rec = []
        for ix in order.values():
            dev = cp.CryPersonal(_W["basis"], None, acoustic_path=_W["ac"], actions=actions, hint_seconds=10, **{"sound_method": "proto", **kw})
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
            dev = cp.CryPersonal(_W["basis"], None, acoustic_path=_W["ac"], actions=actions, hint_seconds=10, **{"sound_method": "proto", **kw})
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


# --- 5. 더 해 볼 것: 아기별 판별 분석 · '아니었어요' · 답 없는 울음 --------------------------------------
MORE_CFG = {"이력만": dict(w_sound=0.0),
            "proto": dict(sound_method="proto", gate=False),
            "lda": dict(sound_method="lda", gate=False),
            "proto + 자가 검증(지금 기본)": dict(sound_method="proto"),
            "lda + 자가 검증": dict(sound_method="lda")}
MORE_ANS = {"답 100%": (1.0, False, False), "답 50%": (0.5, False, False), "답 50% + 아니었어요": (0.5, True, False),
            "답 50% + 답 없는 울음": (0.5, False, True), "답 50% + 둘 다": (0.5, True, True)}
POS_BUCKETS = [(20, 39), (40, 59), (60, 79), (80, 999)]


def replay_more(y, actions, vorder, En, Mn, seed, cfgs, answers, cfgdict=None):
    """가상 부모: 울음마다 p_full 확률로 최종 답. 답을 안 한 울음에서 힌트가 보였고 1순위가 틀렸으면 '아니었어요'.
    답 여부는 울음마다 미리 뽑아 설정 간에 같다. 반환: (설정, 답 방식, 정답, 1순위, 2순위 적중, 몇 번째 울음, 소리 켜짐)."""
    rec = []
    for vb, ix in vorder.items():
        draw = np.random.default_rng([seed, len(ix), int(ix[0])]).random(len(ix))
        for aname in answers:
            p_full, use_not, semi = MORE_ANS[aname]
            for cname in cfgs:
                if aname not in ("답 100%", "답 50%") and cname in ("이력만", "proto + 자가 검증(지금 기본)"):
                    continue
                dev = cp.CryPersonal(_W["basis"], None, acoustic_path=_W["ac"], actions=actions, hint_seconds=10,
                                     use_not=use_not, semi=semi, **{"sound_method": "proto", **(cfgdict or MORE_CFG)[cname]})
                for pos, i in enumerate(ix):
                    t = 1.7e9 + pos * 3600.0
                    h = dev.hint(t, embs=[En[i]], mfcc=Mn[i])
                    rec.append((cname, aname, y[i], actions.index(h["top"]), actions[y[i]] in h["top2"], pos, h["sound_on"]))
                    dev.start(t)
                    dev.add_window(En[i])
                    eid = dev.end(t + 60, mfcc=(Mn[i], Mn[i]))
                    if draw[pos] < p_full:
                        dev.feedback(eid, actions[y[i]])
                    elif use_not and h["n_feedback"] >= dev.k_min and h["top"] != actions[y[i]]:
                        dev.feedback_not(eid, h["top"])
    return rec


def _virtual(world, seed, gsize, F=None):
    """긴 사용 기간 시험과 같은 가상 아기(아기마다 평균을 빼고 시간 순서를 지켜 번갈아)."""
    m, E0, MS, order = _W["m"], _W["E0"], _W["MS"], _W["order"]
    rng = np.random.default_rng(seed)
    babies = list(order)
    rng.shuffle(babies)
    groups = [babies[i:i + gsize] for i in range(0, len(babies) - len(babies) % gsize, gsize)]
    En, Mn, Fn = E0.copy(), MS.copy(), (_W["worlds"][world] if F is None else F).copy()
    allix = np.concatenate(list(order.values()))
    gE, gM = E0[allix].mean(0), MS[allix].mean(0)
    for ix in order.values():
        En[ix] += gE - E0[ix].mean(0)
        Mn[ix] += gM - MS[ix].mean(0)
        Fn[ix] = (Fn[ix] - Fn[ix].mean(0)) / (Fn[ix].std(0) + 1e-9)
    vorder = {}
    for gi, grp in enumerate(groups):
        pos = np.concatenate([np.arange(len(order[b])) / len(order[b]) for b in grp])
        ix = np.concatenate([order[b] for b in grp])
        vorder[f"v{gi}"] = ix[np.argsort(pos + rng.random(len(pos)) * 1e-6)]
    return vorder, En, Mn, Fn


def _more_task(job):
    kind, world, seed = job
    m = _W["m"]
    if kind == "sim":
        vorder, En, Mn, Fn = _virtual(world, seed, 4)
        y = baby_rule(m, Fn, vorder, 3, None, seed)
        rec = replay_more(y, cp.ACTIONS3, vorder, En, Mn, seed, list(MORE_CFG), list(MORE_ANS))
    else:                                                     # 실제 Enes(답 100%), seed > 0이면 귀무
        order = enes_order(m, min_n=2)
        y = np.array(["FIS".index(s) if s in ("F", "I", "S") else 0 for s in m.stop.astype(str)])
        if seed:
            rng = np.random.default_rng(seed)
            for ix in order.values():
                y[ix] = rng.permutation(y[ix])
        rec = replay_more(y, cp.ACTIONS3, order, _W["E0"], _W["MS"], seed, list(MORE_CFG), ["답 100%"])
    return [(kind, world, seed) + r for r in rec]


def cmd_more(a):
    import warnings
    from multiprocessing import Pool
    warnings.filterwarnings("ignore")
    jobs = [("sim", w, s) for w in ("요약 6", "eGeMAPS", "MFCC", "YAMNet", "스펙트럼 모양", "리듬") for s in range(a.seeds)]
    jobs += [("real", "실제 Enes", s) for s in range(a.perms + 1)]
    rows, t0 = [], time.time()
    with Pool(a.workers, initializer=_init_run) as pool:
        for n, r in enumerate(pool.imap_unordered(_more_task, jobs)):
            rows += r
            if (n + 1) % 12 == 0:
                print(f"  {n + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows, columns=["kind", "world", "seed", "cfg", "ans", "y", "pred", "hit2", "pos", "on"])
    D.to_csv(os.path.join(sa.RUNS, "kclass_more.csv"), index=False)
    report_more(D)


def _score(x):
    """(world, seed)마다 1순위 균형 정확도와 2순위 안 적중률(행동별 평균)을 내서 평균."""
    t1, t2 = [], []
    for _, g in x.groupby(["world", "seed"]):
        t1.append(sa.bal_np(g.y, g.pred))
        t2.append(np.mean([g.hit2[g.y == k].mean() for k in g.y.unique()]))
    return np.mean(t1), np.mean(t2)


def report_more(D):
    fair = ["YAMNet", "스펙트럼 모양", "리듬"]                     # 음향 변환 + MFCC와 무관한 특징으로 만든 세계
    S = D[(D.kind == "sim") & D.world.isin(fair)]
    lab = lambda lo, hi: f"{lo}~{hi}" if hi < 999 else f"{lo}+"  # noqa: E731
    L = ["# 아기별 판별 분석 · '아니었어요' · 답 없는 울음 (행동 3개, 가상 아기 울음 약 100개)", "",
         "- 가상 아기 = Enes 아기 4명을 평균 빼고 이어 붙임(5명 × 시드). 공정한 세계(YAMNet·스펙트럼 모양·리듬) 평균.",
         "- 칸 = 1순위 균형 정확도 / 2순위 안 적중률(무작위 0.33 / 0.67). 구간 = 몇 번째 울음인지(답 비율이 달라도 같은 기간).",
         "- '아니었어요' = 최종 답을 안 한 울음에서 힌트 1순위를 해 봤는데 안 그친 경우만 누름.", "",
         "| 답 방식 | 방법 | " + " | ".join(f"울음 {lab(lo, hi)}번째" for lo, hi in POS_BUCKETS) + " | 소리 켜짐 |",
         "|---|---|" + "---:|" * (len(POS_BUCKETS) + 1)]
    for aname in MORE_ANS:
        for cname in MORE_CFG:
            g = S[(S.ans == aname) & (S.cfg == cname)]
            if g.empty:
                continue
            cells = []
            for lo, hi in POS_BUCKETS:
                t1, t2 = _score(g[(g.pos >= lo) & (g.pos <= hi)])
                cells.append(f"{t1:.2f} / {t2:.2f}")
            on = f"{g[g.pos >= 20].on.mean():.0%}" if "자가" in cname else "-"
            L.append(f"| {aname} | {cname} | " + " | ".join(cells) + f" | {on} |")
    R = D[D.kind == "real"]
    L += ["", "## 실제 Enes (답 100%, 아기 24명, 답 10개 이상 = 11번째 울음부터)", "",
          "| 방법 | 실제 | 귀무 95% (아기 안 라벨 섞기 20번) | 소리 켜짐 |", "|---|---:|---:|---:|"]
    for cname in MORE_CFG:
        x = R[(R.cfg == cname) & (R.pos >= 10)]
        r1, r2 = _score(x[x.seed == 0])
        nul = [_score(g) for _, g in x[x.seed > 0].groupby("seed")]
        n1, n2 = np.quantile([v[0] for v in nul], .95), np.quantile([v[1] for v in nul], .95)
        L.append(f"| {cname} | {r1:.3f} / {r2:.3f} | {n1:.3f} / {n2:.3f} | "
                 + (f"{x[x.seed == 0].on.mean():.0%}" if "자가" in cname else "-") + " |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(sa.RUNS, "KCLASS_MORE_REPORT.md"), "w", encoding="utf-8").write(txt)
    print(txt)


# --- 6. 리듬·시간 특징 블록 ----------------------------------------------------------
RHY_CFG = {"이력만": dict(w_sound=0.0),
           "음향+MFCC (지금)": dict(blocks=["acoustic", "mfcc"], gate=False),
           "음향+MFCC+리듬": dict(blocks=["acoustic", "mfcc", "rhythm"], gate=False),
           "리듬만": dict(blocks=["rhythm"], gate=False),
           "음향+MFCC, 자가 검증 (지금 기본)": dict(blocks=["acoustic", "mfcc"]),
           "음향+MFCC+리듬, 자가 검증": dict(blocks=["acoustic", "mfcc", "rhythm"])}


def _rhy_task(job):
    kind, world, seed, gsize = job
    m = _W["m"]
    if kind == "sim":
        vorder, En, Mn, Fn = _virtual(world, seed, gsize)
        y = baby_rule(m, Fn, vorder, 3, None, seed)
        rec = replay_more(y, cp.ACTIONS3, vorder, En, Mn, seed, list(RHY_CFG), ["답 100%"], RHY_CFG)
    else:
        order = enes_order(m, min_n=2)
        y = np.array(["FIS".index(s) if s in ("F", "I", "S") else 0 for s in m.stop.astype(str)])
        if seed:
            rng = np.random.default_rng(seed)
            for ix in order.values():
                y[ix] = rng.permutation(y[ix])
        rec = replay_more(y, cp.ACTIONS3, order, _W["E0"], _W["MS"], seed, list(RHY_CFG), ["답 100%"], RHY_CFG)
    return [(kind, world, seed) + r for r in rec]


def cmd_rhythm(a):
    import warnings
    from multiprocessing import Pool
    warnings.filterwarnings("ignore")
    jobs = [("sim", w, s, a.group) for w in ("요약 6", "eGeMAPS", "MFCC", "YAMNet", "스펙트럼 모양", "리듬")
            for s in range(a.seeds)]
    jobs += [("real", "실제 Enes", s, a.group) for s in range(a.perms + 1)]
    buckets = POS_BUCKETS if a.group <= 4 else [(20, 39), (40, 79), (80, 119), (120, 159), (160, 199), (200, 999)]
    tag = "" if a.group == 4 else f"_g{a.group}"
    rows, t0 = [], time.time()
    with Pool(a.workers, initializer=_init_run) as pool:
        for n, r in enumerate(pool.imap_unordered(_rhy_task, jobs)):
            rows += r
            if (n + 1) % 12 == 0:
                print(f"  {n + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows, columns=["kind", "world", "seed", "cfg", "ans", "y", "pred", "hit2", "pos", "on"])
    D.to_csv(os.path.join(sa.RUNS, f"kclass_rhythm{tag}.csv"), index=False)
    lab = lambda lo, hi: f"{lo}~{hi}" if hi < 999 else f"{lo}+"  # noqa: E731
    L = [f"# 리듬·시간 특징 블록 (행동 3개, 아기 {a.group}명을 이어 붙인 가상 아기 울음 약 {25 * a.group}개, 답 100%)", "",
         "- 리듬 8개 = 프레임 세기에서 소리 나는 비율, 초당 덩어리 수, 덩어리·쉼 길이와 변동, 세기 기울기, 세기 폭.",
         "- 공정 = YAMNet·스펙트럼 모양 세계(어떤 블록과도 무관). 리듬 세계 = Enes 메타(음절 속도·쉼·유성·울음 길이)로 만든",
         "  정답이라 리듬 블록과 개념이 겹친다('배울 수 있는가' 확인용). 칸 = 1순위 / 2순위 안(무작위 0.33 / 0.67).", ""]
    groups = (("공정한 세계 (YAMNet·스펙트럼 모양)", ["YAMNet", "스펙트럼 모양"]), ("리듬 세계", ["리듬"]),
              ("음향 특징 세계 (요약 6·eGeMAPS·MFCC, 음향+MFCC에는 순환)", ["요약 6", "eGeMAPS", "MFCC"]))
    for title, ws in groups:
        S = D[(D.kind == "sim") & D.world.isin(ws)]
        L += [f"## {title}", "",
              "| 방법 | " + " | ".join(f"울음 {lab(lo, hi)}번째" for lo, hi in buckets) + " | 소리 켜짐 |",
              "|---|" + "---:|" * (len(buckets) + 1)]
        for c in RHY_CFG:
            g = S[S.cfg == c]
            cells = ["%.2f / %.2f" % _score(g[(g.pos >= lo) & (g.pos <= hi)]) for lo, hi in buckets]
            on = f"{g[g.pos >= 20].on.mean():.0%}" if "자가" in c else "-"
            L.append(f"| {c} | " + " | ".join(cells) + f" | {on} |")
        L.append("")
    R = D[D.kind == "real"]
    L += ["## 실제 Enes (아기 24명, 11번째 울음부터)", "", "| 방법 | 실제 | 귀무 95% | 소리 켜짐 |", "|---|---:|---:|---:|"]
    for c in RHY_CFG:
        x = R[(R.cfg == c) & (R.pos >= 10)]
        r1, r2 = _score(x[x.seed == 0])
        nul = [_score(g) for _, g in x[x.seed > 0].groupby("seed")]
        on = f"{x[x.seed == 0].on.mean():.0%}" if "자가" in c else "-"
        nn = (f"{np.quantile([v[0] for v in nul], .95):.3f} / {np.quantile([v[1] for v in nul], .95):.3f}"
              if nul else "-")
        L.append(f"| {c} | {r1:.3f} / {r2:.3f} | {nn} | {on} |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(sa.RUNS, f"KCLASS_RHYTHM_REPORT{tag}.md"), "w", encoding="utf-8").write(txt)
    print(txt)


# --- 7. 목소리 질·스펙트럼 블록과 모든 블록 합치기 (가상 세계 점수 최대화) --------------------------------
VOICE_CFG = {"이력만": dict(w_sound=0.0),
             "지금 (음향+MFCC+리듬)": dict(blocks=["acoustic", "mfcc", "rhythm"], gate=False),
             "+목소리": dict(blocks=["acoustic", "mfcc", "rhythm", "voice"], gate=False),
             "+스펙트럼": dict(blocks=["acoustic", "mfcc", "rhythm", "spectral"], gate=False),
             "+목소리+스펙트럼": dict(blocks=["acoustic", "mfcc", "rhythm", "voice", "spectral"], gate=False),
             "전부(+YAMNet16)": dict(blocks=["yamnet16", "acoustic", "mfcc", "rhythm", "voice", "spectral"], gate=False),
             "전부, 아기별 판별 분석": dict(blocks=["yamnet16", "acoustic", "mfcc", "rhythm", "voice", "spectral"],
                                     sound_method="lda", gate=False),
             "목소리만": dict(blocks=["voice"], gate=False)}
VOICE_WORLDS = ("요약 6", "eGeMAPS", "MFCC", "YAMNet", "스펙트럼 모양", "리듬", "목소리", "혼합")


def _voice_task(job):
    world, seed = job
    m = _W["m"]
    F = mixed_world(seed) if world == "혼합" else None
    vorder, En, Mn, Fn = _virtual(world, seed, 4, F)
    y = baby_rule(m, Fn, vorder, 3, None, seed)
    rec = replay_more(y, cp.ACTIONS3, vorder, En, Mn, seed, list(VOICE_CFG), ["답 100%"], VOICE_CFG)
    return [("sim", world, seed) + r for r in rec]


def cmd_voice(a):
    import warnings
    from multiprocessing import Pool
    warnings.filterwarnings("ignore")
    jobs = [(w, s) for w in VOICE_WORLDS for s in range(a.seeds)]
    rows, t0 = [], time.time()
    with Pool(a.workers, initializer=_init_run) as pool:
        for n, r in enumerate(pool.imap_unordered(_voice_task, jobs)):
            rows += r
            if (n + 1) % 12 == 0:
                print(f"  {n + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows, columns=["kind", "world", "seed", "cfg", "ans", "y", "pred", "hit2", "pos", "on"])
    D.to_csv(os.path.join(sa.RUNS, "kclass_voice.csv"), index=False)
    report_voice(D)


def report_voice(D):
    S = D[D.pos >= 20]
    worlds = [w for w in VOICE_WORLDS if w in set(S.world)]
    sc = {(c, w): _score(S[(S.cfg == c) & (S.world == w)]) for c in VOICE_CFG for w in worlds}
    L = ["# 목소리 질·스펙트럼 블록과 블록 합치기 — 가상 세계 점수 (행동 3개, 가상 아기 울음 약 100개, 답 100%)", "",
         "- 칸 = 20번째 울음부터의 1순위 균형 정확도 / 2순위 안 적중률(무작위 0.33 / 0.67). 소리는 항상 켬(자가 검증 없음).",
         "- 정답 세계 8가지: 특징 묶음 7가지 + 혼합(모든 묶음을 무작위 비율로 섞음). **굵게** = 그 공간이 같은 특징을 쓰는 세계(순환).",
         "- 전체 평균 = 8개 세계 평균(목표 점수). 공정 평균 = 순환 세계를 뺀 평균.", "",
         "| 비교 공간 | " + " | ".join(worlds) + " | **전체 평균** | 공정 평균 |", "|---|" + "---:|" * (len(worlds) + 2)]
    for c, kw in VOICE_CFG.items():
        circ = set().union(*[FAMILY[b] for b in kw.get("blocks", [])]) if c != "이력만" else set()
        cells = [(f"**{sc[(c, w)][0]:.2f} / {sc[(c, w)][1]:.2f}**" if w in circ else f"{sc[(c, w)][0]:.2f} / {sc[(c, w)][1]:.2f}")
                 for w in worlds]
        allm = np.mean([sc[(c, w)] for w in worlds], 0)
        fair = [sc[(c, w)] for w in worlds if w not in circ]
        fm = np.mean(fair, 0) if fair else (np.nan, np.nan)
        L.append(f"| {c} | " + " | ".join(cells) + f" | **{allm[0]:.3f} / {allm[1]:.3f}** | {fm[0]:.3f} / {fm[1]:.3f} |")
    lab = lambda lo, hi: f"{lo}~{hi}" if hi < 999 else f"{lo}+"  # noqa: E731
    L += ["", "## 전체 평균을 울음 구간별로", "", "| 비교 공간 | " + " | ".join(f"울음 {lab(lo, hi)}번째" for lo, hi in POS_BUCKETS) + " |",
          "|---|" + "---:|" * len(POS_BUCKETS)]
    for c in VOICE_CFG:
        cells = []
        for lo, hi in POS_BUCKETS:
            x = D[(D.cfg == c) & (D.pos >= lo) & (D.pos <= hi)]
            v = np.mean([_score(x[x.world == w]) for w in worlds], 0)
            cells.append(f"{v[0]:.2f} / {v[1]:.2f}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(sa.RUNS, "KCLASS_VOICE_REPORT.md"), "w", encoding="utf-8").write(txt)
    print(txt)


# --- 8. 13차 기본값으로 긴 사용 기간 (답이 몇 개 쌓이면 쓸 만한가) ---------------------------------------
# `long`(10차)은 그때 설정(음향 변환 + MFCC, proto)으로 고정돼 있어 13차 기본값을 재지 않는다. 이 시험은 13차와 같은
# 8가지 정답 세계·같은 재생(replay_more)을 쓰고, 가상 아기만 길게(아기 11명 → 울음 약 275개) 만든다.
LONG13_CFG = {"이력만": dict(w_sound=0.0),
              "12차 기본 (음향+MFCC+리듬, 자가 검증)": dict(blocks=["acoustic", "mfcc", "rhythm"]),
              "13차 기본 (전부 + 아기별 판별 분석, 자가 검증)": dict(sound_method="lda"),
              "13차, 자가 검증 끔": dict(sound_method="lda", gate=False)}
LONG13_BUCKETS = [(10, 19), (20, 39), (40, 79), (80, 119), (120, 159), (160, 199), (200, 999)]


def _long13_task(job):
    world, seed, gsize, answers = job
    m = _W["m"]
    F = mixed_world(seed) if world == "혼합" else None
    vorder, En, Mn, Fn = _virtual(world, seed, gsize, F)
    y = baby_rule(m, Fn, vorder, 3, None, seed)
    rec = replay_more(y, cp.ACTIONS3, vorder, En, Mn, seed, list(LONG13_CFG), answers, LONG13_CFG)
    return [("sim", world, seed) + r for r in rec]


def cmd_long13(a):
    """13차 기본값(파이 모듈 기본: 블록 6개 + 아기별 판별 분석 + 자가 검증)을 긴 사용 기간으로 잰다.
    칸 = 그 구간 울음의 1순위 균형 정확도 / 2순위 안 적중률을 (세계, 시드)마다 내서 8개 세계 평균."""
    import warnings
    from multiprocessing import Pool
    warnings.filterwarnings("ignore")
    answers = ["답 100%"] + (["답 50%"] if a.half else [])
    jobs = [(w, s, a.group, answers) for w in VOICE_WORLDS for s in range(a.seeds)]
    rows, t0 = [], time.time()
    with Pool(a.workers, initializer=_init_run) as pool:
        for n, r in enumerate(pool.imap_unordered(_long13_task, jobs)):
            rows += r
            if (n + 1) % 8 == 0:
                print(f"  {n + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows, columns=["kind", "world", "seed", "cfg", "ans", "y", "pred", "hit2", "pos", "on"])
    tag = f"_g{a.group}"
    os.makedirs(sa.RUNS, exist_ok=True)
    D.to_csv(os.path.join(sa.RUNS, f"kclass_long13{tag}.csv"), index=False)
    report_long13(D, a.group, a.seeds)


def report_long13(D, group, seeds):
    worlds = [w for w in VOICE_WORLDS if w in set(D.world)]
    nb = 22 // group
    lab = lambda lo, hi: f"{lo}~{hi}" if hi < 999 else f"{lo}+"  # noqa: E731
    L = [f"# 13차 기본값의 긴 사용 기간 — 쌓인 답 수별 정확도 (행동 3개, 아기 {group}명을 이어 붙인 가상 아기, 울음 약 {25 * group}개)", "",
         f"- 가상 아기 {nb}명 × 시드 {seeds}개 × 정답 세계 {len(worlds)}가지(13차와 같음: 특징 묶음 7가지 + 혼합). "
         "아기마다 평균을 빼 목소리 차이를 지우고 시간 순서를 지켜 번갈아 넣었다.",
         "- 칸 = 그 구간 울음의 1순위 균형 정확도 / 정답이 1·2순위 안에 든 비율(행동별 평균). 무작위 0.33 / 0.67.",
         "- 구간 = 몇 번째 울음인지. 답 100%면 그때까지 쌓인 답 수와 같다. 힌트는 답 10개부터 보인다.",
         "- 13차 기본은 블록 6개를 모두 써서 혼합 외 7개 세계와 특징이 겹친다(순환). 혼합 세계 줄을 함께 본다.", ""]
    for aname in dict.fromkeys(D.ans):
        A = D[D.ans == aname]
        for title, ws in ((f"{len(worlds)}개 세계 평균 ({aname})", worlds), (f"혼합 세계만 ({aname})", ["혼합"])):
            if not set(ws) & set(A.world):
                continue
            L += [f"## {title}", "", "| 설정 | " + " | ".join(f"울음 {lab(lo, hi)}번째" for lo, hi in LONG13_BUCKETS) + " | 소리 켜짐(20번째~) |",
                  "|---|" + "---:|" * (len(LONG13_BUCKETS) + 1)]
            for c in LONG13_CFG:
                g = A[(A.cfg == c) & A.world.isin(ws)]
                if g.empty:
                    continue
                cells = []
                for lo, hi in LONG13_BUCKETS:
                    x = g[(g.pos >= lo) & (g.pos <= hi)]
                    if len(x) < 50 * len(ws):
                        cells.append("-")
                        continue
                    v = np.mean([_score(x[x.world == w]) for w in ws if (x.world == w).any()], 0)
                    cells.append(f"{v[0]:.2f} / {v[1]:.2f}")
                on = f"{g[g.pos >= 20].on.mean():.0%}" if "자가 검증)" in c else "-"
                L.append(f"| {c} | " + " | ".join(cells) + f" | {on} |")
            n = [int(((A.cfg == "이력만") & A.world.isin(ws) & (A.pos >= lo) & (A.pos <= hi)).sum()) for lo, hi in LONG13_BUCKETS]
            L += ["| (울음 수) | " + " | ".join(str(v) for v in n) + " | |", ""]
    txt = "\n".join(L) + "\n"
    open(os.path.join(sa.RUNS, f"KCLASS_LONG13_REPORT_g{group}.md"), "w", encoding="utf-8").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("feats")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--every", type=int, default=500, help="진행 표시 간격")
    sp.add_parser("export")
    p = sp.add_parser("run")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--perms", type=int, default=20)
    p = sp.add_parser("long")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--group", type=int, default=4, help="가상 아기 하나에 묶을 Enes 아기 수")
    p = sp.add_parser("more")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--perms", type=int, default=20)
    p = sp.add_parser("rhythm")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--group", type=int, default=4, help="가상 아기 하나에 묶을 Enes 아기 수(11이면 울음 약 275개)")
    p.add_argument("--perms", type=int, default=20)
    p = sp.add_parser("voice")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=6)
    p = sp.add_parser("long13", help="13차 기본값으로 긴 사용 기간(답 수별 정확도)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--group", type=int, default=11, help="가상 아기 하나에 묶을 Enes 아기 수(11이면 울음 약 275개)")
    p.add_argument("--half", action="store_true", help="부모가 절반만 답하는 경우(답 50%%)도 함께 잰다")
    a = ap.parse_args()
    {"feats": cmd_feats, "export": cmd_export, "run": cmd_run, "long": cmd_long, "more": cmd_more,
     "rhythm": cmd_rhythm, "voice": cmd_voice, "long13": cmd_long13}[a.cmd](a)


if __name__ == "__main__":
    main()
