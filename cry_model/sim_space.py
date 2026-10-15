"""같은 아기 개인화에 쓸 '비교 공간(변환)'을 공정하게 고른다(9차). sim_augment.py의 후속.

    python sim_space.py feats --workers 12   # 울음 2,521개(앞 20초): eGeMAPS 88 + MFCC 40 (원본만)
    python sim_space.py run                  # → DATA/runs/synthetic/SPACE_REPORT.md

문제: 8차에서 'YAMNet → 음향 요약 6개' 변환이 이겼지만, 가상 정답도 그 6개로 만들었다(시험이 그 변환에 유리).
그래서 정답 규칙을 서로 다른 특징 묶음으로 만든 가상 세계 4개에서 모든 공간을 시험하고,
자기와 같은 특징 묶음으로 만든 세계(순환)를 뺀 평균으로 순위를 매긴다.
  정답 세계(모두 ③ 아기마다 다른 규칙: 아기 안에서 표준화 → 3유형 → 아기마다 다른 행동 짝)
    음향 요약 6 / eGeMAPS 88 / MFCC 40 / YAMNet 임베딩
  평가: 아기마다 시간순 재생(먼저 맞히고 답 받기), 답 10개 이상 쌓인 뒤의 균형 정확도, 시드 5개.
  변환은 공개 데이터(Enes 제외)로만 배우고 원인 라벨은 쓰지 않는다.
"""
import argparse
import os
import sys
import time

if len(sys.argv) > 1 and sys.argv[1] == "feats":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_augment as sa  # noqa: E402

FEATS = os.path.join(sa.OUT, "feats.npz")
_W = {}


# --- 1. 특징 -----------------------------------------------------------------------
def _init():
    import opensmile
    _W["smile"] = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                  feature_level=opensmile.FeatureLevel.Functionals)


def mfcc_stats(x):
    """소리 나는 프레임(최대 -30dB 이내)의 MFCC 20개 평균·표준편차."""
    import librosa
    mf = librosa.feature.mfcc(y=x, sr=sa.SR, n_mfcc=20, n_fft=1024, hop_length=256)
    r = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0][: mf.shape[1]]
    db = 20 * np.log10(r + 1e-9)
    act = db > db.max() - 30
    mf = mf[:, act] if act.sum() >= 3 else mf
    return np.r_[mf.mean(1), mf.std(1)]


def _one(job):
    k, paths = job
    x = sa.load_clip(paths)
    if len(x) < sa.SR:
        x = np.pad(x, (0, sa.SR - len(x)))
    return k, _W["smile"].process_signal(x, sa.SR).values[0].astype(np.float32), mfcc_stats(x).astype(np.float32)


def cmd_feats(a):
    from multiprocessing import Pool
    m = sa.items()
    G = np.zeros((len(m), 88), np.float32)
    M = np.zeros((len(m), 40), np.float32)
    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for n, (k, g, mf) in enumerate(pool.imap_unordered(_one, list(enumerate(m.paths)), chunksize=4)):
            G[k], M[k] = g, mf
            if (n + 1) % 500 == 0:
                print(f"  {n + 1}/{len(m)} {time.time() - t0:.0f}s", flush=True)
    np.savez(FEATS, egemaps=np.nan_to_num(G), mfcc=np.nan_to_num(M))
    print(f"저장 {FEATS} {time.time() - t0:.0f}s", flush=True)


# --- 2. 변환 -----------------------------------------------------------------------
def ridge_map(m, X, T, versions=1, nonlinear=False, return_model=False):
    """공개 데이터로 YAMNet(X: [N, V, 1024]) → 목표 특징(T: [N, d], 원본 기준)을 배운다. 규제 세기는 공개 그룹 20%로 고른다.
    반환: 모든 울음 원본의 예측 [N, d](공개 데이터 기준 표준화 단위)."""
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    pub = (m.source != "enes").values
    xm, xs = X[pub, 0].mean(0), X[pub, 0].std(0) + 1e-6
    tm, ts = T[pub].mean(0), T[pub].std(0) + 1e-9
    Xs, Ts = (X - xm) / xs, (T - tm) / ts
    ug = np.random.default_rng(0).permutation(np.unique(m.group.values[pub]))
    hold = pub & m.group.isin(ug[: len(ug) // 5]).values
    fit = pub & ~hold
    Xf, Tf = Xs[fit, :versions].reshape(-1, X.shape[-1]), np.repeat(Ts[fit], versions, axis=0)
    if nonlinear:
        make = lambda al: MLPRegressor((256,), alpha=al, max_iter=300, early_stopping=True, random_state=0)  # noqa: E731
        grid = (1e-3, 1e-2, 1e-1, 1.0)
    else:
        make = lambda al: Ridge(alpha=al)  # noqa: E731
        grid = np.logspace(0, 5, 11)
    r2 = lambda g: (1 - ((g.predict(Xs[hold, 0]) - Ts[hold]) ** 2).sum(0)  # noqa: E731
                    / ((Ts[hold] - Ts[hold].mean(0)) ** 2).sum(0)).mean()
    best = max(grid, key=lambda al: r2(make(al).fit(Xf, Tf)))
    g = make(best).fit(Xs[pub, :versions].reshape(-1, X.shape[-1]), np.repeat(Ts[pub], versions, axis=0))
    if return_model:
        return g.predict(Xs[:, 0]), r2(make(best).fit(Xf, Tf)), {"x_mean": xm, "x_std": xs, "coef": g.coef_,
                                                                  "intercept": g.intercept_, "alpha": best}
    return g.predict(Xs[:, 0]), r2(make(best).fit(Xf, Tf))


def zpub(m, F):
    pub = (m.source != "enes").values
    return (F - F[pub].mean(0)) / (F[pub].std(0) + 1e-9)


# --- 3. 정답 세계 · 평가 ---------------------------------------------------------------
def baby_rule(m, F, babies, seed):
    """③ 아기마다 다른 규칙: 아기 안에서 표준화(차원이 많으면 PCA 10) → k-평균 3유형 → 아기마다 다른 행동 짝.
    유형 크기는 같게 맞춘다(k-평균 중심은 그대로, 각 유형에 1/3씩 배정). 맞추지 않으면 YAMNet 세계는 22명 모두
    가장 작은 유형이 10% 미만이라 '그 아기의 흔한 답 찍기'만으로 점수가 났다(귀무 knn 0.65)."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    y = np.zeros(len(m), int)
    for b in babies:
        ix = np.where(m.baby.values == b)[0]
        Z = (F[ix] - F[ix].mean(0)) / (F[ix].std(0) + 1e-9)
        if Z.shape[1] > 10:
            Z = PCA(10, random_state=seed).fit_transform(Z)
        C = KMeans(3, n_init=10, random_state=seed).fit(Z).cluster_centers_
        cap = int(np.ceil(len(ix) / 3))
        D = ((Z[:, None] - C[None]) ** 2).sum(-1)
        _, col = linear_sum_assignment(np.repeat(D, cap, axis=1))
        y[ix] = rng.permutation(3)[col // cap]
    return y


def replay(R, y, order, method, n_min=10):
    """시간순 재생: 울음마다 이전 답으로 먼저 맞힌다. 답 n_min개 이상 쌓인 뒤의 (정답, 예측, 답 수).
    method == "history"면 지금까지 가장 많이 받은 답을 찍는다."""
    S = R[:, None, :] if R is not None else None
    out = []
    for ix in order.values():
        for t in range(n_min, len(ix)):
            if method == "history":
                c = np.bincount(y[ix[:t]], minlength=3) + np.arange(3) * 1e-6
                out.append((y[ix[t]], int(c.argmax()), t))
                continue
            p = sa.device_probs(S, y, ix[:t], ix[t], method)
            out.append((y[ix[t]], int(p.argmax()), t))
    return np.array(out)


def cmd_run(a):
    import warnings
    warnings.filterwarnings("ignore")
    m, A, E, G, O, VAR, MODES, view = sa.load_all()
    f = np.load(FEATS)
    GM, MF = f["egemaps"].astype(np.float64), f["mfcc"].astype(np.float64)
    A0 = A[:, 0]
    S0, _ = sa.spaces(m, A, E, view)
    b = np.load(os.path.join(sa.ART, "cry_personal_basis.npz"))
    t0 = time.time()
    reps, r2s = {}, {}
    reps["YAMNet 16 (지금 파이)"] = S0["YAMNet 16차원(파이 기본)"][:, 0]
    reps["YAMNet 1024 전체(표준화)"] = (E[:, 0] - b["mean"]) / b["std"]
    reps["YAMNet→요약 6 (선형, 증강)"] = S0["음향 요약 예측(증강으로 배움)"][:, 0]
    reps["YAMNet→요약 6 (선형)"] = S0["음향 요약 예측(원본으로 배움)"][:, 0]
    reps["YAMNet→요약 6 (MLP, 증강)"], r2s["MLP 6"] = ridge_map(m, E[:, :5], A0, versions=5, nonlinear=True)
    reps["YAMNet→eGeMAPS 88 (선형)"], r2s["eGeMAPS"] = ridge_map(m, E, GM)
    reps["YAMNet→MFCC 40 (선형)"], r2s["MFCC"] = ridge_map(m, E, MF)
    allT = np.c_[A0, GM, MF]
    reps["YAMNet→음향 전체 134 (선형)"], r2s["전체"] = ridge_map(m, E, allT)
    reps["직접 계산: 요약 6"] = zpub(m, A0)
    reps["직접 계산: eGeMAPS 88"] = zpub(m, GM)
    reps["직접 계산: MFCC 40"] = zpub(m, MF)
    enes = (m.source == "enes").values
    babies = sorted(bb for bb, n in m.baby[enes].value_counts().items() if n >= 15)
    order = {}
    for bb in babies:
        ix = np.where(m.baby.values == bb)[0]
        order[bb] = ix[np.lexsort((m.bout.values[ix], m.t.values[ix]))]

    def spread(R):                                               # 아기 안 퍼짐(공개 데이터 극단값 영향 없음)
        d = np.concatenate([R[ix] - R[ix].mean(0) for ix in order.values()])
        return np.sqrt((d ** 2).sum(1).mean())
    y16 = reps["YAMNet 16 (지금 파이)"]
    for name in ("직접 계산: MFCC 40", "YAMNet→eGeMAPS 88 (선형)"):
        r = reps[name]
        reps[f"YAMNet 16 + {name.replace('직접 계산: ', '직접 ')}"] = np.c_[y16 / spread(y16), r / spread(r)]
    print(f"  변환 {time.time() - t0:.0f}s, R² " + ", ".join(f"{k} {v:.2f}" for k, v in r2s.items()), flush=True)
    family = {"YAMNet 16 (지금 파이)": "YAMNet", "YAMNet 1024 전체(표준화)": "YAMNet",
              "YAMNet→요약 6 (선형, 증강)": "요약 6", "YAMNet→요약 6 (선형)": "요약 6", "YAMNet→요약 6 (MLP, 증강)": "요약 6",
              "YAMNet→eGeMAPS 88 (선형)": "eGeMAPS", "YAMNet→MFCC 40 (선형)": "MFCC",
              "YAMNet→음향 전체 134 (선형)": "전체", "직접 계산: 요약 6": "요약 6", "직접 계산: eGeMAPS 88": "eGeMAPS",
              "직접 계산: MFCC 40": "MFCC", "YAMNet 16 + 직접 MFCC 40": "YAMNet+MFCC",
              "YAMNet 16 + YAMNet→eGeMAPS 88 (선형)": "YAMNet+eGeMAPS"}
    worlds = {"요약 6": A0, "eGeMAPS": GM, "MFCC": MF, "YAMNet": E[:, 0]}
    reps["이력만(가장 많이 받은 답)"] = None
    family["이력만(가장 많이 받은 답)"] = ""
    rows = []
    for wname, F in worlds.items():
        for seed in range(a.seeds):
            y = baby_rule(m, F, babies, seed)
            for rname, R in reps.items():
                for me in ("proto", "knn"):                       # 이력만은 방법이 없어 두 칸에 같은 값
                    D = replay(R, y, order, me if R is not None else "history")
                    rows.append({"world": wname, "seed": seed, "rep": rname, "method": me,
                                 "bal10": sa.bal_np(D[:, 0], D[:, 1]),
                                 "bal20": sa.bal_np(D[D[:, 2] >= 20, 0], D[D[:, 2] >= 20, 1])})
            for me in ("proto", "knn"):                          # 귀무: 아기 안에서 라벨 섞기
                yp = y.copy()
                rng = np.random.default_rng(900 + seed)
                for ix in order.values():
                    yp[ix] = rng.permutation(yp[ix])
                D = replay(reps["YAMNet 16 (지금 파이)"], yp, order, me)
                rows.append({"world": wname, "seed": seed, "rep": "귀무(라벨 섞음)", "method": me,
                             "bal10": sa.bal_np(D[:, 0], D[:, 1]), "bal20": sa.bal_np(D[D[:, 2] >= 20, 0], D[D[:, 2] >= 20, 1])})
        print(f"  세계 {wname} {time.time() - t0:.0f}s", flush=True)
    T = pd.DataFrame(rows)
    os.makedirs(sa.RUNS, exist_ok=True)
    T.to_csv(os.path.join(sa.RUNS, "space_compare.csv"), index=False)
    report(T, family, r2s, len(babies))


def report(T, family, r2s, n_b):
    W = list(dict.fromkeys(T.world))
    L = ["# 9차: 같은 아기 개인화의 비교 공간 고르기 (정답을 여러 특징으로 만든 가상 세계)", "",
         f"- Enes 아기 {n_b}명, 시간순 재생(먼저 맞히고 답 받기), 답 10개 이상 쌓인 뒤 균형 정확도(무작위 0.333), 시드 5개 평균.",
         "- 정답 세계 = ③ 아기마다 다른 규칙을 각 특징 묶음으로 만든 것. **굵은 칸 = 자기 특징으로 만든 세계(순환, 순위에서 뺌).**",
         "- 공정 평균 = 순환이 아닌 세계들의 평균. 칸은 proto / knn.",
         "- 변환 설명력 R²(공개 데이터 처음 보는 그룹): " + ", ".join(f"{k} {v:.2f}" for k, v in r2s.items()), "",
         "| 비교 공간 | " + " | ".join(f"정답: {w}" for w in W) + " | 공정 평균 (proto / knn) | 답 20개 이상 공정 평균 |",
         "|---|" + "---:|" * (len(W) + 2)]
    rank = []
    for rname, g in T.groupby("rep", sort=False):
        v = g.groupby(["world", "method"])[["bal10", "bal20"]].mean()
        fam = family.get(rname, "")
        circ = [w for w in W if w == fam or (fam == "전체" and w != "YAMNet")
                or (fam.startswith("YAMNet+") and w in ("YAMNet", fam.split("+")[1]))]
        cells = []
        for w in W:
            c = f"{v.bal10[(w, 'proto')]:.2f} / {v.bal10[(w, 'knn')]:.2f}"
            cells.append(f"**{c}**" if w in circ else c)
        fair = [w for w in W if w not in circ] or W
        fp = np.mean([v.bal10[(w, "proto")] for w in fair])
        fk = np.mean([v.bal10[(w, "knn")] for w in fair])
        f20 = np.mean([max(v.bal20[(w, "proto")], v.bal20[(w, "knn")]) for w in fair])
        rank.append((max(fp, fk), rname))
        L.append(f"| {rname} | " + " | ".join(cells) + f" | {fp:.3f} / {fk:.3f} | {f20:.3f} |")
    L += ["", "공정 평균 순위(proto·knn 중 나은 쪽): " + " > ".join(f"{r} {s:.3f}" for s, r in sorted(rank, reverse=True)[:6])]
    txt = "\n".join(L) + "\n"
    open(os.path.join(sa.RUNS, "SPACE_REPORT.md"), "w", encoding="utf-8").write(txt)
    print(txt)


# --- 4. 파이 배포: 변환 파일 + 모듈 그대로 재생 -----------------------------------------------
def acoustic_targets(m, A, f):
    import opensmile
    gm = list(opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                              feature_level=opensmile.FeatureLevel.Functionals).feature_names)
    names = ["f0_med", "f0_iqr", "loud_db", "loud_cv", "active", "centroid"] + gm + \
        [f"mfcc{i}_mean" for i in range(20)] + [f"mfcc{i}_std" for i in range(20)]
    return np.c_[A[:, 0], f["egemaps"], f["mfcc"]].astype(np.float64), names


def cmd_export(_):
    """artifacts/cry_personal_acoustic.npz: YAMNet 임베딩(1024) → 음향 특징 134개 선형 변환(공개 데이터, 원인 라벨 없음)."""
    m, A, E, G, O, VAR, MODES, view = sa.load_all()
    T, names = acoustic_targets(m, A, np.load(FEATS))
    _, r2, p = ridge_map(m, E, T, return_model=True)
    out = os.path.join(sa.ART, "cry_personal_acoustic.npz")
    np.savez(out, x_mean=p["x_mean"].astype(np.float32), x_std=p["x_std"].astype(np.float32),
             coef=p["coef"].astype(np.float32), intercept=p["intercept"].astype(np.float32), names=np.array(names))
    print(f"저장 {out}: 1024 → {len(names)}, 규제 {p['alpha']:.0f}, 처음 보는 그룹 R² {r2:.2f}")


COLS = ["bal", "on", "top2", "top2_10", "top2_20"]


def cmd_device(a):
    """../cry_personal.py를 그대로 시간순 재생해 설정을 비교한다(답 10개 이상 쌓인 뒤 힌트의 균형 정확도).
    가상 ③ 세계 4개(유형 크기 균등)와 실제 Enes 라벨. 맥락 모델은 쓰지 않는다(이력 × 소리만 비교)."""
    import warnings
    warnings.filterwarnings("ignore")
    sys.path[:0] = [os.path.dirname(sa.HERE), sa.HERE]           # 저장소 루트(로컬) 또는 같은 폴더(서버)
    import cry_personal as cp
    m, A, E, G, O, VAR, MODES, view = sa.load_all()
    f = np.load(FEATS)
    b = np.load(os.path.join(sa.ART, "cry_personal_basis.npz"))
    basis = {k: b[k] for k in ("mean", "std", "pca")}
    z = np.load(os.path.join(sa.ART, "cry_personal_acoustic.npz"))
    ac = {k: z[k] for k in ("x_mean", "x_std", "coef", "intercept")}
    enes = (m.source == "enes").values
    all_b = sorted(m.baby[enes].unique())
    rule_b = sorted(bb for bb, n in m.baby[enes].value_counts().items() if n >= 15)
    cfgs = {"이력만(소리 안 씀)": dict(w_sound=0.0),
            "YAMNet 16, 소리 항상": dict(gate=False),
            "음향 변환, 소리 항상": dict(acoustic=ac, gate=False),
            "음향 변환 + 자가 검증(기본)": dict(acoustic=ac)}

    def run(y, babies):
        """반환: {설정: (상위 1개 균형 정확도, 소리 켜진 비율, 상위 2개 적중률 — 답 10개 이상 / 10~19개 / 20개 이상)}.
        상위 2개 적중률은 정답 행동마다 낸 뒤 평균(균형)."""
        out = {}
        for name, kw in cfgs.items():
            yt, yp, on, hit2, nf = [], [], [], [], []
            for bb in babies:
                ix = np.where(m.baby.values == bb)[0]
                ix = ix[np.lexsort((m.bout.values[ix], m.t.values[ix]))]
                dev = cp.CryPersonal(basis, None, actions=cp.ACTIONS3, acoustic_path=kw.get("acoustic"), w_sound=kw.get("w_sound", 1.0),
                                     gate=kw.get("gate", True), hint_seconds=10)
                for i in ix:
                    t = float(m.t.values[i])
                    h = dev.hint(t, embs=[E[i, 0]])
                    if h["n_feedback"] >= 10:
                        yt.append(y[i])
                        yp.append(cp.ACTIONS3.index(h["top"]))
                        on.append(h["sound_on"])
                        hit2.append(cp.ACTIONS3[y[i]] in h["top2"])
                        nf.append(h["n_feedback"])
                    dev.start(t)
                    dev.add_window(E[i, 0])
                    dev.feedback(dev.end(t + 60), cp.ACTIONS3[y[i]])
            yt, hit2, nf = np.array(yt), np.array(hit2), np.array(nf)

            def top2(sel):
                return float(np.mean([hit2[sel & (yt == c)].mean() for c in range(3) if (sel & (yt == c)).any()]))
            out[name] = (sa.bal_np(yt, yp), float(np.mean(on)), top2(nf >= 10), top2(nf < 20), top2(nf >= 20))
        return out

    T, _ = acoustic_targets(m, A, f)
    worlds = {"요약 6": A[:, 0], "eGeMAPS": f["egemaps"].astype(float), "MFCC": f["mfcc"].astype(float), "YAMNet": E[:, 0]}
    rows, t0 = [], time.time()
    for wname, F in worlds.items():
        for seed in range(a.seeds):
            for name, r in run(baby_rule(m, F, rule_b, seed), rule_b).items():
                rows.append({"label": f"가상: {wname}", "seed": seed, "cfg": name, **dict(zip(COLS, r))})
        print(f"  {wname} {time.time() - t0:.0f}s", flush=True)
    real = np.array(["FIS".index(s) if s in ("F", "I", "S") else 0 for s in m.stop])
    for name, r in run(real, all_b).items():
        rows.append({"label": "실제 Enes", "seed": 0, "cfg": name, **dict(zip(COLS, r))})
    rng = np.random.default_rng(3)
    for r_ in range(a.perms):
        yp = real.copy()
        for bb in all_b:
            ix = np.where(m.baby.values == bb)[0]
            yp[ix] = rng.permutation(yp[ix])
        for name, r in run(yp, all_b).items():
            rows.append({"label": "실제 Enes 귀무", "seed": r_, "cfg": name, **dict(zip(COLS, r))})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(sa.RUNS, "device_replay.csv"), index=False)
    L = ["# 파이 모듈(cry_personal.py) 그대로 재생: 답 10개 이상 쌓인 뒤 힌트 균형 정확도", "",
         "맥락 없이 이력 × 소리만 비교. 칸 = 정확도 (소리가 켜진 비율). 가상 세계는 시드 평균, 실제 귀무는 라벨 섞기 95%.", "",
         "| 설정 | " + " | ".join(dict.fromkeys(D.label)) + " |", "|---|" + "---:|" * D.label.nunique()]
    for name in cfgs:
        cells = []
        for lab in dict.fromkeys(D.label):
            g = D[(D.label == lab) & (D.cfg == name)]
            v = g.bal.quantile(0.95) if lab.endswith("귀무") else g.bal.mean()
            cells.append(f"{v:.3f} ({g.on.mean():.0%})")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## 상위 2개 안에 정답이 있을 확률 (정답 행동마다 낸 뒤 평균. 아무렇게나 2개 고르면 0.667)", "",
          "칸 = 답 10개 이상 전체 (답 10~19개 / 20개 이상). 귀무는 95%.", "",
          "| 설정 | " + " | ".join(dict.fromkeys(D.label)) + " |", "|---|" + "---:|" * D.label.nunique()]
    for name in cfgs:
        cells = []
        for lab in dict.fromkeys(D.label):
            g = D[(D.label == lab) & (D.cfg == name)]
            q = (lambda c: g[c].quantile(0.95)) if lab.endswith("귀무") else (lambda c: g[c].mean())
            cells.append(f"{q('top2'):.3f} ({q('top2_10'):.2f} / {q('top2_20'):.2f})")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(sa.RUNS, "DEVICE_REPLAY.md"), "w", encoding="utf-8").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("feats")
    p.add_argument("--workers", type=int, default=4)
    p = sp.add_parser("run")
    p.add_argument("--seeds", type=int, default=5)
    sp.add_parser("export")
    p = sp.add_parser("device")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--perms", type=int, default=10)
    a = ap.parse_args()
    {"feats": cmd_feats, "run": cmd_run, "export": cmd_export, "device": cmd_device}[a.cmd](a)


if __name__ == "__main__":
    main()
