"""가상 행동 라벨 시뮬레이션: 접근 가능한 울음을 모두 모아 '행동'(수유 / 기저귀·자세 / 안아주기) 라벨을 규칙으로 붙이고,
일반 모델의 학습·검증·테스트와 파이 피드백 개인화(../cry_personal.py) 루프를 시험한다(로컬, CPU).

    python sim_synthetic.py pool    # 울음 약 3,900개: 파이식 YAMNet 임베딩 + 음향 요약(음높이·크기·끊김·음색)
    python sim_synthetic.py run     # 가상 세계 3종 × 라벨 잡음 → 결과 DATA_ROOT/runs/synthetic/SYNTHETIC_REPORT.md

이건 실제 원인과 무관한 '기능 시험'이다. 라벨은 YAMNet이 아닌 별도 음향 요약으로 정하고(순환 방지),
모델은 파이와 같은 YAMNet 임베딩만 본다.
  ① 신호 없음      : 행동을 무작위로 붙인다 → 시스템이 없는 패턴을 배운 척하지 않는지
  ② 공통 규칙      : 전체 울음을 음향으로 3유형으로 나누고 유형마다 같은 행동 → 일반 모델이 배우는지
  ③ 아기마다 다른 규칙: 아기 안에서 3유형으로 나누고 유형→행동 짝을 아기마다 다르게 → 일반 모델은 실패하고
                     피드백 개인화만 배우는지(실제 Enes 결과와 닮은 상황), 피드백 몇 개면 되는지
라벨 잡음 ε: 규칙 대신 무작위 행동이 붙을 비율(0 / 0.3 / 0.6) — 신호 세기 조절.
"""
import argparse
import collections
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cry_personal as cp  # noqa: E402

DATA = os.environ.get("NUNI_CRY_DATA", "D:/datasets/cry")
OUT = os.path.join(DATA, "synthetic")
RUNS = os.path.join(DATA, "runs", "synthetic")
ART = os.path.join(HERE, "artifacts")
SR, WIN, HOP, CRY_THR = 16000, 15600, 16000, 0.3
ACOUSTIC = ["f0_med", "f0_iqr", "loud_db", "loud_cv", "active", "centroid"]


# --- 1. 울음 모으기 ------------------------------------------------------------------
def pool_items():
    items = []
    e = pd.read_csv(os.path.join(DATA, "enes3", "meta.csv"), dtype={"date": str, "time": str})
    dt = (e.date.str.zfill(8) + e.time.str.zfill(4)).where(e.date.str.len() <= 8, e.date + e.time)
    t = pd.to_datetime(dt.str[:12], format="%d%m%Y%H%M", errors="coerce")
    for r, tt in zip(e.itertuples(), t):
        items.append({"path": os.path.join(DATA, "enes3", "wav", r.id + ".wav"), "source": "enes", "group": f"enes:{r.baby}",
                      "baby": r.baby, "t": time.mktime(tt.timetuple()) + r.t0, "dur_hint": r.dur})
    j = pd.read_csv(os.path.join(DATA, "joint", "meta.csv"))
    for r in j[j.dataset != "enes"].itertuples():                 # Enes는 위의 조각 단위로만
        items.append({"path": os.path.join(DATA, "joint", r.path), "source": r.dataset,
                      "group": f"{r.dataset}:{r.group}", "baby": r.group if r.dataset == "corvin" else "", "t": np.nan})
    u = pd.read_csv(os.path.join(DATA, "joint", "meta_unlab.csv"))
    for r in u[u.dataset != "enes_other"].itertuples():
        items.append({"path": os.path.join(DATA, "joint", r.path), "source": r.dataset, "group": f"{r.dataset}:{r.group}",
                      "baby": "", "t": np.nan})
    return items


def acoustic(path):
    """음높이(YIN)·크기·끊김(소리 나는 비율)·음색(스펙트럼 중심) 요약. YAMNet과 무관한 특징."""
    import librosa
    x, _ = sf.read(path, dtype="float32")
    x = x if x.ndim == 1 else x.mean(1)
    if len(x) < SR:
        x = np.pad(x, (0, SR - len(x)))
    R, F, Cn = [], [], []
    for s0 in range(0, len(x), 30 * SR):                 # 긴 울음은 30초씩(메모리)
        seg = x[s0:s0 + 30 * SR]
        if len(seg) < 2048:
            continue
        r = librosa.feature.rms(y=seg, frame_length=1024, hop_length=256)[0]
        f = librosa.yin(seg, fmin=150, fmax=1000, sr=SR, frame_length=1024, hop_length=256)
        c = librosa.feature.spectral_centroid(y=seg, sr=SR, n_fft=1024, hop_length=256)[0]
        k = min(len(r), len(f), len(c))
        R.append(r[:k]), F.append(f[:k]), Cn.append(c[:k])
    rms, f0, cen = np.concatenate(R), np.concatenate(F), np.concatenate(Cn)
    db = 20 * np.log10(rms + 1e-9)
    act = db > db.max() - 30
    n = len(act)
    a = act if act.any() else np.ones(n, bool)
    semi = 12 * np.log2(f0[a] / 100)
    return [float(np.median(semi)), float(np.subtract(*np.percentile(semi, [75, 25]))), float(np.mean(db[a])),
            float(np.std(rms[a]) / (np.mean(rms[a]) + 1e-9)), float(act.mean()), float(np.log(np.median(cen[a]) + 1))]


def cmd_pool(a):
    from multiprocessing import Pool
    from sim_personal import Yam, device_windows
    items = pool_items()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        A = pool.map(acoustic, [it["path"] for it in items], chunksize=8)
    print(f"음향 요약 {len(A)}개 {time.time() - t0:.0f}s", flush=True)
    yam = Yam()
    E_all, E_first, gated = [], [], []
    for i, it in enumerate(items):
        x, _ = sf.read(it["path"], dtype="float32")
        x = x if x.ndim == 1 else x.mean(1)
        embs, cries = [], []
        for w in device_windows(x):
            e, s = yam(w)
            embs.append(e)
            cries.append(max(s[19], s[20]))
        embs, cries = np.array(embs), np.array(cries)
        g = cries >= CRY_THR
        use = embs[g] if g.any() else embs
        E_all.append(use.mean(0))
        E_first.append(use[:10].mean(0))
        gated.append(float(g.mean()))
        if (i + 1) % 500 == 0:
            print(f"  임베딩 {i + 1}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    m = pd.DataFrame(items)
    m["cry_frac"] = gated
    m[ACOUSTIC] = np.array(A)
    m.to_csv(os.path.join(OUT, "pool.csv"), index=False)
    np.savez(os.path.join(OUT, "pool.npz"), emb=np.array(E_all, np.float32), emb10=np.array(E_first, np.float32))
    print(f"울음 {len(m)}개 저장: {m.source.value_counts().to_dict()}")


# --- 2. 가상 라벨 ---------------------------------------------------------------------
def kmeans_types(Z, seed):
    from sklearn.cluster import KMeans
    return KMeans(3, n_init=10, random_state=seed).fit_predict(Z)


def make_labels(m, world, eps, seed):
    """world: none / global / baby. 반환: 행동 번호(0 수유, 1 기저귀·자세, 2 안아주기)."""
    rng = np.random.default_rng(seed)
    A = m[ACOUSTIC].values
    y = rng.integers(0, 3, len(m))
    if world == "none":
        return y
    if world == "global":
        Z = np.zeros_like(A)                                   # 출처(녹음 환경)마다 기준을 맞춘 뒤 나눈다.
        for src, idx in m.groupby("source").indices.items():   # 그러지 않으면 데이터셋 차이로 유형이 갈린다
            Z[idx] = (A[idx] - A[idx].mean(0)) / (A[idx].std(0) + 1e-9)
        rule = rng.permutation(3)[kmeans_types(Z, seed)]
    else:
        rule = np.full(len(m), -1)
        for b, idx in m.groupby("baby").indices.items():
            if not b or len(idx) < 15:
                continue
            Zb = (A[idx] - A[idx].mean(0)) / (A[idx].std(0) + 1e-9)
            rule[idx] = rng.permutation(3)[kmeans_types(Zb, seed)]            # 아기마다 다른 짝
    keep = (rng.random(len(m)) >= eps) & (rule >= 0)
    y[keep] = rule[keep]
    return y


def bout_level(m, E, E10):
    """Enes 조각(5초 공백 기준)을 원래 울음 하나로 합친다. 부모는 울음 한 번에 한 번 답하므로 피드백 단위는 울음이다.
    (조각 단위로 넣으면 같은 울음 조각끼리 라벨이 같아 '직전 답 따라 하기'가 부풀려졌다.)"""
    en = m.source == "enes"
    other = m[~en].copy()
    e = m[en].copy()
    e["bout_id"] = e.path.str.extract(r"(b\d{4})_e\d{3}\.wav$", expand=False)
    e["w"] = e.dur_hint.fillna(1.0).clip(lower=0.1)
    rows, Eb, E10b = [], [], []
    for (b, bid), g in e.groupby(["baby", "bout_id"], sort=False):
        g = g.sort_values("t")
        w = g.w.values / g.w.sum()
        rows.append({"path": g.path.iloc[0], "source": "enes", "group": g.group.iloc[0], "baby": b,
                     "t": g.t.iloc[0], "dur_hint": float(g.dur_hint.sum()), "cry_frac": float((g.cry_frac * w).sum()),
                     **{c: float((g[c] * w).sum()) for c in ACOUSTIC}})
        Eb.append((E[g.index] * w[:, None]).sum(0))
        E10b.append(E10[g.index[0]])
    M = pd.concat([other, pd.DataFrame(rows)], ignore_index=True)
    E2 = np.concatenate([E[other.index], np.array(Eb)])
    E10_2 = np.concatenate([E10[other.index], np.array(E10b)])
    return M, E2.astype(np.float32), E10_2.astype(np.float32)


# --- 3. 일반 모델: 학습·검증·테스트(그룹 단위 분할) ------------------------------------------------
def general_model(E, y, groups, seed):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(seed)
    ug = rng.permutation(np.unique(groups))
    n = len(ug)
    tr_g, va_g = set(ug[: int(n * 0.6)]), set(ug[int(n * 0.6): int(n * 0.8)])
    tr = np.array([g in tr_g for g in groups])
    va = np.array([g in va_g for g in groups])
    te = ~(tr | va)
    mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-6
    pca = PCA(32, random_state=seed).fit((E[tr] - mu) / sd)
    Z = pca.transform((E - mu) / sd)
    best = max((0.01, 0.1, 1.0), key=lambda C: balanced_accuracy_score(
        y[va], LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Z[tr], y[tr]).predict(Z[va])))
    clf = LogisticRegression(C=best, max_iter=2000, class_weight="balanced").fit(Z[tr | va], y[tr | va])
    return balanced_accuracy_score(y[te], clf.predict(Z[te])), int(tr.sum()), int(va.sum()), int(te.sum())


# --- 4. 파이 피드백 개인화 재생(아기별 시간순) ----------------------------------------------------
BUCKETS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 10 ** 6)]


def replay(m, E, E10, y, basis, p_answer, noise, seed, sound_method="proto"):
    rng = np.random.default_rng(seed)
    rec = []
    babies = m[(m.source == "enes")].baby.unique()
    for b in babies:
        idx = m.index[(m.source == "enes") & (m.baby == b)].values
        idx = idx[np.argsort(m.t.values[idx], kind="stable")]
        dev = cp.CryPersonal(basis, None, k_min=10, w_sound=1.0, half_life=8, hint_seconds=10,
                             sound_method=sound_method)
        for i in idx:
            t = float(m.t.values[i])
            h = dev.hint(t, embs=[E10[i]])
            P = {k: np.array(v) for k, v in h["parts"].items()}
            rec.append({"baby": b, "nfb": h["n_feedback"], "y": y[i],
                        "dev": int(np.argmax(P["history"] * P.get("sound", 1))),
                        "hist": int(np.argmax(P["history"])),
                        "sound": int(np.argmax(P["sound"])) if "sound" in P else -1})
            dev.start(t)
            dev.add_window(E[i])
            eid = dev.end(t + float(m.dur_hint.values[i]))
            if eid is not None and rng.random() < p_answer:
                ans = y[i] if rng.random() >= noise else rng.choice([c for c in range(3) if c != y[i]])
                dev.feedback(eid, cp.ACTIONS[ans])
    return pd.DataFrame(rec)


def bucket_scores(D):
    from sklearn.metrics import balanced_accuracy_score
    out = {}
    for lo, hi in BUCKETS:
        s = D[(D.nfb >= lo) & (D.nfb < hi)]
        if len(s) < 20:
            continue
        out[(lo, hi)] = {k: balanced_accuracy_score(s.y, s[k]) for k in ("dev", "hist")}
        ss = s[s.sound >= 0]
        out[(lo, hi)]["sound"] = balanced_accuracy_score(ss.y, ss.sound) if len(ss) >= 20 else np.nan
        out[(lo, hi)]["n"] = len(s)
    return out


def cmd_run(a):
    m = pd.read_csv(os.path.join(OUT, "pool.csv"))
    z = np.load(os.path.join(OUT, "pool.npz"))
    m, E, E10 = bout_level(m, z["emb"], z["emb10"])
    b = np.load(os.path.join(ART, "cry_personal_basis.npz"))
    basis = {"mean": b["mean"], "std": b["std"], "pca": b["pca"]}
    enes = (m.source == "enes").values
    rows, curves = [], []
    worlds = [("none", 1.0)] + [(w, e) for w in ("global", "baby") for e in (0.0, 0.3, 0.6)]
    t0 = time.time()
    for world, eps in worlds:
        for seed in range(a.seeds):
            y = make_labels(m, world, eps, seed)
            # 일반 모델: 공통 규칙·신호 없음은 전체 울음, 아기별 규칙은 Enes(아기 정보가 있는 곳)
            sel = np.ones(len(m), bool) if world != "baby" else enes
            bal, ntr, nva, nte = general_model(E[sel], y[sel], m.group.values[sel], seed)
            rows.append({"world": world, "eps": eps, "seed": seed, "general_test": bal, "n_train": ntr, "n_val": nva,
                         "n_test": nte})
            for sm in ("proto", "knn"):
                for pa, nz in ((1.0, 0.0), (0.5, 0.0), (0.25, 0.0), (1.0, 0.2)):
                    D = replay(m, E, E10, y, basis, pa, nz, seed, sound_method=sm)
                    for (lo, hi), sc in bucket_scores(D).items():
                        curves.append({"world": world, "eps": eps, "seed": seed, "answer": pa, "noise": nz, "method": sm,
                                       "bucket": f"{lo}~{hi - 1}" if hi < 10 ** 6 else f"{lo}+", **sc})
        print(f"  {world} ε={eps} 완료 {time.time() - t0:.0f}s", flush=True)
    R, C = pd.DataFrame(rows), pd.DataFrame(curves)
    os.makedirs(RUNS, exist_ok=True)
    R.to_csv(os.path.join(RUNS, "general.csv"), index=False)
    C.to_csv(os.path.join(RUNS, "feedback.csv"), index=False)
    report(m, R, C)


def report(m, R, C):
    name = {"none": "① 신호 없음", "global": "② 공통 규칙", "baby": "③ 아기마다 다른 규칙"}
    L = ["# 가상 행동 라벨 시뮬레이션 (기능 시험 — 실제 원인과 무관)", "",
         f"- 울음 {len(m)}개: " + ", ".join(f"{k} {v}" for k, v in m.source.value_counts().items()),
         "- 라벨: 행동 3개(수유 / 기저귀·자세 / 안아주기). 규칙은 YAMNet이 아닌 음향 요약(음높이·크기·끊김·음색) k-평균 3유형.",
         "- 모델 입력은 파이와 같은 YAMNet 임베딩(0.975초 창, 1초 간격, 울음 판정 창 평균). 균형 정확도 무작위 0.333.",
         "- 일반 모델: 그룹(아기·업로더) 단위로 학습 60% / 검증 20%(규제 선택) / 테스트 20%. 시드 평균.",
         "- 피드백 개인화: Enes 아기 24명의 울음 612개(조각을 울음 단위로 합침)를 아기별 시간순으로 파이 모듈에 넣음(이전 답 10개부터 소리 사용).", "",
         "## 1. 일반 모델 테스트 (균형 정확도)", "", "| 가상 세계 | 라벨 잡음 0% | 30% | 60% |", "|---|---:|---:|---:|"]
    for w in ("none", "global", "baby"):
        g = R[R.world == w].groupby("eps").general_test.mean()
        cells = [f"{g.get(e, np.nan):.3f}" if e in g else "-" for e in (0.0, 0.3, 0.6)]
        if w == "none":
            cells = [f"{g.iloc[0]:.3f} (전부 무작위)", "-", "-"]
        L.append(f"| {name[w]} | " + " | ".join(cells) + " |")
    L += ["", "## 2. 피드백 개인화: 쌓인 답 수별 균형 정확도 (부모가 매번 정확히 답함)", "",
          "이력 = 이전 답 빈도만(소리 없음). 소리 비교: proto = 원인별 평균과 코사인, knn = 가장 비슷한 이전 울음 3개 투표.", ""]
    base = C[(C.answer == 1.0) & (C.noise == 0.0)]
    buckets = list(dict.fromkeys(base.bucket))
    L += ["| 가상 세계 | 라벨 잡음 | 비교 | " + " | ".join(f"답 {b}개" for b in buckets) + " |",
          "|---|---|---|" + "---:|" * len(buckets)]
    for (w, e), g in base.groupby(["world", "eps"], sort=False):
        rows_ = [("이력만", g[g.method == "proto"], "hist")]
        rows_ += [(f"이력×소리({sm})", g[g.method == sm], "dev") for sm in ("proto", "knn")]
        for lab, gg, k in rows_:
            v = gg.groupby("bucket")[k].mean()
            L.append(f"| {name[w]} | {e:.0%} | {lab} | " + " | ".join(f"{v.get(b, np.nan):.3f}" for b in buckets) + " |")
    L += ["", "## 3. 부모 답이 불완전할 때 (③ 아기마다 다른 규칙, 라벨 잡음 0%, 답 10개 이상 쌓인 뒤)", "",
          "| 부모 답 비율 | 잘못 누름 | 이력×소리(knn) | 이력×소리(proto) | 이력만 |", "|---|---|---:|---:|---:|"]
    g3 = C[(C.world == "baby") & (C.eps == 0.0) & C.bucket.isin(["10~14", "15~19", "20+"])]
    for (pa, nz), g in g3.groupby(["answer", "noise"]):
        L.append(f"| {pa:.0%} | {nz:.0%} | {g[g.method == 'knn']['dev'].mean():.3f} | "
                 f"{g[g.method == 'proto']['dev'].mean():.3f} | {g[g.method == 'proto']['hist'].mean():.3f} |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(RUNS, "SYNTHETIC_REPORT.md"), "w", encoding="utf-8").write(txt)
    print(txt)
    plot(C, buckets, name)


def plot(C, buckets, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    base = C[(C.answer == 1.0) & (C.noise == 0.0)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    styles = {("none", 1.0): ("#9e9e9e", "-"), ("global", 0.3): ("#1f77b4", "-"), ("baby", 0.3): ("#d62728", "-"),
              ("baby", 0.0): ("#d62728", ":"), ("baby", 0.6): ("#d62728", "--")}
    for (w, e), (col, ls) in styles.items():
        g = base[(base.world == w) & (base.eps == e) & (base.method == "knn")].groupby("bucket").dev.mean().reindex(buckets)
        lab = f"{name[w]}" + ("" if w == "none" else f", 라벨 잡음 {e:.0%}")
        ax.plot(range(len(buckets)), g.values, ls, color=col, marker="o", label=lab)
    ax.axhline(1 / 3, color="k", lw=0.8, alpha=0.4)
    ax.text(len(buckets) - 1, 1 / 3 + 0.01, "무작위 0.333", ha="right", fontsize=8, alpha=0.6)
    ax.set_xticks(range(len(buckets)), [f"답 {b}개" for b in buckets])
    ax.set_ylabel("균형 정확도")
    ax.set_title("피드백이 쌓일수록 파이 개인화 힌트 정확도 (가상 라벨, 이력×소리 knn)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RUNS, "synthetic_learning_curve.png"), dpi=150)
    print("그림:", os.path.join(RUNS, "synthetic_learning_curve.png"))


def cmd_report(_):
    m = pd.read_csv(os.path.join(OUT, "pool.csv"))
    z = np.load(os.path.join(OUT, "pool.npz"))
    m, _, _ = bout_level(m, z["emb"], z["emb10"])
    report(m, pd.read_csv(os.path.join(RUNS, "general.csv")), pd.read_csv(os.path.join(RUNS, "feedback.csv")))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("report")
    p = sp.add_parser("pool")
    p.add_argument("--workers", type=int, default=3)
    p = sp.add_parser("run")
    p.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    {"pool": cmd_pool, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
