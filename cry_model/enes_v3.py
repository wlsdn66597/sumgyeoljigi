"""울음 원인 3차 점검: Enes 데이터 구성을 고쳐서 다시 본다(build·yamnet은 로컬, extract·eval은 GPU 서버).

    python enes_v3.py build                  # 로컬: 음절을 무음 없이 이어 붙인 울음 조각 wav + meta.csv
    python enes_v3.py yamnet                 # 로컬: YAMNet 창 임베딩 → feats/yamnet.npz
    python enes_v3.py extract --model clap   # 서버: 대형 모델 창 임베딩 → feats/<model>.npz
    python enes_v3.py smile --set compare    # 서버: openSMILE 음향 특징(ComParE 6,373개 / eGeMAPS 88개) → feats/<set>.npz
    python enes_v3.py eval --model clap      # 서버: 아기 분리(반복·귀무·나이별) + 아기별 시간순 개인화
    python enes_v3.py report                 # 서버: 모든 결과 → ENES_V3_REPORT.md

2차 점검에서 찾은 문제를 고쳤다
  - 음절 사이 공백을 디지털 무음(0)으로 채워 소리의 53%가 무음이었다
    → 음절 파일(앞뒤 50ms 실제 배경 포함)을 그대로 이어 붙인다. 공백 길이는 meta의 리듬 값으로 따로 둔다.
  - 60초 상한(공백 포함) 때문에 612개 중 367개가 앞부분만 쓰였다 → 상한 없이 울음 전체를 쓴다.
  - 울음 묶음(612개)만 봤다 → 5초 넘는 공백에서 나눈 조각(약 2,000개, Bonafos 2025와 같은 단위)도 본다.
  - 라벨: 멈춘 행동(논문 기준) / 부모 판단 / 둘이 일치한 것
  - 평가: 아기 분리 + 녹음 회차(나이)별 + 아기별 시간순 개인화(그 아기의 이전 울음 라벨을 쓰는 제품 상황)

데이터 위치: CRY_ENES3(기본 로컬 DATA_ROOT/enes3, 서버 ~/cry/data/enes3)
"""
import argparse
import collections
import glob
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
CAUSES = ["F", "I", "S"]                 # 배고픔, 불편, 혼자 둠(안아줘서 멈춤)
NAMES = {"F": "hunger", "I": "discomfort", "S": "isolation"}
SPLIT_GAP_MS = 5000                      # 이보다 긴 공백에서 울음을 조각으로 나눈다
WIN, HOP = 4.0, 2.0                      # 창 임베딩
LAST_WINDOWS = 14                        # 울음 끝부분 약 30초(4초 창, 2초 간격)
SEED = 42


def root():
    d = os.environ.get("CRY_ENES3")
    if d:
        return os.path.expanduser(d)
    try:
        import config
        return os.path.join(config.DATA_ROOT, "enes3")
    except Exception:
        return os.path.expanduser("~/cry/data/enes3")


# --- 로컬: 데이터 만들기 -----------------------------------------------------------
def cmd_build(_):
    import librosa
    import config
    src = os.path.join(config.DATA_ROOT, "research", "enes", "00_pooled_separate")
    out = root()
    os.makedirs(os.path.join(out, "wav"), exist_ok=True)
    groups = collections.defaultdict(list)
    for e in os.scandir(src):
        p = e.name[:-4].split("_")
        if len(p) != 8 or "-" not in p[7] or p[4][:1] not in CAUSES:
            continue
        s, t = p[7].split("-")
        groups["_".join(p[:7])].append((int(s), int(t), e.name))
    rows = []
    for b, (key, items) in enumerate(sorted(groups.items())):
        items.sort()
        p = key.split("_")
        eps, cur = [], [items[0]]
        for prev, it in zip(items, items[1:]):
            if it[0] - prev[1] > SPLIT_GAP_MS:
                eps.append(cur)
                cur = []
            cur.append(it)
        eps.append(cur)
        for k, ep in enumerate(eps):
            cid = f"b{b:04d}_e{k:03d}"
            path = os.path.join(out, "wav", cid + ".wav")
            if not os.path.exists(path):
                parts, sr = [], None
                for _, _, n in ep:
                    x, sr = sf.read(os.path.join(src, n), dtype="float32")
                    parts.append(x.mean(axis=1) if x.ndim > 1 else x)
                w = librosa.resample(np.concatenate(parts), orig_sr=sr, target_sr=SR)
                sf.write(path, (np.clip(w, -1, 1) * 32767).astype(np.int16), SR, subtype="PCM_16")
            st = np.array([s for s, _, _ in ep], float)
            en = np.array([t for _, t, _ in ep], float)
            gaps = np.clip(st[1:] - en[:-1], 0, None) / 1000 if len(ep) > 1 else np.zeros(1)
            rows.append({"id": cid, "bout": b, "ep": k, "key": key, "baby": p[0], "age": int(p[2]),
                         "stop": p[4][:1], "parent": p[3][:1], "date": p[5], "time": p[6],
                         "t0": float(st.min()) / 1000, "dur": sf.info(path).duration, "n_syl": len(ep),
                         "gap_med": float(np.median(gaps)), "voiced": float((en - st).sum()) / 1000})
        if (b + 1) % 100 == 0:
            print(f"  {b + 1}/{len(groups)}", flush=True)
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(out, "meta.csv"), index=False)
    bo = m.groupby("bout").agg(stop=("stop", "first"), dur=("dur", "sum"), n_ep=("ep", "count"))
    print(f"울음 {len(bo)}개 → 조각 {len(m)}개, 소리 {m.dur.sum() / 3600:.2f}시간 (디지털 무음 없음, 길이 상한 없음)")
    print("라벨(멈춘 행동):", bo.stop.value_counts().to_dict(), "| 울음 길이 중앙값(초):", round(bo.dur.median(), 1))


def load_meta():
    m = pd.read_csv(os.path.join(root(), "meta.csv"), dtype={"date": str, "time": str})
    # 파일명 하나가 날짜·시각 경계가 한 칸 밀려 있다(101020192_058 = 10102019_2058). 붙여서 다시 자른다.
    dt = (m.date.str.zfill(8) + m.time.str.zfill(4)).where(m.date.str.len() <= 8, m.date + m.time)
    m["t"] = pd.to_datetime(dt.str[:12], format="%d%m%Y%H%M", errors="coerce")
    return m


def windows(x, n=int(WIN * SR), h=int(HOP * SR)):
    if len(x) <= n:
        return [np.tile(x, int(np.ceil(n / max(len(x), 1))))[:n]]
    st = list(range(0, len(x) - n + 1, h))
    if st[-1] + n < len(x):
        st.append(len(x) - n)
    return [x[s:s + n] for s in st]


# --- 로컬: YAMNet 창 임베딩 ----------------------------------------------------------
def cmd_yamnet(_):
    import features
    m = load_meta()
    X, ep_idx = [], []
    for i, r in enumerate(m.itertuples()):
        x, _ = sf.read(os.path.join(root(), "wav", r.id + ".wav"), dtype="float32")
        for w in windows(x):
            _, emb = features.frames(w)
            X.append(emb.mean(0)[None].astype(np.float16))
            ep_idx.append(i)
        if (i + 1) % 300 == 0:
            print(f"  yamnet {i + 1}/{len(m)}", flush=True)
    os.makedirs(os.path.join(root(), "feats"), exist_ok=True)
    np.savez(os.path.join(root(), "feats", "yamnet.npz"), ids=m.id.values, X=np.stack(X), ep=np.array(ep_idx))
    print("yamnet 창", len(X))


# --- 서버: 대형 모델 창 임베딩 -------------------------------------------------------
def cmd_extract(a):
    import torch
    import joint_train as jt
    m = load_meta()
    out = os.path.join(root(), "feats", f"{a.model}.npz")
    if os.path.exists(out):
        print("이미 있음:", out)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    emb, _ = jt._embedder(a.model)
    X, ep_idx, t0 = [], [], time.time()
    with torch.no_grad():
        for i, r in enumerate(m.itertuples()):
            x, _ = sf.read(os.path.join(root(), "wav", r.id + ".wav"), dtype="float32")
            for w in windows(x):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    X.append(emb(w).float().cpu().numpy().astype(np.float16))
                ep_idx.append(i)
            if (i + 1) % 300 == 0:
                print(f"  {a.model} {i + 1}/{len(m)} {time.time() - t0:.0f}s", flush=True)
    np.savez(out, ids=m.id.values, X=np.stack(X), ep=np.array(ep_idx))
    print(f"{a.model}: 창 {len(X)}개 {np.stack(X).shape}")


# --- 서버: openSMILE 음향 특징 ---------------------------------------------------------
SMILE_SETS = {"compare": "ComParE_2016", "egemaps": "eGeMAPSv02"}
_SMILE = None


def _smile_init(name):
    global _SMILE
    import opensmile
    _SMILE = opensmile.Smile(feature_set=getattr(opensmile.FeatureSet, name),
                             feature_level=opensmile.FeatureLevel.Functionals)


def _smile_episode(args):
    """조각 하나를 4초 창으로 나눠 창마다 openSMILE 통계 특징을 계산한다(작업자 프로세스)."""
    i, path = args
    x, _ = sf.read(path, dtype="float32")
    out = []
    for w in windows(x):
        f = _SMILE.process_signal(w, SR).values[0].astype(np.float32)
        out.append(f)
    return i, np.stack(out)


def cmd_smile(a):
    """ChatterBaby(UCLA)가 쓴 openSMILE 특징. 음높이(F0), 크기, 떨림(jitter·shimmer), 음색(스펙트럼·MFCC) 통계.
    딥러닝 임베딩과 달리 사람이 설계한 목소리 특징이라, 아기별 울음 차이를 다르게 잡을 수 있다."""
    from multiprocessing import Pool
    m = load_meta()
    out = os.path.join(root(), "feats", f"{a.set}.npz")
    if os.path.exists(out):
        print("이미 있음:", out)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    jobs = [(i, os.path.join(root(), "wav", r.id + ".wav")) for i, r in enumerate(m.itertuples())]
    res, t0 = {}, time.time()
    with Pool(a.workers, initializer=_smile_init, initargs=(SMILE_SETS[a.set],)) as pool:
        for k, (i, F) in enumerate(pool.imap_unordered(_smile_episode, jobs, chunksize=4)):
            res[i] = F
            if (k + 1) % 200 == 0:
                print(f"  {a.set} {k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    X = np.concatenate([res[i] for i in range(len(m))])
    ep = np.concatenate([np.full(len(res[i]), i) for i in range(len(m))])
    bad = ~np.isfinite(X)
    X[bad] = 0
    np.savez(out, ids=m.id.values, X=X[:, None, :].astype(np.float32), ep=ep)
    print(f"{a.set}: 창 {len(X)}개 × 특징 {X.shape[1]}개, 비정상 값 {int(bad.sum())}개를 0으로 ({time.time() - t0:.0f}s)")


# --- 서버: 평가 ------------------------------------------------------------------------
def unit_feats(m, model):
    """창 임베딩 → 단위별 평균. ep: 조각, bout: 울음 전체, last: 울음 끝 약 30초."""
    z = np.load(os.path.join(root(), "feats", f"{model}.npz"), allow_pickle=True)
    assert (z["ids"] == m.id.values).all()
    Xw, ep = z["X"], z["ep"]                      # 한 번만 꺼낸다(npz는 접근마다 다시 읽음)
    bout = m.bout.values[ep]
    nb = m.bout.max() + 1
    out = {}
    for name, key, n in (("ep", ep, len(m)), ("bout", bout, nb)):
        acc = np.zeros((n,) + Xw.shape[1:], np.float32)
        cnt = np.zeros(n)
        for s in range(0, len(Xw), 4000):
            np.add.at(acc, key[s:s + 4000], Xw[s:s + 4000].astype(np.float32))
            np.add.at(cnt, key[s:s + 4000], 1)
        out[name] = acc / np.maximum(cnt, 1)[:, None, None]
    last = np.zeros((nb,) + Xw.shape[1:], np.float32)
    for b in range(nb):
        idx = np.where(bout == b)[0][-LAST_WINDOWS:]
        last[b] = Xw[idx].astype(np.float32).mean(0)
    out["last"] = last
    return out


def bout_table(m):
    b = m.groupby("bout").agg(baby=("baby", "first"), age=("age", "first"), stop=("stop", "first"),
                              parent=("parent", "first"), t=("t", "first"), key=("key", "first")).reset_index()
    b["agree"] = np.where(b.stop == b.parent, b.stop, "")
    return b


def metrics(y, P):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    pred = P.argmax(1)
    return {"bal": balanced_accuracy_score(y, pred),
            "f1": f1_score(y, pred, average="macro", labels=range(3), zero_division=0),
            "auc": roc_auc_score(y == 0, P[:, 0])}


def probe_cv(X, y, groups, folds, k_folds=5):
    """층 가중합 선형 프로브(joint_train.fit_probe), 아기 단위 겹으로 예측."""
    import joint_train as jt
    P = np.zeros((len(y), 3))
    cw = len(y) / (3 * np.bincount(y, minlength=3).clip(1))
    for k in range(k_folds):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        f = jt.fit_probe(X[tr], y[tr], np.ones((len(tr), 3), bool), cw[y[tr]], n_out=3)
        import torch
        P[te] = torch.softmax(f(X[te]), -1).cpu().numpy()
    return P


def make_folds(y, groups, seed):
    from sklearn.model_selection import StratifiedGroupKFold
    f = np.zeros(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, y, groups)):
        f[te] = k
    return f


def cmd_eval(a):
    import joint_train as jt
    jt.mem_guard()
    m = load_meta()
    B = bout_table(m)
    U = unit_feats(m, a.model)
    out_dir = os.path.join(root(), "runs", a.model)
    os.makedirs(out_dir, exist_ok=True)
    rows, P0 = [], {}
    # A. 아기 분리: 라벨 3종 × 단위 3종, 반복 5회 + 라벨 섞기 5회, 녹음 회차(나이)별
    for lab in ("stop", "parent", "agree"):
        bsel = np.where(B[lab].isin(CAUSES))[0]
        yb = B[lab].values[bsel]
        yb = np.array([CAUSES.index(c) for c in yb])
        gb = B.baby.values[bsel]
        for unit in ("bout", "last", "ep"):
            for kind, R in (("repeat", a.repeats), ("perm", a.perms)):
                for r in range(R):
                    rng = np.random.default_rng(1000 + r)
                    y = yb.copy() if kind == "repeat" else rng.permutation(yb)
                    folds = make_folds(y, gb, SEED + 1 + r)
                    if unit == "ep":                  # 조각으로 학습·예측 → 울음 단위로 평균
                        eb = np.where(np.isin(m.bout.values, B.bout.values[bsel]))[0]
                        pos = {b: i for i, b in enumerate(B.bout.values[bsel])}
                        bi = np.array([pos[b] for b in m.bout.values[eb]])
                        Pe = probe_cv(U["ep"][eb], y[bi], gb[bi], folds[bi])
                        P = np.zeros((len(bsel), 3))
                        np.add.at(P, bi, Pe)
                        P /= np.bincount(bi, minlength=len(bsel))[:, None]
                    else:
                        P = probe_cv(U[unit][B.bout.values[bsel]], y, gb, folds)
                    rows.append({"model": a.model, "label": lab, "unit": unit, "kind": kind, "r": r,
                                 "age": "all", "n": len(y), **metrics(y, P)})
                    for age in sorted(B.age.unique()):
                        s = B.age.values[bsel] == age
                        if len(np.unique(y[s])) == 3:
                            rows.append({"model": a.model, "label": lab, "unit": unit, "kind": kind, "r": r,
                                         "age": int(age), "n": int(s.sum()), **metrics(y[s], P[s])})
                    if kind == "repeat" and r == 0:
                        P0[(lab, unit)] = (bsel, P)
            print(f"  {a.model} {lab}/{unit} 완료", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "babysplit.csv"), index=False)

    # B. 아기별 시간순 개인화(라벨 = 멈춘 행동, 울음 전체 단위)
    pers = personalize(B, U["bout"], P0[("stop", "bout")], a)
    pers.to_csv(os.path.join(out_dir, "personal.csv"), index=False)
    g = df[df.age == "all"].groupby(["label", "unit", "kind"])[["bal", "auc"]].mean().unstack("kind").round(3)
    print(g.to_string())
    print(pers.groupby(["method", "kind"])[["bal", "auc"]].mean().unstack("kind").round(3).to_string())


def personalize(B, Xb, P_global, a, k_min=10):
    """각 아기의 울음을 시간순으로 보며, 그 시점까지 부모가 알려 준 원인(이전 라벨)만 쓴다.
    - prior: 그 아기의 이전 원인 빈도(라플라스)
    - global: 다른 아기들로 학습한 소리 모델(아기 분리 예측)
    - proto: 그 아기의 이전 울음 소리와의 유사도(이전 울음 평균을 빼고 원인별 평균과 코사인)
    귀무: 아기 안에서 라벨을 섞는다(이전 원인 빈도는 그대로라 prior 기준은 같다)."""
    from sklearn.decomposition import PCA
    bsel, Pg_sel = P_global
    sel = np.where(B.stop.isin(CAUSES))[0]
    assert (sel == bsel).all()
    y_all = np.array([CAUSES.index(c) for c in B.stop.values[sel]])
    Pg = Pg_sel
    X = Xb[B.bout.values[sel]].reshape(len(sel), -1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Z = PCA(64, random_state=SEED).fit_transform(X)
    babies = B.baby.values[sel]
    order = {b: sorted(np.where(babies == b)[0], key=lambda i: (B.t.values[sel][i], B.key.values[sel][i]))
             for b in np.unique(babies)}
    rows = []

    def run(y, kind, r):
        res = collections.defaultdict(list)
        for b, idx in order.items():
            for k, i in enumerate(idx):
                if k < k_min:
                    continue
                past = idx[:k]
                cnt = np.bincount(y[past], minlength=3)
                prior = (cnt + 1) / (cnt.sum() + 3)
                zc = Z[i] - Z[past].mean(0)
                zp = Z[past] - Z[past].mean(0)
                sims = np.full(3, -1.0)
                for c in range(3):
                    if cnt[c]:
                        pc = zp[y[past] == c].mean(0)
                        sims[c] = zc @ pc / (np.linalg.norm(zc) * np.linalg.norm(pc) + 1e-9)
                proto = np.exp(5 * sims)
                proto /= proto.sum()
                for name, p in (("prior", prior), ("global", Pg[i]), ("proto", proto),
                                ("prior×global", prior * Pg[i]), ("prior×proto", prior * proto),
                                ("prior×global×proto", prior * Pg[i] * proto)):
                    res[name].append(p / p.sum())
                res["_y"].append(y[i])
        yy = np.array(res.pop("_y"))
        for name, Ps in res.items():
            rows.append({"method": name, "kind": kind, "r": r, "n": len(yy), **metrics(yy, np.stack(Ps))})

    run(y_all, "real", 0)
    rng = np.random.default_rng(7)
    for r in range(a.pperms):
        y = y_all.copy()
        for b, idx in order.items():
            y[idx] = rng.permutation(y[idx])
        run(y, "perm", r)
    return pd.DataFrame(rows)


def cmd_personal(a):
    """개인화 결과가 '울음 소리' 때문인지 '같은 날·같은 시간대' 때문인지 가른다.
    history=all: 이전 울음 전부 / prev_days: 같은 날 울음은 빼고 이전 날만 / prev_age: 이전 녹음 회차(2주~1개월 전)만
    timeproto: 소리 대신 녹음 시각(하루 중 시간)만으로 같은 방식의 유사도 → 이것만으로 비슷하게 나오면 시간대 효과다."""
    import joint_train as jt
    jt.mem_guard()
    from sklearn.decomposition import PCA
    m = load_meta()
    B = bout_table(m)
    sel = np.where(B.stop.isin(CAUSES))[0]
    y_all = np.array([CAUSES.index(c) for c in B.stop.values[sel]])
    babies, t = B.baby.values[sel], pd.to_datetime(B.t.values[sel])
    X = unit_feats(m, a.model)["bout"][B.bout.values[sel]].reshape(len(sel), -1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Z = PCA(64, random_state=SEED).fit_transform(X)
    hour = t.hour.values + t.minute.values / 60
    H = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)], 1)
    day = t.normalize().values
    age = B.age.values[sel]
    order = {b: sorted(np.where(babies == b)[0], key=lambda i: (t[i], B.key.values[sel][i])) for b in np.unique(babies)}

    def sim_probs(F, i, past, y, center):
        cnt = np.bincount(y[past], minlength=3)
        sims = np.full(3, -1.0)
        mu = F[past].mean(0) if center else 0
        fc = F[i] - mu
        for c in range(3):
            if cnt[c]:
                pc = (F[past][y[past] == c] - mu).mean(0)
                sims[c] = fc @ pc / (np.linalg.norm(fc) * np.linalg.norm(pc) + 1e-9)
        e = np.exp(5 * sims)
        return e / e.sum()

    rows = []

    def run(y, kind, r, history):
        res = collections.defaultdict(list)
        for b, idx in order.items():
            for k, i in enumerate(idx):
                past = [j for j in idx[:k] if history == "all" or (history == "prev_days" and day[j] < day[i])
                        or (history == "prev_age" and age[j] < age[i])]
                if len(past) < 10:
                    continue
                cnt = np.bincount(y[past], minlength=3)
                prior = (cnt + 1) / (cnt.sum() + 3)
                proto = sim_probs(Z, i, past, y, True)
                tp = sim_probs(H, i, past, y, False)
                for name, p in (("prior", prior), ("proto", proto), ("timeproto", tp), ("prior×proto", prior * proto),
                                ("prior×timeproto", prior * tp), ("prior×timeproto×proto", prior * tp * proto)):
                    res[name].append(p / p.sum())
                res["_y"].append(y[i])
        yy = np.array(res.pop("_y"))
        for name, Ps in res.items():
            rows.append({"model": a.model, "history": history, "method": name, "kind": kind, "r": r, "n": len(yy),
                         **metrics(yy, np.stack(Ps))})

    for history in ("all", "prev_days", "prev_age"):
        run(y_all, "real", 0, history)
        rng = np.random.default_rng(7)
        for r in range(a.pperms):
            y = y_all.copy()
            for b, idx in order.items():
                y[idx] = rng.permutation(y[idx])
            run(y, "perm", r, history)
    df = pd.DataFrame(rows)
    out = os.path.join(root(), "runs", a.model, "personal2.csv")
    df.to_csv(out, index=False)
    g = df.groupby(["history", "method", "kind"])[["bal", "auc"]].mean().unstack("kind").round(3)
    print(g.to_string())
    print("n:", df.groupby("history").n.first().to_dict())


def context_probs(B, sel, y):
    """맥락·리듬 특징(소리 모델 없음)으로 한 아기씩 빼고 학습한 RF 확률. 이전 울음 라벨은 그 시점 이전 것만 쓴다."""
    from sklearn.ensemble import RandomForestClassifier
    b = B.iloc[sel].reset_index(drop=True)
    t = pd.to_datetime(b.t)
    hour = t.dt.hour + t.dt.minute / 60
    F = pd.DataFrame({"hs": np.sin(2 * np.pi * hour / 24), "hc": np.cos(2 * np.pi * hour / 24), "age": b.age,
                      "n_syl": b.n_syl, "voiced": b.voiced, "n_ep": b.n_ep, "gap": b.gap_med})
    for c in ("h_prev", "h_hunger", "p0", "p1", "p2"):
        F[c] = 0.0
    for bb in b.baby.unique():
        idx = sorted(np.where(b.baby == bb)[0], key=lambda i: t[i])
        last_t, last_h, last_c = None, None, None
        for i in idx:
            F.loc[i, "h_prev"] = min((t[i] - last_t).total_seconds() / 3600, 12) if last_t is not None else 12
            F.loc[i, "h_hunger"] = min((t[i] - last_h).total_seconds() / 3600, 12) if last_h is not None else 12
            if last_c is not None:
                F.loc[i, f"p{last_c}"] = 1
            last_t, last_c = t[i], y[i]
            if y[i] == 0:
                last_h = t[i]
    X = F.values.astype(float)
    P = np.zeros((len(y), 3))
    for bb in b.baby.unique():
        te, tr = (b.baby == bb).values, (b.baby != bb).values
        rf = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        P[np.ix_(te, rf.classes_)] = rf.predict_proba(X[te])
    return P


def cmd_final(a):
    """제품 경로 최종 비교: 맥락 + 아기 이력(기존 최고)에 '그 아기의 이전 울음 소리와의 유사도'를 더하면 얼마나 오르나.
    이전 울음 10개 이상 쌓인 뒤만 평가. Δ는 아기 단위 부트스트랩 95% 구간."""
    import joint_train as jt
    jt.mem_guard()
    from sklearn.decomposition import PCA
    m = load_meta()
    B = bout_table(m)
    agg = m.groupby("bout").agg(n_syl=("n_syl", "sum"), voiced=("voiced", "sum"), n_ep=("ep", "count"),
                                gap_med=("gap_med", "median"))
    B = B.join(agg, on="bout")
    sel = np.where(B.stop.isin(CAUSES))[0]
    y = np.array([CAUSES.index(c) for c in B.stop.values[sel]])
    babies, t = B.baby.values[sel], pd.to_datetime(B.t.values[sel])
    day = t.normalize().values
    Pctx = context_probs(B, sel, y)
    order = {b: sorted(np.where(babies == b)[0], key=lambda i: (t[i], B.key.values[sel][i])) for b in np.unique(babies)}
    rows, per_item = [], []
    for model in a.models.split(","):
        X = unit_feats(m, model)["bout"][B.bout.values[sel]].reshape(len(sel), -1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        Z = PCA(64, random_state=SEED).fit_transform(X)
        for history in ("all", "prev_days"):
            for b, idx in order.items():
                for k, i in enumerate(idx):
                    past = [j for j in idx[:k] if history == "all" or day[j] < day[i]]
                    if len(past) < 10:
                        continue
                    cnt = np.bincount(y[past], minlength=3)
                    prior = (cnt + 1) / (cnt.sum() + 3)
                    mu = Z[past].mean(0)
                    zc = Z[i] - mu
                    sims = np.full(3, -1.0)
                    for c in range(3):
                        if cnt[c]:
                            pc = (Z[past][y[past] == c] - mu).mean(0)
                            sims[c] = zc @ pc / (np.linalg.norm(zc) * np.linalg.norm(pc) + 1e-9)
                    proto = np.exp(5 * sims)
                    proto /= proto.sum()
                    for name, p in (("이력만", prior), ("맥락+이력(기존 최고)", Pctx[i] * prior),
                                    ("이력+소리 개인화", prior * proto), ("맥락+이력+소리 개인화", Pctx[i] * prior * proto)):
                        per_item.append({"model": model, "history": history, "method": name, "i": i, "baby": b,
                                         "y": y[i], **{f"p{c}": (p / p.sum())[c] for c in range(3)}})
    D = pd.DataFrame(per_item)
    rng = np.random.default_rng(SEED)
    for (model, history), g in D.groupby(["model", "history"]):
        base = g[g.method == "맥락+이력(기존 최고)"].set_index("i").sort_index()
        ub = base.baby.unique()
        boots = [rng.choice(ub, len(ub)) for _ in range(1000)]
        for name, h in g.groupby("method"):
            h = h.set_index("i").loc[base.index]
            P, yy = h[["p0", "p1", "p2"]].values, h.y.values
            r = metrics(yy, P)
            dl = []
            for bs in boots:
                ix = np.concatenate([np.where(base.baby.values == u)[0] for u in bs])
                dl.append(metrics(yy[ix], P[ix])["bal"] - metrics(yy[ix], base[["p0", "p1", "p2"]].values[ix])["bal"])
            lo, hi = np.percentile(dl, [2.5, 97.5])
            rows.append({"model": model, "history": history, "method": name, "n": len(yy), **r,
                         "d_bal": r["bal"] - metrics(yy, base[["p0", "p1", "p2"]].values)["bal"], "d_lo": lo, "d_hi": hi})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(root(), "runs", "final.csv"), index=False)
    print(R.round(3).to_string())


def cmd_report(_):
    base = os.path.join(root(), "runs")
    A = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(base, "*", "babysplit.csv"))])
    Bp = pd.concat([pd.read_csv(p).assign(model=os.path.basename(os.path.dirname(p)))
                    for p in glob.glob(os.path.join(base, "*", "personal.csv"))])
    L = ["# 울음 원인 3차 점검 (Enes 데이터 구성 수정)", "",
         "- 음절을 무음 없이 이어 붙이고 길이 상한 없이 울음 전체를 썼다. 5초 넘는 공백에서 나눈 조각 단위도 봤다.",
         "- 균형 정확도 무작위 0.333, 배고픔 AUC 무작위 0.5. '귀무 95%'는 라벨을 섞어 같은 절차를 돌린 값의 95백분위.", "",
         "## A. 아기 분리 (5겹 × 5회 반복 평균)", "",
         "| 모델 | 라벨 | 단위 | n | 균형 정확도 | 귀무 95% | 배고픔 AUC | 귀무 95% |", "|---|---|---|---:|---:|---:|---:|---:|"]
    a = A[A.age == "all"]
    for (mod, lab, unit), g in a.groupby(["model", "label", "unit"]):
        rp, pm = g[g.kind == "repeat"], g[g.kind == "perm"]
        L.append(f"| {mod} | {lab} | {unit} | {int(rp.n.iloc[0])} | {rp.bal.mean():.3f} | {pm.bal.quantile(.95):.3f} | "
                 f"{rp.auc.mean():.3f} | {pm.auc.quantile(.95):.3f} |")
    L += ["", "## A-2. 녹음 회차(나이)별 균형 정확도 (라벨=멈춘 행동, 단위=울음 전체)", "",
          "| 모델 | 1회차(15일) | 2회차(1.5개월) | 3회차(2.5개월) | 4회차(3.5개월) |", "|---|---:|---:|---:|---:|"]
    ag = A[(A.age != "all") & (A.label == "stop") & (A.unit == "bout") & (A.kind == "repeat")]
    for mod, g in ag.groupby("model"):
        v = g.groupby("age").bal.mean()
        L.append(f"| {mod} | " + " | ".join(f"{v.get(str(x), v.get(x, float('nan'))):.3f}" for x in (1, 2, 3, 4)) + " |")
    L += ["", "## B. 아기별 시간순 개인화 (이전 울음 10개 이상 쌓인 뒤, 라벨=멈춘 행동)", "",
          "| 모델 | 방법 | 균형 정확도 | 귀무 95% (아기 안 라벨 섞기) | 배고픔 AUC | 귀무 95% |", "|---|---|---:|---:|---:|---:|"]
    for (mod, meth), g in Bp.groupby(["model", "method"]):
        re, pm = g[g.kind == "real"], g[g.kind == "perm"]
        L.append(f"| {mod} | {meth} | {re.bal.mean():.3f} | {pm.bal.quantile(.95):.3f} | {re.auc.mean():.3f} | "
                 f"{pm.auc.quantile(.95):.3f} |")
    L += ["", "prior는 소리를 안 쓰고 그 아기의 이전 원인 빈도만 쓴다. 소리가 쓸모 있으려면 prior보다, 그리고 귀무보다 높아야 한다."]
    P2 = [pd.read_csv(p) for p in glob.glob(os.path.join(base, "*", "personal2.csv"))]
    if P2:
        P2 = pd.concat(P2)
        L += ["", "## B-2. 개인화 대조: 같은 날 울음 제외 / 시간대만", "",
              "history=all은 이전 울음 전부, prev_days는 같은 날 울음을 빼고 이전 날만, prev_age는 이전 녹음 회차(2주~1개월 전)만 쓴다.",
              "timeproto는 소리 대신 녹음 시각만으로 같은 계산을 한 것이다. 이것도 높으면 시간대 효과다.", "",
              "| 모델 | 이력 | 방법 | n | 균형 정확도 | 귀무 95% | 배고픔 AUC | 귀무 95% |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for (mod, hist, meth), g in P2.groupby(["model", "history", "method"]):
            re, pm = g[g.kind == "real"], g[g.kind == "perm"]
            L.append(f"| {mod} | {hist} | {meth} | {int(re.n.iloc[0])} | {re.bal.mean():.3f} | {pm.bal.quantile(.95):.3f} | "
                     f"{re.auc.mean():.3f} | {pm.auc.quantile(.95):.3f} |")
    fp = os.path.join(base, "final.csv")
    if os.path.exists(fp):
        R = pd.read_csv(fp)
        L += ["", "## C. 제품 경로 최종 비교 (이전 울음 10개 이상 쌓인 뒤, 라벨=멈춘 행동)", "",
              "맥락 RF는 소리 모델 없이 시간대·나이·직전 울음 이후 시간·직전 원인·음절 리듬만 쓴다(한 아기씩 빼고 학습).",
              "소리 개인화는 그 아기의 이전 울음 중 원인별 평균과의 코사인 유사도다. Δ는 기존 최고 대비, 아기 단위 부트스트랩 95% 구간.", "",
              "| 소리 모델 | 이력 | 방법 | n | 균형 정확도 | 배고픔 AUC | Δ균형 정확도 (95% 구간) |", "|---|---|---|---:|---:|---:|---|"]
        for r in R.itertuples():
            L.append(f"| {r.model} | {r.history} | {r.method} | {r.n} | {r.bal:.3f} | {r.auc:.3f} | "
                     f"{r.d_bal:+.3f} ({r.d_lo:+.2f}~{r.d_hi:+.2f}) |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(root(), "ENES_V3_REPORT.md"), "w", encoding="utf-8").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build")
    sp.add_parser("yamnet")
    p = sp.add_parser("extract")
    p.add_argument("--model", required=True)
    p = sp.add_parser("smile")
    p.add_argument("--set", choices=list(SMILE_SETS), required=True)
    p.add_argument("--workers", type=int, default=12)
    p = sp.add_parser("eval")
    p.add_argument("--model", required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--perms", type=int, default=5)
    p.add_argument("--pperms", type=int, default=20)
    p = sp.add_parser("final")
    p.add_argument("--models", default="yamnet,clap,ast")
    p = sp.add_parser("personal")
    p.add_argument("--model", required=True)
    p.add_argument("--pperms", type=int, default=50)
    sp.add_parser("report")
    a = ap.parse_args()
    {"build": cmd_build, "yamnet": cmd_yamnet, "extract": cmd_extract, "smile": cmd_smile, "eval": cmd_eval, "personal": cmd_personal, "final": cmd_final,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
