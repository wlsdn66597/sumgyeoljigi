"""파이에 올릴 개인화 힌트(../cry_personal.py)를 Enes 울음으로 시간순 재생해 시험한다(로컬, CPU).

    python sim_personal.py embed     # 파이와 같은 방식(0.975초 창, 1초 간격, TFLite)으로 YAMNet 임베딩 계산
    python sim_personal.py basis     # 공개 데이터(DAC·Baidu, Enes 제외)로 표준화·PCA → artifacts/cry_personal_basis.npz
    python sim_personal.py ctx       # Enes 전체로 맥락 모델 학습 → artifacts/cry_personal_ctx.json (파이 배포용)
    python sim_personal.py run       # 시나리오별 재생 → DATA_ROOT/runs/enes3/SIM_PERSONAL_REPORT.md
    python sim_personal.py tune      # 최근 답 가중·소리 반영 세기·기준 변환을 아기 절반에서 고르고 나머지에서 평가

부모 답은 Enes 라벨(울음을 멈추게 한 행동)을 그대로 쓴다: F → 수유(feed), I → 기저귀·자세(diaper), S → 안아주기(hold).
시나리오
  - 힌트 시점: 울음 시작 10초 / 30초 / 울음 전체 (실제로는 울기 시작하고 곧바로 힌트를 줘야 한다)
  - 부모 답 비율: 매번 / 절반 / 4번에 1번 (답이 없으면 그 울음은 라벨 없이 저장되고 '마지막 수유' 시각도 모른다)
  - 잘못 누름: 0% / 20%
맥락 모델은 평가하는 아기를 빼고 학습한다(한 아기씩 빼기). 정답은 언제나 실제 Enes 라벨이다.
귀무: 아기 안에서 라벨을 섞어 같은 절차를 반복한다.
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
ENES3 = os.path.join(DATA, "enes3")
ART = os.path.join(HERE, "artifacts")
OUT = os.path.join(DATA, "runs", "enes3")
WIN, HOP, SR = 15600, 16000, 16000
CRY_IDX = (19, 20)                    # Crying, sobbing / Baby cry, infant cry
CRY_THR = 0.3                         # 파이 울음 판정 임계값과 같게
MAP = {"F": "feed", "I": "diaper", "S": "hold"}
K_MIN = 10


class Yam:
    def __init__(self):
        import tensorflow as tf
        self.it = tf.lite.Interpreter(model_content=open(os.path.join(ART, "yamnet_emb.tflite"), "rb").read())
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = {int(d["shape"][-1]): d["index"] for d in self.it.get_output_details()}

    def __call__(self, x):
        self.it.set_tensor(self.inp, x.astype(np.float32))
        self.it.invoke()
        return self.it.get_tensor(self.out[1024]).reshape(-1), self.it.get_tensor(self.out[521]).reshape(-1)


def device_windows(x):
    """파이처럼 1초마다 최근 0.975초를 본다."""
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    return [x[s:s + WIN] for s in range(0, len(x) - WIN + 1, HOP)]


def embed_files(paths, yam, tag):
    E, C, owner = [], [], []
    t0 = time.time()
    for i, p in enumerate(paths):
        x, _ = sf.read(p, dtype="float32")
        for w in device_windows(x):
            e, s = yam(w)
            E.append(e.astype(np.float16))
            C.append(max(s[j] for j in CRY_IDX))
            owner.append(i)
        if (i + 1) % 300 == 0:
            print(f"  {tag} {i + 1}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    return np.stack(E), np.array(C, np.float32), np.array(owner)


def cmd_embed(_):
    yam = Yam()
    m = pd.read_csv(os.path.join(ENES3, "meta.csv"), dtype={"date": str, "time": str})
    E, C, ow = embed_files([os.path.join(ENES3, "wav", i + ".wav") for i in m.id], yam, "enes")
    np.savez(os.path.join(ENES3, "feats", "device_yamnet.npz"), ids=m.id.values, X=E, cry=C, ep=ow)
    pub = sorted(glob.glob(os.path.join(DATA, "joint", "wav", "dac", "*.wav")) +
                 glob.glob(os.path.join(DATA, "joint", "wav", "baidu", "*.wav")))
    E2, C2, ow2 = embed_files(pub, yam, "public")
    np.savez(os.path.join(ENES3, "feats", "device_yamnet_public.npz"), X=E2, cry=C2, ep=ow2)
    print(f"Enes 창 {len(E)}개(울음 판정 {float((C >= CRY_THR).mean()):.0%}), 공개 데이터 창 {len(E2)}개")


def cmd_basis(_):
    z = np.load(os.path.join(ENES3, "feats", "device_yamnet_public.npz"))
    X = z["X"].astype(np.float64)[z["cry"] >= CRY_THR]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    P = Vt[:16].T
    np.savez(os.path.join(ART, "cry_personal_basis.npz"), mean=mu, std=sd, pca=P)
    print(f"basis: 공개 울음 창 {len(X)}개, PCA 16차원 설명 분산 {float((S[:16] ** 2).sum() / (S ** 2).sum()):.0%}")


# --- Enes 울음(bout) 테이블 ------------------------------------------------------------
def bouts():
    m = pd.read_csv(os.path.join(ENES3, "meta.csv"), dtype={"date": str, "time": str})
    # 파일명 하나가 날짜·시각 경계가 한 칸 밀려 있다(101020192_058 = 10102019_2058). 붙여서 다시 자른다.
    dt = (m.date.str.zfill(8) + m.time.str.zfill(4)).where(m.date.str.len() <= 8, m.date + m.time)
    m["t"] = pd.to_datetime(dt.str[:12], format="%d%m%Y%H%M", errors="coerce")
    z = np.load(os.path.join(ENES3, "feats", "device_yamnet.npz"), allow_pickle=True)
    assert (z["ids"] == m.id.values).all()
    X, C, ep = z["X"], z["cry"], z["ep"]
    rows = []
    for b, g in m.groupby("bout"):
        g = g.sort_values("ep")
        if g.stop.iloc[0] not in MAP:
            continue
        idx = np.concatenate([np.where(ep == i)[0] for i in g.index])     # 시간순 창
        keep = idx[C[idx] >= CRY_THR]
        t0 = time.mktime(pd.Timestamp(g.t.iloc[0]).timetuple())
        span = float((g.t0 + g.dur).max() - g.t0.min())
        rows.append({"bout": b, "baby": g.baby.iloc[0], "y": "FIS".index(g.stop.iloc[0]), "t0": t0, "t1": t0 + span,
                     "win": keep, "key": g.key.iloc[0]})
    rows.sort(key=lambda r: (r["baby"], r["t0"], r["key"]))
    return rows, X


def ctx_train(rows, exclude=None):
    """맥락 로지스틱 회귀(3클래스). 학습용 특징은 부모 답이 모두 있다고 보고 만든다."""
    from sklearn.linear_model import LogisticRegression
    F, Y = [], []
    by = collections.defaultdict(list)
    for r in rows:
        by[r["baby"]].append(r)
    for bb, rs in by.items():
        if bb == exclude:
            continue
        last_cry, last_feed = None, None
        for r in rs:
            F.append(cp.ctx_features(r["t0"], last_cry, last_feed))
            Y.append(r["y"])
            last_cry = r["t1"]
            if r["y"] == 0:
                last_feed = r["t1"]
    F, Y = np.array(F), np.array(Y)
    mu, sd = F.mean(0), F.std(0) + 1e-6
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit((F - mu) / sd, Y)
    return {"coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(), "mean": mu.tolist(), "std": sd.tolist(),
            "features": ["hour_sin", "hour_cos", "h_since_prev_cry", "h_since_feed"], "classes": cp.ACTIONS3}


def cmd_ctx(_):
    import json
    rows, _ = bouts()
    c = ctx_train(rows)
    json.dump(c, open(os.path.join(ART, "cry_personal_ctx.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장: artifacts/cry_personal_ctx.json")


# --- 재생 ---------------------------------------------------------------------------
def replay(rows, X, basis, ctxs, hint_s, p_answer, noise, seed, labels, half_life=None, w_sound=1.0,
           proto_from="all", sound_method="proto", **kw):
    """아기마다 새 CryPersonal을 만들어 울음을 시간순으로 넣는다. 반환: 평가 대상 울음(이전 울음 10개 이상)의 기록."""
    rng = np.random.default_rng(seed)
    rec = []
    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["baby"]].append(k)
    for bb, ks in by.items():
        bs = basis[bb] if bb in basis else basis
        dev = cp.CryPersonal(bs, ctxs[bb], actions=cp.ACTIONS3, k_min=K_MIN, hint_seconds=hint_s or 10 ** 6, half_life=half_life,
                             w_sound=w_sound, proto_from=proto_from, sound_method=sound_method, **kw)
        for n, k in enumerate(ks):
            r = rows[k]
            y = labels[k]
            win = r["win"]
            first = win if hint_s is None else win[:hint_s]
            h = dev.hint(r["t0"], embs=X[first].astype(np.float32) if len(first) else [])
            if n >= K_MIN:
                rec.append({"baby": bb, "n": n, "y": y, "nfb": h["n_feedback"], "show": h["show"],
                            **{f"{part}{c}": h["parts"].get(part, [1 / 3] * 3)[c]
                               for part in ("context", "history", "sound") for c in range(3)}})
            dev.start(r["t0"])
            for w in win:
                dev.add_window(X[w].astype(np.float32))
            eid = dev.end(r["t1"])
            if eid is not None and rng.random() < p_answer:
                a = y if rng.random() >= noise else rng.choice([c for c in range(3) if c != y])
                dev.feedback(eid, cp.ACTIONS3[a])
    return pd.DataFrame(rec)


def score(D, combo):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    P = np.ones((len(D), 3))
    for part in combo:
        P = P * D[[f"{part}{c}" for c in range(3)]].values
    P = P / P.sum(1, keepdims=True)
    y = D.y.values
    return balanced_accuracy_score(y, P.argmax(1)), roc_auc_score(y == 0, P[:, 0]), P


COMBOS = {"맥락만": ("context",), "이력만": ("history",), "맥락+이력(기기 기본)": ("context", "history"),
          "소리 개인화만": ("sound",), "맥락+이력+소리": ("context", "history", "sound")}
DEVICE = "맥락+이력(기기 기본)"


def cmd_run(a):
    rows, X = bouts()
    b = np.load(os.path.join(ART, "cry_personal_basis.npz"))
    basis = {"mean": b["mean"], "std": b["std"], "pca": b["pca"]}
    ctxs = {bb: ctx_train(rows, exclude=bb) for bb in sorted(set(r["baby"] for r in rows))}
    y_true = np.array([r["y"] for r in rows])
    res = []
    scen = [(hs, pa, nz) for hs in (10, 30, None) for pa, nz in ((1.0, 0.0), (0.5, 0.0), (0.25, 0.0), (1.0, 0.2), (0.5, 0.2))]
    t0 = time.time()
    for hs, pa, nz in scen:
        seeds = [0] if (pa == 1.0 and nz == 0.0) else list(range(a.seeds))
        for sd in seeds:
            D = replay(rows, X, basis, ctxs, hs, pa, nz, sd, y_true, half_life=a.half_life)
            for name, combo in COMBOS.items():
                bal, auc, P = score(D, combo)
                row = {"hint": "전체" if hs is None else f"{hs}초", "answer": pa, "noise": nz, "seed": sd, "combo": name,
                       "kind": "real", "n": len(D), "bal": bal, "auc": auc}
                if name == DEVICE:
                    for th in (0.4, 0.5, 0.6):
                        sh = (P.max(1) >= th) & (D.nfb.values >= K_MIN)
                        row[f"cov{th}"] = float(sh.mean())
                        row[f"acc{th}"] = float((P.argmax(1)[sh] == D.y.values[sh]).mean()) if sh.any() else np.nan
                res.append(row)
        print(f"  힌트 {hs}, 답 {pa}, 오답 {nz} 완료 {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(7)                   # 귀무: 기본 시나리오(10초, 매번 답, 오답 0)
    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["baby"]].append(k)
    for r_ in range(a.perms):
        yp = y_true.copy()
        for bb, ks in by.items():
            yp[ks] = rng.permutation(yp[ks])
        D = replay(rows, X, basis, ctxs, 10, 1.0, 0.0, 0, yp, half_life=a.half_life)
        for name, combo in COMBOS.items():
            bal, auc, _ = score(D, combo)
            res.append({"hint": "10초", "answer": 1.0, "noise": 0.0, "seed": r_, "combo": name, "kind": "perm",
                        "n": len(D), "bal": bal, "auc": auc})
    R = pd.DataFrame(res)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(os.path.join(OUT, "sim_personal.csv"), index=False)
    report(R)


def report(R):
    re, pm = R[R.kind == "real"], R[R.kind == "perm"]
    L = ["# 파이 개인화 힌트 재생 시험 (cry_personal.py, EnesBabyCries 아기 24명)", "",
         "- 파이와 같은 TFLite YAMNet(0.975초 창, 1초 간격, 울음 판정 0.3 이상만), 공개 데이터로 만든 PCA 16차원.",
         "- 부모 답 = Enes 라벨(F 수유 / I 기저귀·자세 / S 안아주기). 각 아기의 11번째 울음부터 평가(377개).",
         "- 이력은 최근 답 가중(답 8개마다 절반), 소리 비교는 이전 울음 전체 평균과 코사인(가중 없음).",
         "- 균형 정확도 무작위 0.333, 배고픔 AUC 무작위 0.5. 귀무는 아기 안 라벨 섞기 20회(10초·매번 답).", "",
         "## 1. 기본 시나리오 (부모가 매번 정확히 답함)", "",
         "| 조합 | 힌트 10초 | 30초 | 울음 전체 | 귀무 95% (10초) | 배고픔 AUC 10초 |", "|---|---:|---:|---:|---:|---:|"]
    base = re[(re.answer == 1.0) & (re.noise == 0.0)]
    for name in COMBOS:
        g = base[base.combo == name].set_index("hint")
        nb = pm[pm.combo == name]
        L.append(f"| {name} | {g.loc['10초', 'bal']:.3f} | {g.loc['30초', 'bal']:.3f} | {g.loc['전체', 'bal']:.3f} | "
                 f"{nb.bal.quantile(.95):.3f} | {g.loc['10초', 'auc']:.3f} |")
    L += ["", "## 2. 부모 답이 불완전할 때 (힌트 10초, 무작위 시드 평균)", "",
          "| 부모 답 비율 | 잘못 누름 | 맥락+이력(기기 기본) | 배고픔 AUC | (비교) +소리 |", "|---|---|---:|---:|---:|"]
    t10 = re[re.hint == "10초"]
    for (pa, nz), g in t10.groupby(["answer", "noise"]):
        dev = g[g.combo == DEVICE]
        snd = g[g.combo == "맥락+이력+소리"]
        L.append(f"| {pa:.0%} | {nz:.0%} | {dev.bal.mean():.3f} | {dev.auc.mean():.3f} | {snd.bal.mean():.3f} |")
    L += ["", "## 3. 확신이 높을 때만 보여주기 (맥락+이력, 힌트 10초, 매번 답)", "",
          "| 기준 확신 | 보여준 비율 | 보여준 힌트의 정확도 |", "|---|---:|---:|"]
    d0 = base[(base.hint == "10초") & (base.combo == DEVICE)].iloc[0]
    for th in (0.4, 0.5, 0.6):
        L.append(f"| {th:.0%} 이상 | {d0[f'cov{th}']:.0%} | {d0[f'acc{th}']:.3f} |")
    L += ["", "3분류에서 무작위로 찍으면 정확도는 약 0.33~0.43(배고픔 비율)이다."]
    txt = "\n".join(L) + "\n"
    open(os.path.join(OUT, "SIM_PERSONAL_REPORT.md"), "w", encoding="utf-8").write(txt)
    print(txt)


def enes_lobo_basis(rows, X):
    """평가하는 아기를 뺀 Enes 울음 창으로 표준화·PCA(라벨은 안 쓴다)."""
    out = {}
    for bb in sorted(set(r["baby"] for r in rows)):
        idx = np.concatenate([r["win"] for r in rows if r["baby"] != bb])
        A = X[idx].astype(np.float64)
        mu, sd = A.mean(0), A.std(0) + 1e-6
        As = (A - mu) / sd
        _, _, Vt = np.linalg.svd(As - As.mean(0), full_matrices=False)
        out[bb] = {"mean": mu, "std": sd, "pca": Vt[:16].T}
    return out


def device_probs(D, w_sound):
    P = np.ones((len(D), 3))
    for part, w in (("context", 1.0), ("history", 1.0), ("sound", w_sound)):
        P = P * D[[f"{part}{c}" for c in range(3)]].values ** w
    return P / P.sum(1, keepdims=True)


def nested_pick(preds, y, babies, splits=5):
    """preds: 설정별 확률. 아기 절반에서 고르고 나머지 절반에서 평가(서로 바꿔서), 분할 평균."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    ub = np.unique(babies)
    bal, auc, picks = [], [], []
    for sp in range(splits):
        A = set(np.random.default_rng(100 + sp).choice(ub, len(ub) // 2, replace=False))
        P = np.zeros_like(preds[0])
        for tune, test in ((A, set(ub) - A), (set(ub) - A, A)):
            tm = np.isin(babies, list(tune))
            best = max(range(len(preds)), key=lambda c: balanced_accuracy_score(y[tm], preds[c][tm].argmax(1)))
            picks.append(best)
            em = np.isin(babies, list(test))
            P[em] = preds[best][em]
        bal.append(balanced_accuracy_score(y, P.argmax(1)))
        auc.append(roc_auc_score(y == 0, P[:, 0]))
    return float(np.mean(bal)), float(np.mean(auc)), collections.Counter(picks).most_common(1)[0][0]


def cmd_tune(a):
    rows, X = bouts()
    b = np.load(os.path.join(ART, "cry_personal_basis.npz"))
    bases = {"public": {"mean": b["mean"], "std": b["std"], "pca": b["pca"]}, "enes_lobo": enes_lobo_basis(rows, X)}
    ctxs = {bb: ctx_train(rows, exclude=bb) for bb in sorted(set(r["baby"] for r in rows))}
    y_true = np.array([r["y"] for r in rows])
    grid = [(bn, hl, ws) for bn in bases for hl in (None, 8, 16) for ws in (0.5, 1.0, 2.0)]
    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["baby"]].append(k)
    res, t0 = [], time.time()

    def run_grid(labels):
        out = {}                                 # 소리 세기는 결합 단계에서만 쓰여서 재생은 (기준, 가중)마다 한 번
        for bn in bases:
            for hl in (None, 8, 16):
                D = replay(rows, X, bases[bn], ctxs, 10, 1.0, 0.0, 0, labels, half_life=hl)
                for ws in (0.5, 1.0, 2.0):
                    out[(bn, hl, ws)] = D
        return out

    def summarize(runs, kind, r):
        D0 = next(iter(runs.values()))
        y, babies = D0.y.values, D0.baby.values
        cfgs = list(runs)
        dev = [device_probs(runs[c], c[2]) for c in cfgs]
        ch = [device_probs(runs[c], 0.0) for c in cfgs]                 # 맥락+이력(최근 가중만 선택)
        snd = [runs[c][[f"sound{k}" for k in range(3)]].values for c in cfgs]
        for name, preds in (("맥락+이력+소리", dev), ("맥락+이력", ch), ("소리만", snd)):
            bal, auc, pick = nested_pick(preds, y, babies)
            res.append({"combo": name, "kind": kind, "r": r, "bal": bal, "auc": auc, "cfg": str(cfgs[pick])})

    summarize(run_grid(y_true), "real", 0)
    print(f"  실제 라벨 완료 {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(7)
    for r_ in range(a.perms):
        yp = y_true.copy()
        for bb, ks in by.items():
            yp[ks] = rng.permutation(yp[ks])
        summarize(run_grid(yp), "perm", r_)
        print(f"  귀무 {r_ + 1}/{a.perms} {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "sim_personal_tune.csv"), index=False)
    re, pm = R[R.kind == "real"], R[R.kind == "perm"]
    L = ["# 파이 개인화 힌트 조정 (힌트 10초, 부모가 매번 답, 아기 절반에서 설정 선택·나머지 평가)", "",
         "설정 후보: 기준 변환(공개 데이터 / 평가 아기를 뺀 Enes) × 최근 답 가중(없음·8개·16개마다 절반) × 소리 반영 세기(0.5·1·2)", "",
         "| 조합 | 균형 정확도 | 귀무 95% | 배고픔 AUC | 귀무 95% | 자주 고른 설정 |", "|---|---:|---:|---:|---:|---|"]
    for name in ("맥락+이력", "소리만", "맥락+이력+소리"):
        g, n = re[re.combo == name].iloc[0], pm[pm.combo == name]
        L.append(f"| {name} | {g.bal:.3f} | {n.bal.quantile(.95):.3f} | {g.auc:.3f} | {n.auc.quantile(.95):.3f} | {g.cfg} |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(OUT, "SIM_PERSONAL_TUNE.md"), "w", encoding="utf-8").write(txt)
    print(txt)


def cmd_ctxp(a):
    """아기별 맥락(이 아기의 답으로 다시 배운 맥락 모델) vs 공통 맥락. 실제 Enes, 힌트 10초, 매번 답, 소리 안 섞음.
    공통 모델은 평가하는 아기를 뺀 아기들로 학습(한 아기씩 빼기). 귀무 = 아기 안 라벨 섞기."""
    rows, X = bouts()
    b = np.load(os.path.join(ART, "cry_personal_basis.npz"))
    basis = {"mean": b["mean"], "std": b["std"], "pca": b["pca"]}
    ctxs = {bb: ctx_train(rows, exclude=bb) for bb in sorted(set(r["baby"] for r in rows))}
    y_true = np.array([r["y"] for r in rows])
    cfgs = {"공통 맥락": dict(ctx_mode="global")}
    for lam in (1.0, 3.0, 10.0):
        cfgs[f"아기별 맥락(당김 {lam:g})"] = dict(ctx_mode="personal", ctx_lambda=lam)
    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["baby"]].append(k)
    res, rng = [], np.random.default_rng(7)
    for r_ in range(a.perms + 1):
        yp = y_true.copy()
        if r_:
            for bb, ks in by.items():
                yp[ks] = rng.permutation(yp[ks])
        for name, kw in cfgs.items():
            D = replay(rows, X, basis, ctxs, 10, 1.0, 0.0, 0, yp, half_life=a.half_life, w_sound=0.0, **kw)
            for combo in ("맥락만", "맥락+이력(기기 기본)"):
                bal, auc, _ = score(D, COMBOS[combo])
                res.append({"cfg": name, "combo": combo, "kind": "perm" if r_ else "real", "r": r_, "bal": bal,
                            "auc": auc})
        print(f"  {r_}/{a.perms}", flush=True)
    R = pd.DataFrame(res)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(os.path.join(OUT, "ctx_personal.csv"), index=False)
    L = ["# 아기별 맥락 vs 공통 맥락 (실제 Enes 아기 24명, 이전 답 10개 이상, 매번 답, 소리 없음)", "",
         "| 맥락 모델 | 조합 | 균형 정확도 | 귀무 95% | 배고픔 AUC | 귀무 95% |", "|---|---|---:|---:|---:|---:|"]
    for (c, combo), g in R.groupby(["cfg", "combo"], sort=False):
        re_, pm = g[g.kind == "real"].iloc[0], g[g.kind == "perm"]
        L.append(f"| {c} | {combo} | {re_.bal:.3f} | {pm.bal.quantile(.95):.3f} | {re_.auc:.3f} | "
                 f"{pm.auc.quantile(.95):.3f} |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(OUT, "CTX_PERSONAL.md"), "w", encoding="utf-8").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("embed", "basis", "ctx"):
        sp.add_parser(c)
    p = sp.add_parser("run")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--perms", type=int, default=20)
    p.add_argument("--half-life", dest="half_life", type=float, default=8)
    p = sp.add_parser("ctxp")
    p.add_argument("--perms", type=int, default=20)
    p.add_argument("--half-life", type=float, default=8)
    p = sp.add_parser("tune")
    p.add_argument("--perms", type=int, default=20)
    a = ap.parse_args()
    {"embed": cmd_embed, "basis": cmd_basis, "ctx": cmd_ctx, "run": cmd_run, "tune": cmd_tune, "ctxp": cmd_ctxp}[a.cmd](a)


if __name__ == "__main__":
    main()
