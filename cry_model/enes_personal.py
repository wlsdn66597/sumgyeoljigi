"""아기별 개인화 강화: 그 아기의 이전 울음(부모 피드백 라벨)으로 지금 울음의 원인을 맞히는 방법 비교.
enes_v3.py가 만든 데이터·특징(ENES3/feats)을 쓴다. GPU 서버에서 실행.

    python enes_personal.py run --model clap          # 모델 하나(또는 ens: yamnet·clap·ast 평균)
    python enes_personal.py report                    # → ENES3/PERSONAL4_REPORT.md

방법 (모두 시간순으로, 그 시점 이전 울음과 그 라벨만 쓴다)
  proto  : 원인별 평균과 코사인 (3차 방식)
  knn    : 가장 비슷한 이전 울음 k개의 가중 투표
  lr     : 그 아기의 이전 울음만으로 학습한 작은 로지스틱 회귀
  setsim : 4초 창 단위 비교. 지금 울음의 각 창이 원인별 이전 창과 얼마나 닮았는지
  meta   : 다른 아기들로 '같은 아기 안에서 원인을 가르는 방향'을 학습한 투영(prototypical loss) 뒤 proto
엿보기 방지: 설정(차원·온도·k·C)은 아기를 반으로 나눠 한쪽에서 고르고 다른 쪽에서 평가한다(서로 바꿔서, 분할 5번).
귀무: 아기 안에서 라벨을 섞고 설정 고르기까지 같은 절차를 반복한다.
"""
import argparse
import collections
import glob
import os
import time

import numpy as np
import pandas as pd

import enes_v3 as ev

K_MIN = 10
SPLITS = 5
ENS = ["yamnet", "clap", "ast"]
GRID = {
    "proto": [{"d": d, "T": T} for d in (16, 32, 64) for T in (3, 10)],
    "knn": [{"d": 32, "k": k, "T": 10} for k in (3, 7, 15)],
    "lr": [{"d": d, "C": C} for d in (16, 32) for C in (0.1, 1.0)],
    "setsim": [{"d": 32, "T": T} for T in (5, 15)],
    "meta": [{"d": 64, "out": o, "T": 10} for o in (8, 16)],
}


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# --- 개별 방법: (특징, 지금 울음 i, 이전 울음 past, 라벨 y, 설정) → 확률 3개 -------------------
def m_proto(Z, i, past, y, c):
    mu = Z[past].mean(0)
    s = np.full(3, -1.0)
    for k in range(3):
        if (y[past] == k).any():
            s[k] = cos(Z[i] - mu, (Z[past][y[past] == k] - mu).mean(0))
    return softmax(c["T"] * s)


def m_knn(Z, i, past, y, c):
    mu = Z[past].mean(0)
    P, z = Z[past] - mu, Z[i] - mu
    sims = P @ z / (np.linalg.norm(P, axis=1) * np.linalg.norm(z) + 1e-9)
    top = np.argsort(-sims)[: c["k"]]
    p = np.bincount(y[past][top], weights=np.exp(c["T"] * sims[top]), minlength=3) + 1e-3
    return p / p.sum()


def m_lr(Z, i, past, y, c):
    from sklearn.linear_model import LogisticRegression
    mu = Z[past].mean(0)
    if len(np.unique(y[past])) < 2:
        return np.bincount(y[past], minlength=3) + 1.0 / (len(past) + 3)
    lr = LogisticRegression(C=c["C"], max_iter=500, class_weight="balanced").fit(Z[past] - mu, y[past])
    p = np.full(3, 1e-3)
    p[lr.classes_] = lr.predict_proba((Z[i] - mu)[None])[0]
    return p / p.sum()


def m_setsim(WB, i, past, y, c):
    allp = np.concatenate([WB[j] for j in past])
    mu = allp.mean(0)
    cur = WB[i] - mu
    cur = cur / (np.linalg.norm(cur, axis=1, keepdims=True) + 1e-9)
    s = np.full(3, -1.0)
    for k in range(3):
        js = [j for j in past if y[j] == k]
        if js:
            W = np.concatenate([WB[j] for j in js]) - mu
            W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
            s[k] = float((cur @ W.T).max(1).mean())
    return softmax(c["T"] * s)


# --- 특징 준비 ------------------------------------------------------------------------
class Feats:
    """한 모델의 울음 단위 특징(PCA d차원)과 창 단위 특징. 비지도 변환이라 라벨은 안 쓴다."""

    def __init__(self, model, m, B, sel):
        from sklearn.decomposition import PCA
        U = ev.unit_feats(m, model)
        X = U["bout"][B.bout.values[sel]].reshape(len(sel), -1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        self.Z = {d: PCA(d, random_state=ev.SEED).fit_transform(X) for d in (16, 32, 64)}
        z = np.load(os.path.join(ev.root(), "feats", f"{model}.npz"), allow_pickle=True)
        Xw, ep = z["X"], z["ep"]
        Xw = Xw.astype(np.float32).mean(1)                      # 층 평균
        Xw = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-6)
        rng = np.random.default_rng(ev.SEED)
        pca = PCA(32, random_state=ev.SEED).fit(Xw[rng.choice(len(Xw), min(6000, len(Xw)), replace=False)])
        Ww = pca.transform(Xw)
        wb = m.bout.values[ep]
        pos = {b: k for k, b in enumerate(B.bout.values[sel])}
        self.WB = [None] * len(sel)
        for b, k in pos.items():
            self.WB[k] = Ww[wb == b]


_META = {}


def meta_projection(Z, y, babies, hold, out, steps=200, seed=ev.SEED):
    """hold 아기를 뺀 아기들로, 같은 아기 안에서 원인별 무리가 잘 갈리는 선형 투영을 학습(prototypical loss).
    같은 (특징, 라벨, 뺀 아기, 차원)이면 이력 조건이 달라도 결과가 같아서 캐시한다."""
    key = (id(Z), hash(y.tobytes()), hold, out)
    if key in _META:
        return _META[key]
    import torch
    g = torch.Generator().manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(Z.shape[1], out, generator=g) * 0.1)
    opt = torch.optim.Adam([W], lr=1e-2, weight_decay=1e-4)
    data = []
    for b in np.unique(babies):
        if b == hold:
            continue
        ix = np.where(babies == b)[0]
        Zb = Z[ix] - Z[ix].mean(0)
        data.append((torch.as_tensor(Zb, dtype=torch.float32), torch.as_tensor(y[ix], dtype=torch.long)))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        losses = []
        for Zb, yb in data:
            perm = torch.as_tensor(rng.permutation(len(yb)), dtype=torch.long)
            s, q = perm[: len(yb) // 2], perm[len(yb) // 2:]
            ys, yq = yb[s], yb[q]
            cls = [k for k in range(3) if bool((ys == k).any())]
            if len(cls) < 2:
                continue
            keep = torch.stack([yq == k for k in cls]).any(0)
            if not bool(keep.any()):
                continue
            P = Zb @ W
            protos = torch.stack([P[s][ys == k].mean(0) for k in cls])
            d = ((P[q][keep][:, None] - protos[None]) ** 2).sum(-1)
            tgt = torch.as_tensor([cls.index(int(v)) for v in yq[keep]], dtype=torch.long)
            losses.append(torch.nn.functional.cross_entropy(-d, tgt))
        if not losses:
            break
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    _META[key] = Z @ W.detach().numpy()
    return _META[key]


# --- 시간순 실행 ------------------------------------------------------------------------
def online(F, y, order, day, babies, method, cfg, history):
    out = {}
    Zcache = {}
    for b, idx in order.items():
        if method == "meta":
            Zb = meta_projection(F.Z[cfg["d"]], y, babies, b, cfg["out"])
        elif method == "setsim":
            Zb = F.WB
        else:
            Zb = F.Z[cfg["d"]]
        fn = {"proto": m_proto, "knn": m_knn, "lr": m_lr, "setsim": m_setsim, "meta": m_proto}[method]
        for k, i in enumerate(idx):
            past = np.array([j for j in idx[:k] if history == "all" or day[j] < day[i]], int)
            if len(past) < K_MIN:
                continue
            out[i] = fn(Zb, i, past, y, cfg)
    return out


def nested(preds, y, babies, items):
    """preds[cfg_index] = {i: prob}. 아기 반씩 나눠 한쪽에서 설정 고르고 다른 쪽에서 평가, 분할 5번 평균."""
    ub = np.unique(babies[items])
    res, chosen = [], []
    for s in range(SPLITS):
        rng = np.random.default_rng(100 + s)
        A = set(rng.choice(ub, len(ub) // 2, replace=False))
        P = {}
        for tune, test in ((A, set(ub) - A), (set(ub) - A, A)):
            ti = [i for i in items if babies[i] in tune]
            best = max(range(len(preds)), key=lambda c: ev.metrics(y[ti], np.stack([preds[c][i] for i in ti]))["bal"])
            chosen.append(best)
            for i in items:
                if babies[i] in test:
                    P[i] = preds[best][i]
        res.append(ev.metrics(y[items], np.stack([P[i] for i in items])))
    return {k: float(np.mean([r[k] for r in res])) for k in res[0]}, collections.Counter(chosen).most_common(1)[0][0]


def cmd_run(a):
    import joint_train as jt
    jt.mem_guard()
    m = ev.load_meta()
    B = ev.bout_table(m)
    sel = np.where(B.stop.isin(ev.CAUSES))[0]
    y0 = np.array([ev.CAUSES.index(c) for c in B.stop.values[sel]])
    babies = B.baby.values[sel]
    t = pd.to_datetime(B.t.values[sel])
    day = t.normalize().values
    order = {b: sorted(np.where(babies == b)[0], key=lambda i: (t[i], B.key.values[sel][i])) for b in np.unique(babies)}
    models = ENS if a.model == "ens" else [a.model]
    F = {mo: Feats(mo, m, B, sel) for mo in models}
    methods = a.methods.split(",")
    rows, keep = [], {}
    t0 = time.time()
    for kind, R in (("real", 1), ("perm", a.perms)):
        for r in range(R):
            y = y0.copy()
            if kind == "perm":
                rng = np.random.default_rng(7 + r)
                for b, idx in order.items():
                    y[idx] = rng.permutation(y[idx])
            for history in ("all", "prev_days"):
                for meth in methods:
                    preds = []
                    for cfg in GRID[meth]:
                        ps = [online(F[mo], y, order, day, babies, meth, cfg, history) for mo in models]
                        preds.append({i: np.mean([p[i] for p in ps], 0) for i in ps[0]})
                    items = np.array(sorted(preds[0]))
                    met, best = nested(preds, y, babies, items)
                    rows.append({"model": a.model, "method": meth, "history": history, "kind": kind, "r": r,
                                 "n": len(items), "cfg": str(GRID[meth][best]), **met})
                    if kind == "real":
                        keep[(meth, history)] = (preds, items)
                    print(f"    {kind} {r} {history} {meth}: 균형 {met['bal']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            print(f"  {a.model} {kind} {r} 완료 {time.time() - t0:.0f}s", flush=True)
    out_dir = os.path.join(ev.root(), "runs", a.model)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "personal4.csv"), index=False)
    # 실제 라벨 예측(설정별)을 저장: 최종 결합·학습 곡선용
    rec = []
    kpos = {i: k for b, idx in order.items() for k, i in enumerate(idx)}
    for (meth, history), (preds, items) in keep.items():
        for c, p in enumerate(preds):
            for i in items:
                rec.append({"method": meth, "history": history, "cfg": c, "i": int(i), "baby": babies[i], "y": int(y0[i]),
                            "n_past": kpos[int(i)],
                            "p0": p[i][0], "p1": p[i][1], "p2": p[i][2]})
    pd.DataFrame(rec).to_csv(os.path.join(out_dir, "personal4_preds.csv"), index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["history", "method", "kind"])[["bal", "auc"]].mean().unstack("kind").round(3).to_string())


def cmd_final(a):
    """맥락 + 이력(기존 최고)에 개선된 개인화를 더한 최종 비교와, 피드백 수에 따른 학습 곡선.
    개인화 설정은 run과 같은 방식(아기 절반에서 고르고 나머지 절반에서 평가, 분할 5번)으로 정한다."""
    m = ev.load_meta()
    B = ev.bout_table(m)
    agg = m.groupby("bout").agg(n_syl=("n_syl", "sum"), voiced=("voiced", "sum"), n_ep=("ep", "count"),
                                gap_med=("gap_med", "median"))
    B = B.join(agg, on="bout")
    sel = np.where(B.stop.isin(ev.CAUSES))[0]
    y = np.array([ev.CAUSES.index(c) for c in B.stop.values[sel]])
    babies = B.baby.values[sel]
    t = pd.to_datetime(B.t.values[sel])
    day = t.normalize().values
    order = {b: sorted(np.where(babies == b)[0], key=lambda i: (t[i], B.key.values[sel][i])) for b in np.unique(babies)}
    Pctx = ev.context_probs(B, sel, y)
    rows, curve = [], []
    rng = np.random.default_rng(ev.SEED)
    for model in a.models.split(","):
        f = os.path.join(ev.root(), "runs", model, "personal4_preds.csv")
        if not os.path.exists(f):
            continue
        D = pd.read_csv(f)
        for (meth, history), g in D.groupby(["method", "history"]):
            if meth not in ("proto", "knn", "lr"):
                continue
            preds = [{int(r.i): np.array([r.p0, r.p1, r.p2]) for r in gc.itertuples()} for _, gc in g.groupby("cfg")]
            items = np.array(sorted(preds[0]))
            prior = {}
            for i in items:
                idx = order[babies[i]]
                past = [j for j in idx[: idx.index(i)] if history == "all" or day[j] < day[i]]
                cnt = np.bincount(y[past], minlength=3)
                prior[i] = (cnt + 1) / (cnt.sum() + 3)
            ub = np.unique(babies[items])
            res = collections.defaultdict(list)
            pooled0 = None
            for sp in range(SPLITS):
                r_ = np.random.default_rng(100 + sp)
                A = set(r_.choice(ub, len(ub) // 2, replace=False))
                P = {}
                for tune, test in ((A, set(ub) - A), (set(ub) - A, A)):
                    ti = [i for i in items if babies[i] in tune]
                    best = max(range(len(preds)), key=lambda c: ev.metrics(y[ti], np.stack([preds[c][i] for i in ti]))["bal"])
                    P.update({i: preds[best][i] for i in items if babies[i] in test})
                var = {"이력만": {i: prior[i] for i in items},
                       "맥락+이력(기존 최고)": {i: Pctx[i] * prior[i] for i in items},
                       "개인화만": P,
                       "이력+개인화": {i: prior[i] * P[i] for i in items},
                       "맥락+이력+개인화": {i: Pctx[i] * prior[i] * P[i] for i in items}}
                var = {k: np.stack([v[i] / v[i].sum() for i in items]) for k, v in var.items()}
                for k, Pv in var.items():
                    res[k].append(ev.metrics(y[items], Pv))
                if sp == 0:
                    pooled0 = var
            base = pooled0["맥락+이력(기존 최고)"]
            boots = [np.concatenate([np.where(babies[items] == u)[0] for u in rng.choice(ub, len(ub))]) for _ in range(1000)]
            for k in res:
                dl = [ev.metrics(y[items][ix], pooled0[k][ix])["bal"] - ev.metrics(y[items][ix], base[ix])["bal"] for ix in boots]
                rows.append({"model": model, "method": meth, "history": history, "variant": k, "n": len(items),
                             "bal": np.mean([r["bal"] for r in res[k]]), "auc": np.mean([r["auc"] for r in res[k]]),
                             "d_lo": np.percentile(dl, 2.5), "d_hi": np.percentile(dl, 97.5)})
            if history == "all":
                npast = np.array([order[babies[i]].index(i) for i in items])
                for lo, hi in ((10, 15), (15, 20), (20, 30), (30, 999)):
                    s_ = (npast >= lo) & (npast < hi)
                    for k in ("이력만", "맥락+이력(기존 최고)", "개인화만", "맥락+이력+개인화"):
                        curve.append({"model": model, "method": meth, "variant": k,
                                      "past": f"{lo}~{hi - 1}" if hi < 999 else f"{lo}+",
                                      "n": int(s_.sum()), **ev.metrics(y[items][s_], pooled0[k][s_])})
    R = pd.DataFrame(rows)
    C = pd.DataFrame(curve)
    R.to_csv(os.path.join(ev.root(), "runs", "personal4_final.csv"), index=False)
    C.to_csv(os.path.join(ev.root(), "runs", "personal4_curve.csv"), index=False)
    print(R.round(3).to_string())


def cmd_report(a):
    base = os.path.join(ev.root(), "runs")
    D = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(base, "*", "personal4.csv"))])
    L = ["# 아기별 개인화 강화 (이전 울음 10개 이상 쌓인 뒤, 라벨=멈춘 행동)", "",
         "- 설정은 아기 절반에서 고르고 나머지 절반에서 평가(서로 바꿔서, 분할 5번 평균). 귀무는 아기 안 라벨 섞기 + 같은 절차.",
         "- 3차 기준: proto(원인별 평균 코사인) CLAP 0.470 / AST 0.440 / YAMNet 0.429, 이전 원인 빈도만 0.397.", "",
         "| 모델 | 이력 | 방법 | n | 균형 정확도 | 귀무 95% | 배고픔 AUC | 귀무 95% | 자주 고른 설정 |",
         "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for (mod, hist, meth), g in D.groupby(["model", "history", "method"]):
        re, pm = g[g.kind == "real"], g[g.kind == "perm"]
        L.append(f"| {mod} | {hist} | {meth} | {int(re.n.iloc[0])} | {re.bal.mean():.3f} | {pm.bal.quantile(.95):.3f} | "
                 f"{re.auc.mean():.3f} | {pm.auc.quantile(.95):.3f} | {re.cfg.iloc[0]} |")
    fp = os.path.join(base, "personal4_final.csv")
    if os.path.exists(fp):
        R = pd.read_csv(fp)
        L += ["", "## 최종 결합 (맥락 + 이력 + 개인화)", "",
              "Δ 구간은 맥락+이력(기존 최고) 대비 균형 정확도 차이의 아기 단위 부트스트랩 95% 구간(분할 1번 기준).", "",
              "| 모델 | 방법 | 이력 | 조합 | 균형 정확도 | 배고픔 AUC | Δ 구간 |", "|---|---|---|---|---:|---:|---|"]
        for r in R.itertuples():
            L.append(f"| {r.model} | {r.method} | {r.history} | {r.variant} | {r.bal:.3f} | {r.auc:.3f} | "
                     f"{r.d_lo:+.2f}~{r.d_hi:+.2f} |")
    cp = os.path.join(base, "personal4_curve.csv")
    if os.path.exists(cp):
        C = pd.read_csv(cp)
        pasts = list(dict.fromkeys(C.past))
        L += ["", "## 학습 곡선: 그 아기의 이전 울음(피드백) 수별 균형 정확도", "",
              "| 모델 | 방법 | 조합 | " + " | ".join(f"이전 {p}개" for p in pasts) + " |",
              "|---|---|---|" + "---:|" * len(pasts)]
        for (mod, meth, var), g in C.groupby(["model", "method", "variant"], sort=False):
            v = g.set_index("past")
            L.append(f"| {mod} | {meth} | {var} | " + " | ".join(f"{v.loc[p, 'bal']:.3f} (n={int(v.loc[p, 'n'])})"
                                                                  for p in pasts) + " |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(ev.root(), "PERSONAL4_REPORT.md"), "w", encoding="utf-8").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--model", required=True)
    p.add_argument("--methods", default="proto,knn,lr,setsim,meta")
    p.add_argument("--perms", type=int, default=20)
    p = sp.add_parser("final")
    p.add_argument("--models", default="yamnet,clap,ast,ens")
    sp.add_parser("report")
    a = ap.parse_args()
    {"run": cmd_run, "final": cmd_final, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
