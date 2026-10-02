"""여러 울음 데이터를 섞어 대형 사전학습 오디오 모델로 울음 이유를 학습·평가한다(GPU 서버, PyTorch).

    python joint_train.py convert                        # .bin만 있는 모델을 safetensors 사본으로(torch<2.6 대응)
    python joint_train.py folds                          # 데이터별 그룹 분리 5겹 배정(folds.csv)
    python joint_train.py extract --model wavlm_l        # 동결 모델의 층별 임베딩 추출
    python joint_train.py probe --model wavlm_l          # 층 가중합 선형 프로브: 단일 vs 혼합, LODO, 출처 예측
    python joint_train.py robust --model wavlm_l         # 겹 분할 반복 + 라벨 섞은 귀무 분포 비교
    python joint_train.py finetune --model ast --protocol cv --cond single,mixed,dann
    python joint_train.py finetune --model ast --protocol lodo --cond mixed,dann
    python joint_train.py report                         # 모든 결과 → runs/JOINT_REPORT.md

입력: CRY_JOINT(기본 ~/cry/data/joint)/meta.csv + wav/ (joint_build.py가 로컬에서 만든 것)

평가 원칙
  - cv: 데이터마다 그룹(DAC 업로더, Enes 아기, Baidu 유사 중복 묶음)을 5겹으로 나눈다. 같은 겹을 평가로 두고
        single(그 데이터만 학습) / mixed(세 데이터 학습 겹을 모두 섞음) / dann(섞되 출처 판별을 방해)을 비교한다.
  - lodo: 한 데이터를 통째로 빼고 나머지로 학습 → 뺀 데이터로 평가(녹음 환경·라벨 기준이 다른 곳으로 옮겨 쓰기).
  - 데이터마다 가진 클래스가 달라서, 학습 손실과 예측 모두 그 데이터의 클래스만 남기고 마스킹한다.
  - 지표: 균형 정확도(무작위 = 1/클래스 수), Macro-F1, 배고픔 vs 나머지 AUC. 95% 구간은 그룹 단위 부트스트랩.
"""
import argparse
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.environ.get("CRY_JOINT", os.path.expanduser("~/cry/data/joint"))
RUNS = os.path.join(os.path.dirname(ROOT), "runs")
CLASSES = ["hunger", "discomfort", "isolation", "sleepy"]
DATASETS = ["dac", "enes", "baidu"]
DS_CLASSES = {"dac": ["hunger", "discomfort", "sleepy"],
              "enes": ["hunger", "discomfort", "isolation"],
              "baidu": CLASSES}
SR, SEED, N_FOLDS = 16000, 42, 5
MODELS = {  # 키: (허깅페이스 이름, 종류)
    "ast": ("MIT/ast-finetuned-audioset-10-10-0.4593", "ast"),
    "wavlm_bp": ("microsoft/wavlm-base-plus", "ssl"),
    "wavlm_l": ("microsoft/wavlm-large", "ssl"),
    "hubert_l": ("facebook/hubert-large-ll60k", "ssl"),
    "whisper_s": ("openai/whisper-small", "whisper"),
    "clap": ("laion/clap-htsat-unfused", "clap"),
}
DEV = "cuda"
# pytorch_model.bin만 있는 모델은 torch<2.6에서 transformers가 로드를 막는다(CVE-2025-32434)
# → safetensors로 변환해 둔 로컬 사본(CRY_MODELS/<org>__<name>)이 있으면 그걸 쓴다.
LOCAL_MODELS = os.path.expanduser(os.environ.get("CRY_MODELS", "~/cry/models"))


def src(key):
    p = os.path.join(LOCAL_MODELS, MODELS[key][0].replace("/", "__"))
    return p if os.path.isdir(p) else MODELS[key][0]


def cmd_convert(_):
    """pytorch_model.bin만 있는 모델을 safetensors 로컬 사본으로 바꾼다(torch<2.6이면 transformers가 .bin 로드를 막음)."""
    import shutil
    from huggingface_hub import HfApi, snapshot_download
    from safetensors.torch import save_file
    for name, _ in MODELS.values():
        dst = os.path.join(LOCAL_MODELS, name.replace("/", "__"))
        if os.path.isdir(dst) or "model.safetensors" in HfApi().list_repo_files(name):
            continue
        sdir = snapshot_download(name, allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(sdir, "*.json")) + glob.glob(os.path.join(sdir, "*.txt")):
            shutil.copy(f, dst)
        sd = torch.load(os.path.join(sdir, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        save_file({k: v.contiguous().clone() for k, v in sd.items()}, os.path.join(dst, "model.safetensors"),
                  metadata={"format": "pt"})
        print("변환:", name, "→", dst)


# --- 데이터 ---------------------------------------------------------------------
def load_meta():
    m = pd.read_csv(os.path.join(ROOT, "meta.csv"))
    f = os.path.join(ROOT, "folds.csv")
    if os.path.exists(f):
        m = m.merge(pd.read_csv(f), on="id")
    m["y"] = m.label.map(CLASSES.index)
    m["group"] = m.dataset + ":" + m.group.astype(str)
    return m.reset_index(drop=True)


_WAV = {}


def wav(path):
    if path not in _WAV:
        x, _ = sf.read(os.path.join(ROOT, path), dtype="int16")
        _WAV[path] = x
    return _WAV[path].astype(np.float32) / 32768.0


def ds_mask(datasets):
    M = np.zeros((len(datasets), len(CLASSES)), bool)
    for i, d in enumerate(datasets):
        M[i, [CLASSES.index(c) for c in DS_CLASSES[d]]] = True
    return M


def balance_weights(m):
    """데이터마다 같은 비중, 데이터 안에서는 클래스 균형."""
    w = np.zeros(len(m))
    dss = m.dataset.unique()
    for d in dss:
        sub = (m.dataset == d).values
        cnt = m[sub].label.value_counts()
        for lab, c in cnt.items():
            w[sub & (m.label == lab).values] = 1.0 / (len(dss) * len(cnt) * c)
    return w / w.mean()


def windows(x, win, hop):
    W, H = int(win * SR), int(hop * SR)
    if len(x) <= W:
        return [x]
    st = list(range(0, len(x) - W + 1, H))
    if st[-1] + W < len(x):
        st.append(len(x) - W)
    return [x[s:s + W] for s in st]


def tile(x, n):
    return np.tile(x, int(math.ceil(n / max(len(x), 1))))[:n] if len(x) < n else x[:n]


def make_folds(m, seed=SEED):
    from sklearn.model_selection import StratifiedGroupKFold
    fold = np.zeros(len(m), int)
    for d in DATASETS:
        idx = np.where(m.dataset == d)[0]
        sgk = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
        for k, (_, te) in enumerate(sgk.split(idx, m.label.values[idx], m.group.values[idx])):
            fold[idx[te]] = k
    return fold


def cmd_folds(_):
    m = load_meta()
    fold = make_folds(m)
    pd.DataFrame({"id": m.id, "fold": fold}).to_csv(os.path.join(ROOT, "folds.csv"), index=False)
    m["fold"] = fold
    print(pd.crosstab([m.dataset, m.label], m.fold))


# --- 지표 -----------------------------------------------------------------------
def _ds_metric(y, P, groups, ds, n_boot=1000):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    cls = [CLASSES.index(c) for c in DS_CLASSES[ds]]
    yy = np.array([cls.index(v) for v in y])
    Pc = P[:, cls] / P[:, cls].sum(1, keepdims=True)
    h = cls.index(CLASSES.index("hunger"))

    def calc(ix):
        pred = Pc[ix].argmax(1)
        bal = balanced_accuracy_score(yy[ix], pred)
        f1 = f1_score(yy[ix], pred, average="macro", labels=range(len(cls)), zero_division=0)
        yh = yy[ix] == h
        auc = roc_auc_score(yh, Pc[ix, h]) if 0 < yh.sum() < len(yh) else np.nan
        return bal, f1, auc
    point = calc(np.arange(len(yy)))
    if not n_boot:
        return {"n": len(yy), "k": len(cls), "bal": point[0], "f1": point[1], "auc": point[2]}
    rng = np.random.default_rng(SEED)
    ug = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in ug}
    boots = []
    for _ in range(n_boot):
        ix = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
        boots.append(calc(ix))
    lo, hi = np.nanpercentile(np.array(boots), [2.5, 97.5], axis=0)
    return {"n": len(yy), "k": len(cls), "bal": point[0], "bal_lo": lo[0], "bal_hi": hi[0], "f1": point[1],
            "auc": point[2], "auc_lo": lo[2], "auc_hi": hi[2]}


def pred_frame(m_te, P, **tags):
    df = pd.DataFrame({"id": m_te.id.values, "dataset": m_te.dataset.values, "group": m_te.group.values,
                       "y": m_te.y.values})
    for c in range(len(CLASSES)):
        df[f"p{c}"] = P[:, c]
    for k, v in tags.items():
        df[k] = v
    return df


def masked_probs(logits, datasets):
    M = torch.as_tensor(ds_mask(datasets), device=logits.device)
    return torch.softmax(logits.float().masked_fill(~M, -1e4), -1)


# --- 1단계: 동결 임베딩 ----------------------------------------------------------
def _embedder(key):
    from transformers import AutoFeatureExtractor, AutoModel
    name, kind = src(key), MODELS[key][1]
    if kind == "ssl":
        fe, net = AutoFeatureExtractor.from_pretrained(name), AutoModel.from_pretrained(name)
        net = net.to(DEV).eval()

        def emb(w):
            x = torch.as_tensor(fe(w, sampling_rate=SR, return_tensors="np")["input_values"]).to(DEV)
            hs = net(x, output_hidden_states=True).hidden_states
            return torch.stack([h[0].mean(0) for h in hs])
        return emb, 10.0
    if kind == "ast":
        from transformers import ASTFeatureExtractor, ASTModel
        fe, net = ASTFeatureExtractor.from_pretrained(name), ASTModel.from_pretrained(name).to(DEV).eval()

        def emb(w):
            x = torch.as_tensor(fe(w, sampling_rate=SR, return_tensors="np")["input_values"]).to(DEV)
            hs = net(x, output_hidden_states=True).hidden_states
            return torch.stack([h[0, :2].mean(0) for h in hs])     # CLS·DIST 토큰
        return emb, 10.0
    if kind == "whisper":
        from transformers import WhisperFeatureExtractor, WhisperModel
        fe = WhisperFeatureExtractor.from_pretrained(name)
        enc = WhisperModel.from_pretrained(name).encoder.to(DEV).eval()

        def emb(w):
            x = torch.as_tensor(fe(w, sampling_rate=SR, return_tensors="np")["input_features"]).to(DEV)
            hs = enc(x, output_hidden_states=True).hidden_states
            T = max(1, min(hs[0].shape[1], int(math.ceil(len(w) / SR * 50))))   # 30초 패딩 제외
            return torch.stack([h[0, :T].mean(0) for h in hs])
        return emb, 30.0
    if kind == "clap":
        import torchaudio
        from transformers import ClapFeatureExtractor, ClapModel
        fe, net = ClapFeatureExtractor.from_pretrained(name), ClapModel.from_pretrained(name).to(DEV).eval()

        def emb(w):
            w48 = torchaudio.functional.resample(torch.from_numpy(w), SR, 48000).numpy()
            inp = {k: torch.as_tensor(v).to(DEV) for k, v in
                   fe(w48, sampling_rate=48000, return_tensors="np").items()}
            e = net.get_audio_features(**inp)
            e = e if torch.is_tensor(e) else e.pooler_output
            return e[0][None]
        return emb, 10.0
    raise ValueError(key)


@torch.no_grad()
def cmd_extract(a):
    m = load_meta()
    out = os.path.join(RUNS, "feats", f"{a.model}.npz")
    if os.path.exists(out) and not a.force:
        print("이미 있음:", out)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    emb, win = _embedder(a.model)
    feats, t0 = [], time.time()
    for i, r in enumerate(m.itertuples()):
        E = []
        for w in windows(wav(r.path), win, win / 2):
            if len(w) < SR // 2:
                w = tile(w, SR // 2)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                E.append(emb(w).float())
        feats.append(torch.stack(E).mean(0).cpu().numpy().astype(np.float16))
        if (i + 1) % 200 == 0:
            print(f"  {a.model} {i + 1}/{len(m)} {time.time() - t0:.0f}s", flush=True)
    np.savez(out, ids=m.id.values, X=np.stack(feats))
    print(f"{a.model}: {np.stack(feats).shape} → {out} ({time.time() - t0:.0f}s)")


# --- 1단계: 층 가중합 선형 프로브 -------------------------------------------------
class Probe(nn.Module):
    def __init__(self, L, D, n_out):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(L))
        self.fc = nn.Linear(D, n_out)

    def forward(self, X):
        return self.fc((X * torch.softmax(self.a, 0)[None, :, None]).sum(1))


def fit_probe(X, y, M, w, n_out=len(CLASSES), C=0.1, steps=600):
    """sklearn LogisticRegression(C)와 같은 세기의 L2. X: [N, L, D] 표준화 전."""
    X = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-5
    X = (X - mu) / sd
    y = torch.as_tensor(y, device=DEV)
    M = torch.as_tensor(M, device=DEV)
    w = torch.as_tensor(w, dtype=torch.float32, device=DEV)
    torch.manual_seed(SEED)
    net = Probe(X.shape[1], X.shape[2], n_out).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    lam = 1.0 / (2 * C * len(y))
    for _ in range(steps):
        logits = net(X).masked_fill(~M, -1e4)
        loss = (F.cross_entropy(logits, y, reduction="none") * w).mean() + lam * net.fc.weight.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return lambda Z: net((torch.as_tensor(Z, dtype=torch.float32, device=DEV) - mu) / sd).detach()


def cmd_probe(a):
    m = load_meta()
    z = np.load(os.path.join(RUNS, "feats", f"{a.model}.npz"), allow_pickle=True)
    assert (z["ids"] == m.id.values).all()
    X = z["X"].astype(np.float32)
    out_dir = os.path.join(RUNS, "probe", a.model)
    os.makedirs(out_dir, exist_ok=True)
    preds, layer_w = [], {}

    def run(tr, te, **tags):
        mt = m.iloc[tr]
        f = fit_probe(X[tr], mt.y.values, ds_mask(mt.dataset.values), balance_weights(mt))
        P = masked_probs(f(X[te]), m.dataset.values[te]).cpu().numpy()
        preds.append(pred_frame(m.iloc[te], P, stage="probe", model=a.model, **tags))

    for k in range(N_FOLDS):
        te = np.where(m.fold == k)[0]
        for d in DATASETS:
            run(np.where((m.fold != k) & (m.dataset == d))[0], te[m.dataset.values[te] == d],
                protocol="cv", cond="single", fold=k)
        run(np.where(m.fold != k)[0], te, protocol="cv", cond="mixed", fold=k)
    for d in DATASETS:                      # 한 데이터 통째로 빼기 + 한 데이터만으로 옮기기
        run(np.where(m.dataset != d)[0], np.where(m.dataset == d)[0], protocol="lodo", cond="mixed", fold=-1)
        for s in DATASETS:
            if s != d:
                run(np.where(m.dataset == s)[0], np.where(m.dataset == d)[0],
                    protocol="lodo", cond=f"from_{s}", fold=-1)
    df = pd.concat(preds)
    df.to_csv(os.path.join(out_dir, "preds.csv"), index=False)

    # 출처(데이터셋) 예측: 지름길이 얼마나 쉬운지
    yd = m.dataset.map(DATASETS.index).values
    acc = []
    for k in range(N_FOLDS):
        tr, te = np.where(m.fold != k)[0], np.where(m.fold == k)[0]
        f = fit_probe(X[tr], yd[tr], np.ones((len(tr), 3), bool), np.ones(len(tr)), n_out=3)
        acc.append((f(X[te]).argmax(1).cpu().numpy() == yd[te]).mean())
    with open(os.path.join(out_dir, "source_acc.txt"), "w") as fh:
        fh.write(f"{np.mean(acc):.4f}\n")
    print(f"{a.model}: 출처 예측 정확도 {np.mean(acc):.3f}")
    summarize(df, title=f"probe {a.model}")


def cmd_robust(a):
    """겹 분할을 바꿔 반복(repeat)하고, 라벨을 데이터 안에서 섞은 귀무 분포(perm)와 비교한다.
    모델 6종 × 조건 중 최고값을 고르면 우연히 높게 나올 수 있어서, 같은 절차를 무작위 라벨에 돌려 기준을 잡는다."""
    m = load_meta()
    z = np.load(os.path.join(RUNS, "feats", f"{a.model}.npz"), allow_pickle=True)
    X = z["X"].astype(np.float32)
    rows = []

    def cv_eval(fold, y, kind, r):
        P = {c: np.zeros((len(m), len(CLASSES))) for c in ["single", "mixed"]}
        for k in range(N_FOLDS):
            te = np.where(fold == k)[0]
            for d in DATASETS:
                tr = np.where((fold != k) & (m.dataset.values == d))[0]
                ted = te[m.dataset.values[te] == d]
                f = fit_probe(X[tr], y[tr], ds_mask(m.dataset.values[tr]), balance_weights(m.iloc[tr]))
                P["single"][ted] = masked_probs(f(X[ted]), m.dataset.values[ted]).cpu().numpy()
            tr = np.where(fold != k)[0]
            f = fit_probe(X[tr], y[tr], ds_mask(m.dataset.values[tr]), balance_weights(m.iloc[tr]))
            P["mixed"][te] = masked_probs(f(X[te]), m.dataset.values[te]).cpu().numpy()
        for c, PP in P.items():
            for d in DATASETS:
                ix = np.where(m.dataset.values == d)[0]
                r_ = _ds_metric(y[ix], PP[ix], m.group.values[ix], d, n_boot=0)
                rows.append({"model": a.model, "kind": kind, "r": r, "cond": c, "test": d, **r_})
    for r in range(a.repeats):
        cv_eval(make_folds(m, SEED + 1 + r), m.y.values, "repeat", r)
    base_fold = m.fold.values
    for r in range(a.perms):
        rng = np.random.default_rng(1000 + r)
        y = m.y.values.copy()
        for d in DATASETS:
            ix = np.where(m.dataset.values == d)[0]
            y[ix] = rng.permutation(y[ix])
        cv_eval(base_fold, y, "perm", r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RUNS, "probe", a.model, "robust.csv"), index=False)
    print(df.groupby(["kind", "cond", "test"])[["bal", "auc"]].agg(["mean", "std", "max"]).round(3))


# --- 2단계: 전체 미세조정 ---------------------------------------------------------
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class FTNet(nn.Module):
    def __init__(self, key):
        super().__init__()
        from transformers import AutoModel, ASTModel
        name, kind = src(key), MODELS[key][1]
        self.kind = kind
        if kind == "ast":
            self.backbone = ASTModel.from_pretrained(name)
        else:
            self.backbone = AutoModel.from_pretrained(name)
            self.backbone.freeze_feature_encoder()
        D = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, len(CLASSES)))
        self.dom = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, len(DATASETS)))

    def forward(self, x, lam=0.0):
        o = self.backbone(x)
        h = o.pooler_output if self.kind == "ast" else o.last_hidden_state.mean(1)
        return self.head(h), self.dom(GRL.apply(h, lam))


class FTData(torch.utils.data.Dataset):
    """학습: 무작위 CROP초 구간(짧으면 반복해 채움) + 음량·잡음 증강. 평가: CROP/2 간격 창 목록."""
    CROP = 8.0

    def __init__(self, m, key, train):
        from transformers import AutoFeatureExtractor
        self.m, self.train, self.kind = m.reset_index(drop=True), train, MODELS[key][1]
        self.fe = AutoFeatureExtractor.from_pretrained(src(key))
        self.n = int(self.CROP * SR)
        if train:
            self.items = list(range(len(self.m)))
        else:
            self.items = [(i, s) for i, r in enumerate(self.m.itertuples())
                          for s in range(len(windows(wav(r.path), self.CROP, self.CROP / 2)))]

    def __len__(self):
        return len(self.items)

    def _feat(self, w):
        return torch.as_tensor(self.fe(w, sampling_rate=SR, return_tensors="np")["input_values"][0])

    def __getitem__(self, j):
        rng = np.random.default_rng()
        if self.train:
            i = self.items[j]
            x = wav(self.m.path[i])
            if len(x) > self.n:
                s = rng.integers(0, len(x) - self.n + 1)
                x = x[s:s + self.n]
            x = tile(x, self.n) * 10 ** (rng.uniform(-6, 6) / 20)
            if rng.random() < 0.5:                     # 백색 잡음 SNR 15~40dB
                p = np.mean(x ** 2) + 1e-8
                x = x + rng.normal(0, np.sqrt(p / 10 ** (rng.uniform(15, 40) / 10)), len(x)).astype(np.float32)
            f = self._feat(np.clip(x, -1, 1).astype(np.float32))
            if self.kind == "ast":                     # SpecAugment
                for _ in range(2):
                    t0, tw = rng.integers(0, f.shape[0]), rng.integers(0, 80)
                    f[t0:t0 + tw] = 0
                    f0, fw = rng.integers(0, f.shape[1]), rng.integers(0, 24)
                    f[:, f0:f0 + fw] = 0
        else:
            i, s = self.items[j]
            x = windows(wav(self.m.path[i]), self.CROP, self.CROP / 2)[s]
            f = self._feat(tile(x, self.n))
        r = self.m.iloc[i]
        return f, int(r.y), torch.as_tensor(ds_mask([r.dataset])[0]), DATASETS.index(r.dataset), i


def finetune(key, m_tr, m_te, dann, epochs, log):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for p in pd.concat([m_tr, m_te]).path:     # 워커가 fork 전에 공유하도록 미리 적재
        wav(p)
    net = FTNet(key).to(DEV)
    bs = 8 if key.endswith("_l") else 16
    tr = FTData(m_tr, key, True)
    sampler = torch.utils.data.WeightedRandomSampler(balance_weights(tr.m), len(tr), replacement=True)
    dl = torch.utils.data.DataLoader(tr, bs, sampler=sampler, num_workers=8, drop_last=True,
                                     persistent_workers=True)
    lr_bb = 2e-5 if MODELS[key][1] == "ast" else 3e-5
    opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": lr_bb},
                             {"params": list(net.head.parameters()) + list(net.dom.parameters()), "lr": 1e-3}],
                            weight_decay=0.01)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, 0.1 * total)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    step = 0
    for ep in range(epochs):
        net.train()
        tl, td, t0 = 0.0, 0.0, time.time()
        for f, y, M, dom, _ in dl:
            f, y, M, dom = f.to(DEV), y.to(DEV), M.to(DEV), dom.to(DEV)
            lam = (2 / (1 + math.exp(-10 * step / total)) - 1) * 0.5 if dann else 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lc, ld = net(f, lam)
            loss = F.cross_entropy(lc.float().masked_fill(~M, -1e4), y)
            dl_loss = F.cross_entropy(ld.float(), dom)
            if dann:
                loss = loss + dl_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tl += loss.item()
            td += dl_loss.item()
        log(f"    ep{ep + 1}/{epochs} loss {tl / len(dl):.3f} dom {td / len(dl):.3f} {time.time() - t0:.0f}s")
    # 평가: 창별 로짓 평균
    net.eval()
    te = FTData(m_te, key, False)
    L = np.zeros((len(te.m), len(CLASSES)))
    cnt = np.zeros(len(te.m))
    dl = torch.utils.data.DataLoader(te, bs, num_workers=8)
    with torch.no_grad():
        for f, _, _, _, i in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lc, _ = net(f.to(DEV))
            np.add.at(L, i.numpy(), lc.float().cpu().numpy())
            np.add.at(cnt, i.numpy(), 1)
    L = torch.as_tensor(L / cnt[:, None])
    del net
    torch.cuda.empty_cache()
    return masked_probs(L, te.m.dataset.values).numpy()


def cmd_finetune(a):
    m = load_meta()
    out_dir = os.path.join(RUNS, "ft", a.model)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "train.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    jobs = []
    for cond in a.cond.split(","):
        if a.protocol == "cv":
            for k in range(N_FOLDS):
                te = m[m.fold == k]
                if cond == "single":
                    for d in DATASETS:
                        jobs.append((f"cv_single_{d}_f{k}", m[(m.fold != k) & (m.dataset == d)],
                                     te[te.dataset == d], cond, k))
                else:
                    jobs.append((f"cv_{cond}_f{k}", m[m.fold != k], te, cond, k))
        else:
            for d in DATASETS:
                jobs.append((f"lodo_{cond}_{d}", m[m.dataset != d], m[m.dataset == d], cond, -1))
    for tag, m_tr, m_te, cond, k in jobs:
        p = os.path.join(out_dir, tag + ".csv")
        if os.path.exists(p):
            continue
        log(f"[{time.strftime('%H:%M:%S')}] {a.model} {tag}: 학습 {len(m_tr)} / 평가 {len(m_te)}")
        P = finetune(a.model, m_tr, m_te, cond == "dann", a.epochs, log)
        pred_frame(m_te.reset_index(drop=True), P, stage="ft", model=a.model, protocol=a.protocol,
                   cond=cond, fold=k).to_csv(p, index=False)


# --- 보고서 -----------------------------------------------------------------------
def summarize(df, title=""):
    rows = []
    for (stage, model, prot, cond, d), g in df.groupby(["stage", "model", "protocol", "cond", "dataset"]):
        P = g[[f"p{c}" for c in range(len(CLASSES))]].values
        r = _ds_metric(g.y.values, P, g.group.values, d)
        rows.append({"stage": stage, "model": model, "protocol": prot, "cond": cond, "test": d, **r})
    s = pd.DataFrame(rows)
    if title:
        print(f"== {title}")
        print(s[["protocol", "cond", "test", "n", "bal", "bal_lo", "bal_hi", "f1", "auc"]].round(3).to_string())
    return s


def paired_delta(df, base="single", n_boot=1000):
    """같은 평가 클립에서 조건 간 균형 정확도·배고픔 AUC 차이(조건 − base), 그룹 단위 짝 부트스트랩 95% 구간."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rows = []
    cv = df[df.protocol == "cv"]
    for (stage, model, d), g in cv.groupby(["stage", "model", "dataset"]):
        if base not in set(g.cond):
            continue
        cls = [CLASSES.index(c) for c in DS_CLASSES[d]]
        h = cls.index(CLASSES.index("hunger"))
        piv = {c: gg.set_index("id").sort_index() for c, gg in g.groupby("cond")}
        b = piv[base]
        yy = np.array([cls.index(v) for v in b.y])
        groups = b.group.values
        ug = np.unique(groups)
        gi = {u: np.where(groups == u)[0] for u in ug}

        def score(P, ix):
            Pc = P[ix][:, cls]
            return balanced_accuracy_score(yy[ix], Pc.argmax(1)), roc_auc_score(yy[ix] == h, Pc[:, h])
        Pb = b[[f"p{c}" for c in range(len(CLASSES))]].values
        for cond, o in piv.items():
            if cond == base or len(o) != len(b):
                continue
            Po = o.loc[b.index][[f"p{c}" for c in range(len(CLASSES))]].values
            full = np.arange(len(yy))
            d0 = np.subtract(score(Po, full), score(Pb, full))
            rng = np.random.default_rng(SEED)
            bs = []
            for _ in range(n_boot):
                ix = np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
                bs.append(np.subtract(score(Po, ix), score(Pb, ix)))
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            rows.append({"stage": stage, "model": model, "test": d, "cond": cond, "d_bal": d0[0],
                         "d_bal_lo": lo[0], "d_bal_hi": hi[0], "d_auc": d0[1], "d_auc_lo": lo[1], "d_auc_hi": hi[1]})
    return pd.DataFrame(rows)


def cmd_report(_):
    parts = [pd.read_csv(p) for p in glob.glob(os.path.join(RUNS, "probe", "*", "preds.csv"))]
    parts += [pd.read_csv(p) for p in glob.glob(os.path.join(RUNS, "ft", "*", "*.csv"))]
    df = pd.concat(parts)
    s = summarize(df)
    s.to_csv(os.path.join(RUNS, "joint_summary.csv"), index=False)
    src = {os.path.basename(os.path.dirname(p)): float(open(p).read())
           for p in glob.glob(os.path.join(RUNS, "probe", "*", "source_acc.txt"))}
    m = load_meta()
    L = ["# 데이터 혼합 + 대형 사전학습 모델 울음 이유 실험", "",
         f"- 데이터: {len(m)}개 클립, {m.dur.sum() / 3600:.1f}시간. "
         + ", ".join(f"{d} {(m.dataset == d).sum()}개" for d in DATASETS),
         "- 통합 라벨: hunger / discomfort / isolation(안아주길 원함) / sleepy. 데이터마다 없는 클래스는 마스킹.",
         "- 균형 정확도 무작위 기준: DAC·Enes 0.333 (3클래스), Baidu 0.25 (4클래스). AUC 무작위 0.5.",
         "- 괄호는 그룹(아기·업로더·유사 중복 묶음) 단위 부트스트랩 95% 구간.", ""]
    if src:
        L += ["## 출처(데이터셋) 예측 정확도 — 소리만으로 어느 데이터인지 맞히는 정도", ""]
        L += [f"- {k}: {v:.3f}" for k, v in sorted(src.items())]
        L += ["", "출처가 쉽게 구분될수록 섞어서 학습할 때 '녹음 환경 = 원인' 지름길에 빠지기 쉽다.", ""]
    fmt = lambda r: (f"{r.bal:.3f} ({r.bal_lo:.2f}~{r.bal_hi:.2f}) | {r.f1:.3f} | "
                     + (f"{r.auc:.3f} ({r.auc_lo:.2f}~{r.auc_hi:.2f})" if not np.isnan(r.auc) else "-"))
    for prot, head in [("cv", "## 같은 데이터 안 평가(그룹 분리 5겹): 단일 vs 혼합"),
                       ("lodo", "## 데이터 통째로 빼고 평가(LODO, 다른 환경으로 옮기기)")]:
        sub = s[s.protocol == prot]
        if not len(sub):
            continue
        L += [head, "", "| 단계 | 모델 | 조건 | 평가 데이터 | n | 균형 정확도 | Macro-F1 | 배고픔 AUC |",
              "|---|---|---|---|---|---|---|---|"]
        for r in sub.sort_values(["test", "stage", "model", "cond"]).itertuples():
            L.append(f"| {r.stage} | {r.model} | {r.cond} | {r.test} | {r.n} | {fmt(r)} |")
        L.append("")
    dl = paired_delta(df)
    if len(dl):
        L += ["## 섞으면 나아지나: 같은 평가 클립에서 조건 − single 차이", "",
              "| 단계 | 모델 | 평가 데이터 | 조건 | Δ균형 정확도 | Δ배고픔 AUC |", "|---|---|---|---|---|---|"]
        for r in dl.sort_values(["test", "stage", "model", "cond"]).itertuples():
            L.append(f"| {r.stage} | {r.model} | {r.test} | {r.cond} | {r.d_bal:+.3f} ({r.d_bal_lo:+.2f}~{r.d_bal_hi:+.2f}) "
                     f"| {r.d_auc:+.3f} ({r.d_auc_lo:+.2f}~{r.d_auc_hi:+.2f}) |")
        L += ["", "구간이 0을 포함하면 차이가 있다고 말할 수 없다.", ""]
    with open(os.path.join(RUNS, "JOINT_REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("convert")
    sp.add_parser("folds")
    for name in ["extract", "probe"]:
        p = sp.add_parser(name)
        p.add_argument("--model", required=True, choices=list(MODELS))
        p.add_argument("--force", action="store_true")
    p = sp.add_parser("robust")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--perms", type=int, default=5)
    p = sp.add_parser("finetune")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--protocol", choices=["cv", "lodo"], default="cv")
    p.add_argument("--cond", default="single,mixed,dann")
    p.add_argument("--epochs", type=int, default=10)
    sp.add_parser("report")
    a = ap.parse_args()
    {"convert": cmd_convert, "folds": cmd_folds, "extract": cmd_extract, "probe": cmd_probe,
     "robust": cmd_robust, "finetune": cmd_finetune, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
