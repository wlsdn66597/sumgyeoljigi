"""여러 울음 데이터를 섞어 대형 사전학습 오디오 모델로 울음 이유를 학습·평가한다(GPU 서버, PyTorch).

    python joint_train.py convert                        # .bin만 있는 모델을 safetensors 사본으로(torch<2.6 대응)
    python joint_train.py folds                          # 데이터별 그룹 분리 5겹 배정(folds.csv)
    python joint_train.py extract --model wavlm_l        # 동결 모델의 층별 임베딩 추출
    python joint_train.py probe --model wavlm_l          # 층 가중합 선형 프로브: 단일 vs 혼합, LODO, 출처 예측
    python joint_train.py robust --model wavlm_l         # 겹 분할 반복 + 라벨 섞은 귀무 분포 비교
    python joint_train.py robust --model wavlm_l --norm baby   # 아기(그룹)별 평균을 빼고 같은 비교
    python joint_train.py finetune --model ast --protocol cv --cond single,mixed,dann
    python joint_train.py finetune --model ast --protocol lodo --cond mixed,dann
    python joint_train.py report                         # 모든 결과 → runs/JOINT_REPORT.md

2차(데이터 확장) 추가 명령 — CRY_JOINT=~/cry/data/joint2 CRY_RUNS=~/cry/data/runs2 로 따로 돌린다
    python joint_train.py extract --model voc2vec --windows   # 창(4초)별 층 임베딩 저장(시간 구조 모델용)
    python joint_train.py temporal --model voc2vec            # 창 순서를 보는 트랜스포머 vs 창 평균
    python joint_train.py dapt --model voc2vec                # 라벨 없는 울음으로 wav2vec2 추가 사전학습
    python joint_train.py finetune --model voc2vec --cond single,mixed,badv --aug --pool wsum --lr 1e-5 --tag v2

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
RUNS = os.environ.get("CRY_RUNS", os.path.join(os.path.dirname(ROOT), "runs"))
CLASSES = ["hunger", "discomfort", "isolation", "sleepy", "pain"]
ALL_DATASETS = ["dac", "enes", "baidu", "corvin"]
DATASETS = list(ALL_DATASETS)      # load_meta()가 meta에 있는 것만 남긴다
DS_CLASSES = {"dac": ["hunger", "discomfort", "sleepy"],
              "enes": ["hunger", "discomfort", "isolation"],
              "baidu": ["hunger", "discomfort", "isolation", "sleepy"],
              "corvin": ["discomfort", "pain"]}
SR, SEED, N_FOLDS = 16000, 42, 5
MODELS = {  # 키: (허깅페이스 이름, 종류)
    "ast": ("MIT/ast-finetuned-audioset-10-10-0.4593", "ast"),
    "wavlm_bp": ("microsoft/wavlm-base-plus", "ssl"),
    "wavlm_l": ("microsoft/wavlm-large", "ssl"),
    "hubert_l": ("facebook/hubert-large-ll60k", "ssl"),
    "whisper_s": ("openai/whisper-small", "whisper"),
    "clap": ("laion/clap-htsat-unfused", "clap"),
    # 2차: 발성 전용·Bonafos 2025에서 쓴 계열
    "voc2vec": ("alkiskoudounas/voc2vec", "ssl"),                 # 비언어 발성(아기 울음 포함) 125시간 wav2vec2
    "voc2vec_hubert": ("alkiskoudounas/voc2vec-hubert-ls-pt", "ssl"),
    "w2v2_l": ("facebook/wav2vec2-large-lv60", "ssl"),
    "unispeech_l": ("microsoft/unispeech-large-1500h-cv", "ssl"),  # Bonafos 2025에서 아기 식별 최고
    "whisper_l": ("openai/whisper-large-v3", "whisper"),
    "voc2vec_dapt": ("local/voc2vec_dapt", "ssl"),                # dapt 명령으로 만든 울음 추가 사전학습판
}
DEV = "cuda"
WORKERS = int(os.environ.get("CRY_WORKERS", "4"))   # 미세조정 데이터 로더 수(병렬 실험 중 서버가 멈춘 적이 있어 기본 4)
# pytorch_model.bin만 있는 모델은 torch<2.6에서 transformers가 로드를 막는다(CVE-2025-32434)
# → safetensors로 변환해 둔 로컬 사본(CRY_MODELS/<org>__<name>)이 있으면 그걸 쓴다.
LOCAL_MODELS = os.path.expanduser(os.environ.get("CRY_MODELS", "~/cry/models"))


def src(key):
    p = os.path.join(LOCAL_MODELS, MODELS[key][0].replace("/", "__"))
    return p if os.path.isdir(p) else MODELS[key][0]


def cmd_convert(a):
    """pytorch_model.bin만 있는 모델을 safetensors 로컬 사본으로 바꾼다(torch<2.6이면 transformers가 .bin 로드를 막음)."""
    import shutil
    from huggingface_hub import HfApi, snapshot_download
    from safetensors.torch import save_file
    keys = a.models.split(",") if a.models else list(MODELS)
    for name, _ in (MODELS[k] for k in keys):
        dst = os.path.join(LOCAL_MODELS, name.replace("/", "__"))
        if name.startswith("local/") or os.path.isdir(dst) or "model.safetensors" in HfApi().list_repo_files(name):
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
    DATASETS[:] = [d for d in ALL_DATASETS if (m.dataset == d).any()]
    return m.reset_index(drop=True)


def feature_extractor(name):
    from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
    try:
        return AutoFeatureExtractor.from_pretrained(name)
    except Exception:          # preprocessor_config.json이 없는 사전학습 체크포인트
        return Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, padding_value=0.0, do_normalize=True,
                                        return_attention_mask=False)


class Augment:
    """실제 침실 조건 모사: OpenSLR 28 실측·시뮬레이션 잔향(반반) + ESC-50(아기 울음 제외)·실측 잡음 SNR 0~20dB.
    joint_build.py가 만든 ROOT/aug/{rir_real,rir_sim,noise}/*.wav 를 쓴다."""

    def __init__(self):
        d = os.path.join(ROOT, "aug")
        self.rir = [sorted(glob.glob(os.path.join(d, k, "*.wav"))) for k in ("rir_real", "rir_sim")]
        self.noise = sorted(glob.glob(os.path.join(d, "noise", "*.wav")))
        self.ok = all(self.rir) and bool(self.noise)
        self._c = {}

    def _load(self, p):
        if p not in self._c:
            if len(self._c) > 3000:
                self._c.clear()
            x, _ = sf.read(p, dtype="float32")
            self._c[p] = x
        return self._c[p]

    def __call__(self, x, rng):
        from scipy.signal import fftconvolve
        rms = np.sqrt(np.mean(x ** 2)) + 1e-8
        if rng.random() < 0.5:
            pool = self.rir[int(rng.random() < 0.5)]
            h = self._load(pool[rng.integers(len(pool))])[: SR]
            h = h / (np.abs(h).max() + 1e-8)
            x = fftconvolve(x, h)[: len(x)].astype(np.float32)
            x *= rms / (np.sqrt(np.mean(x ** 2)) + 1e-8)
        if rng.random() < 0.5:
            n = self._load(self.noise[rng.integers(len(self.noise))])
            n = tile(n, len(x)) if len(n) < len(x) else n[(o := rng.integers(0, len(n) - len(x) + 1)):o + len(x)]
            pn = np.mean(n ** 2) + 1e-10
            x = x + n * np.sqrt(np.mean(x ** 2) / pn / 10 ** (rng.uniform(0, 20) / 10))
        return x.astype(np.float32)


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
def target_class(ds):
    """AUC를 재는 이진 기준 클래스: 배고픔. 배고픔이 없는 Corvin은 통증."""
    return "hunger" if "hunger" in DS_CLASSES[ds] else "pain"


def _ds_metric(y, P, groups, ds, n_boot=1000):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    cls = [CLASSES.index(c) for c in DS_CLASSES[ds]]
    yy = np.array([cls.index(v) for v in y])
    Pc = P[:, cls] / P[:, cls].sum(1, keepdims=True)
    h = cls.index(CLASSES.index(target_class(ds)))

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
        fe, net = feature_extractor(name), AutoModel.from_pretrained(name)
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
    if a.windows:
        return extract_windows(a, m)
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


WIN, HOP, MAX_WIN = 4.0, 2.0, 30     # 시간 구조 모델용 창(4초, 2초 간격, 울음 하나당 최대 30창 = 약 60초)


@torch.no_grad()
def extract_windows(a, m):
    """울음 하나를 4초 창으로 나눠 창마다 층별 임베딩을 저장한다(시간 구조 모델 입력)."""
    out = os.path.join(RUNS, "feats", f"{a.model}_win.npz")
    if os.path.exists(out) and not a.force:
        print("이미 있음:", out)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    emb, _ = _embedder(a.model)
    feats, offs, t0 = [], [0], time.time()
    for i, r in enumerate(m.itertuples()):
        ws = windows(wav(r.path), WIN, HOP)[:MAX_WIN]
        for w in ws:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats.append(emb(tile(w, int(WIN * SR))).float().cpu().numpy().astype(np.float16))
        offs.append(len(feats))
        if (i + 1) % 300 == 0:
            print(f"  {a.model} 창 {i + 1}/{len(m)} {time.time() - t0:.0f}s", flush=True)
    np.savez(out, ids=m.id.values, X=np.stack(feats), offs=np.array(offs))
    print(f"{a.model}: 창 {len(feats)}개 {np.stack(feats).shape} → {out}")


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


def load_feats(model, m):
    """저장된 특징을 meta 순서(id)로 맞춘다. 메타에서 뺀 클립의 특징은 버린다."""
    z = np.load(os.path.join(RUNS, "feats", f"{model}.npz"), allow_pickle=True)
    pos = {k: i for i, k in enumerate(z["ids"])}
    return z["X"][[pos[k] for k in m.id]].astype(np.float32)


def cmd_probe(a):
    m = load_meta()
    X = load_feats(a.model, m)
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


MIN_GROUP = 5   # 아기(그룹) 평균으로 정규화할 최소 울음 수. 이보다 적으면 데이터 평균으로 대신한다


def enes_time(e):
    """Enes 행의 녹음 시각(파일명의 날짜 ddmmyyyy + 시각 HHMM)."""
    return pd.to_datetime(e.date.astype(int).astype(str).str.zfill(8) + e.time.astype(int).astype(str).str.zfill(4),
                          format="%d%m%Y%H%M", errors="coerce")


def norm_feats(X, m, mode):
    """'누구 울음인지' 성분을 빼는 정규화. 라벨은 쓰지 않는다(제품에서도 아기 울음은 라벨 없이 모인다).
    none: 그대로 / ds: 데이터셋 평균만 뺌(대조군) / baby: 같은 그룹(아기) 전체 평균을 뺌(그룹이 작으면 데이터 평균)
    online: Enes만 시간순으로 그 아기의 '이전' 울음 평균을 뺌(이전 울음 3개 미만이면 데이터 평균). 다른 데이터는 ds"""
    if mode == "none":
        return X
    X = X.copy()
    ds_mean = {d: X[(m.dataset == d).values].mean(0) for d in m.dataset.unique()}
    base = X.copy()
    for d, mu in ds_mean.items():
        X[(m.dataset == d).values] = base[(m.dataset == d).values] - mu
    if mode == "ds":
        return X
    if mode == "baby":
        for g, idx in m.groupby("group").indices.items():
            if len(idx) >= MIN_GROUP:
                X[idx] = base[idx] - base[idx].mean(0)
        return X
    if mode == "online":
        e = m[m.dataset == "enes"].copy()
        e["t"] = enes_time(e)
        for g, sub in e.sort_values("t").groupby("group"):
            idx = sub.index.values
            for k in range(3, len(idx)):
                X[idx[k]] = base[idx[k]] - base[idx[:k]].mean(0)
        return X
    raise ValueError(mode)


def cmd_robust(a):
    """겹 분할을 바꿔 반복(repeat)하고, 라벨을 데이터 안에서 섞은 귀무 분포(perm)와 비교한다.
    모델 6종 × 조건 중 최고값을 고르면 우연히 높게 나올 수 있어서, 같은 절차를 무작위 라벨에 돌려 기준을 잡는다."""
    m = load_meta()
    X = norm_feats(load_feats(a.model, m), m, a.norm)
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
                rows.append({"model": a.model, "norm": a.norm, "kind": kind, "r": r, "cond": c, "test": d, **r_})
                if a.norm == "online" and d == "enes":     # 이전 울음 3개 이상 쌓인 울음만 따로
                    e = m.iloc[ix].copy()
                    e["t"] = enes_time(e)
                    n_prev = e.sort_values("t").groupby("group").cumcount()
                    keep = (n_prev.reindex(e.index).values >= 3)
                    r2 = _ds_metric(y[ix][keep], PP[ix][keep], m.group.values[ix][keep], d, n_boot=0)
                    rows.append({"model": a.model, "norm": a.norm, "kind": kind, "r": r, "cond": c,
                                 "test": "enes_prev3", **r2})
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
    # 아기 식별 정확도(Enes, 무작위 5겹): 정규화로 '누구 울음인지' 성분이 얼마나 빠졌는지
    from sklearn.model_selection import StratifiedKFold
    ei = np.where(m.dataset.values == "enes")[0]
    yb = pd.factorize(m.group.values[ei])[0]
    nb = yb.max() + 1
    acc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(ei, yb):
        f = fit_probe(X[ei[tr]], yb[tr], np.ones((len(tr), nb), bool), np.ones(len(tr)), n_out=nb)
        acc.append((f(X[ei[te]]).argmax(1).cpu().numpy() == yb[te]).mean())
    rows.append({"model": a.model, "norm": a.norm, "kind": "baby_id", "r": 0, "cond": "-", "test": "enes",
                 "bal": float(np.mean(acc))})
    print(f"{a.model} norm={a.norm}: Enes 아기 식별 정확도 {np.mean(acc):.3f} (무작위 {1 / nb:.3f})")
    df = pd.DataFrame(rows)
    name = "robust.csv" if a.norm == "none" else f"robust_{a.norm}.csv"
    df.to_csv(os.path.join(RUNS, "probe", a.model, name), index=False)
    print(df.groupby(["kind", "cond", "test"])[["bal", "auc"]].agg(["mean", "std", "max"]).round(3))


# --- Corvin 통증 대조: 울음 자체인가, 녹음 장소(병원 vs 집)인가 ------------------------------
def cmd_background(a):
    """Corvin 통증(병원 예방접종) vs 불편(집 목욕)을 같은 모델로 세 번 비교한다.
    전체 / 큰 소리 구간(울음) / 조용한 구간(울음 사이 배경). 조용한 구간만으로도 맞히면 장소 지름길이다.
    구간은 25ms 프레임 RMS 상·하위 30%로 나누고, 아기 단위 5겹 + 층 가중합 프로브로 통증 AUC를 잰다."""
    from sklearn.metrics import roc_auc_score
    m = load_meta()
    c = m[m.dataset == "corvin"].reset_index(drop=True)
    emb, _ = _embedder(a.model)
    fr, hop = int(0.025 * SR), int(0.010 * SR)
    parts = {"전체": [], "큰 소리(울음)": [], "조용한 구간(배경)": []}
    keep = []
    for r in c.itertuples():
        x = wav(r.path)
        n = max(1, 1 + (len(x) - fr) // hop)
        e = np.array([np.sqrt(np.mean(x[i * hop:i * hop + fr] ** 2) + 1e-12) for i in range(n)])
        lo, hi = np.percentile(e, 30), np.percentile(e, 70)
        segs = {}
        for name, sel in (("큰 소리(울음)", e >= hi), ("조용한 구간(배경)", e <= lo)):
            idx = np.where(sel)[0]
            segs[name] = np.concatenate([x[i * hop:i * hop + hop] for i in idx]) if len(idx) else np.zeros(0, np.float32)
        ok = len(segs["조용한 구간(배경)"]) >= SR // 4
        keep.append(ok)
        for name, w in (("전체", x), *segs.items()):
            w = tile(w if len(w) else np.zeros(SR, np.float32), max(SR, len(w)))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                E = torch.stack([emb(q).float() for q in windows(w, 10.0, 5.0)]).mean(0)
            parts[name].append(E.cpu().numpy())
    keep = np.array(keep)
    c = c[keep].reset_index(drop=True)
    y = (c.label == "pain").astype(int).values
    fold = c.fold.values
    L = [f"# Corvin 통증 대조 ({a.model})", "",
         f"- 클립 {len(c)}개(조용한 구간 0.25초 이상), 아기 {c.group.nunique()}명, 통증 {y.sum()} / 불편 {len(y) - y.sum()}",
         "- 아기 단위 5겹, 층 가중합 선형 프로브, 통증 AUC (무작위 0.5)", "",
         "| 입력 | 통증 AUC |", "|---|---:|"]
    for name, F_ in parts.items():
        X = np.stack(F_)[keep]
        P = np.zeros(len(c))
        for k in range(N_FOLDS):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            f = fit_probe(X[tr], y[tr], np.ones((len(tr), 2), bool), np.ones(len(tr)), n_out=2)
            P[te] = torch.softmax(f(X[te]), -1)[:, 1].cpu().numpy()
        L.append(f"| {name} | {roc_auc_score(y, P):.3f} |")
    L += ["", "조용한 구간만으로도 AUC가 높으면 '통증 감지'가 아니라 녹음 장소를 맞힌 것이다."]
    out = os.path.join(RUNS, f"CORVIN_BG_{a.model}.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


# --- 시간 구조 모델: 창 순서를 보는 작은 트랜스포머 vs 창 평균 --------------------------------
class SeqNet(nn.Module):
    def __init__(self, L, D, temporal, d=256):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(L))
        self.proj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d), nn.GELU(), nn.Dropout(0.2))
        self.temporal = temporal
        if temporal:
            self.pos = nn.Parameter(torch.zeros(1, MAX_WIN, d))
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.2, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)   # 중첩 텐서 경로는 세그폴트
            self.att = nn.Linear(d, 1)
        self.out = nn.Linear(d, len(CLASSES))

    def forward(self, X, pad):           # X: [B, T, L, D], pad: [B, T] True=빈 칸
        h = self.proj((X * torch.softmax(self.a, 0)[None, None, :, None]).sum(2))
        if self.temporal:
            h = self.enc(h + self.pos[:, : h.shape[1]], src_key_padding_mask=pad)
            w = self.att(h).squeeze(-1).masked_fill(pad, -1e4).softmax(1)
            h = (h * w[..., None]).sum(1)
        else:
            keep = (~pad).float()[..., None]
            h = (h * keep).sum(1) / keep.sum(1)
        return self.out(h)


def cmd_temporal(a):
    m = load_meta()
    z = np.load(os.path.join(RUNS, "feats", f"{a.model}_win.npz"), allow_pickle=True)
    pos = {k: i for i, k in enumerate(z["ids"])}
    sel = [pos[k] for k in m.id]
    Xw = np.concatenate([z["X"][z["offs"][i]:z["offs"][i + 1]] for i in sel])
    offs = np.concatenate([[0], np.cumsum([z["offs"][i + 1] - z["offs"][i] for i in sel])])
    L, D = Xw.shape[1:]
    T = int(np.diff(offs).max())
    Xall = np.zeros((len(m), T, L, D), np.float16)
    pad = np.ones((len(m), T), bool)
    for i in range(len(m)):
        n = offs[i + 1] - offs[i]
        Xall[i, :n], pad[i, :n] = Xw[offs[i]:offs[i + 1]], False
    mu = Xw.astype(np.float32).mean(0)
    sd = Xw.astype(np.float32).std(0) + 1e-5
    out_dir = os.path.join(RUNS, "temporal", a.model)
    os.makedirs(out_dir, exist_ok=True)
    preds = []

    def fit_predict(tr, te, temporal, seed):
        torch.manual_seed(seed)
        net = SeqNet(L, D, temporal).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.05)
        mt = m.iloc[tr]
        w = torch.as_tensor(balance_weights(mt), dtype=torch.float32)
        M = torch.as_tensor(ds_mask(mt.dataset.values))
        y = torch.as_tensor(mt.y.values)
        Mu, Sd = torch.as_tensor(mu, device=DEV), torch.as_tensor(sd, device=DEV)
        rng = np.random.default_rng(seed)
        for ep in range(a.epochs):
            net.train()
            for b in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // 32)):
                xb = (torch.as_tensor(Xall[tr[b]], device=DEV).float() - Mu) / Sd
                lo = net(xb, torch.as_tensor(pad[tr[b]], device=DEV)).masked_fill(~M[b].to(DEV), -1e4)
                loss = (F.cross_entropy(lo, y[b].to(DEV), reduction="none") * w[b].to(DEV)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        P = []
        with torch.no_grad():
            for b in np.array_split(np.arange(len(te)), max(1, len(te) // 64)):
                xb = (torch.as_tensor(Xall[te[b]], device=DEV).float() - Mu) / Sd
                P.append(masked_probs(net(xb, torch.as_tensor(pad[te[b]], device=DEV)), m.dataset.values[te[b]]).cpu())
        return torch.cat(P).numpy()

    for seed in range(a.seeds):
        for k in range(N_FOLDS):
            te_all = np.where(m.fold == k)[0]
            for temporal in (True, False):
                arch = "seq" if temporal else "mean"
                tr = np.where(m.fold != k)[0]
                preds.append(pred_frame(m.iloc[te_all], fit_predict(tr, te_all, temporal, SEED + seed),
                                        stage="temporal", model=f"{a.model}_{arch}", protocol="cv",
                                        cond=f"mixed_s{seed}", fold=k))
                for d in ("enes",):
                    tr = np.where((m.fold != k) & (m.dataset == d))[0]
                    te = te_all[m.dataset.values[te_all] == d]
                    preds.append(pred_frame(m.iloc[te], fit_predict(tr, te, temporal, SEED + seed),
                                            stage="temporal", model=f"{a.model}_{arch}", protocol="cv",
                                            cond=f"single_s{seed}", fold=k))
        print(f"seed {seed} 완료", flush=True)
    df = pd.concat(preds)
    df.to_csv(os.path.join(out_dir, "preds.csv"), index=False)
    s = summarize(df)
    print(s[s.test.isin(["enes", "dac", "baidu"])][["model", "cond", "test", "bal", "f1", "auc"]].round(3).to_string())


# --- 울음 도메인 추가 사전학습(DAPT): wav2vec2 대조 학습을 라벨 없는 울음으로 이어 간다 -------------
def cmd_dapt(a):
    """평가의 중심인 Enes 라벨 울음은 넣지 않는다(Enes 평가는 '본 적 없는 소리'로 유지).
    DAC·Baidu·Corvin·Cry Sense 소리는 라벨 없이 넣으므로 그 데이터 평가는 전이(transductive) 조건이다."""
    from transformers import Wav2Vec2ForPreTraining
    from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
    m = load_meta()
    u = pd.read_csv(os.path.join(ROOT, "meta_unlab.csv"))
    u = u[u.dataset != "enes_other"]
    paths = list(u.path) + list(m[m.dataset != "enes"].path)
    print(f"DAPT 소리 {len(paths)}개 (라벨 없는 울음 {len(u)} + Enes 외 라벨 데이터의 소리 {len(paths) - len(u)})")
    name = src(a.model)
    fe = feature_extractor(name)
    model = Wav2Vec2ForPreTraining.from_pretrained(name).to(DEV)
    model.freeze_feature_encoder()
    crop = int(5 * SR)
    for p in paths:
        wav(p)
    weights = np.array([max(1, len(_WAV[p]) // crop) for p in paths], float)   # 긴 클립은 더 자주
    steps = a.steps
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s_: min(1.0, s_ / (0.08 * steps)) * max(0.0, (steps - s_) / (steps * 0.92)))
    rng = np.random.default_rng(SEED)
    cfg = model.config
    t0 = time.time()
    for step in range(steps):
        model.train()
        idx = rng.choice(len(paths), 16, p=weights / weights.sum())
        batch = []
        for i in idx:
            x = wav(paths[i])
            if len(x) > crop:
                o = rng.integers(0, len(x) - crop + 1)
                x = x[o:o + crop]
            batch.append(tile(x, crop))
        x = torch.as_tensor(np.stack([fe(b, sampling_rate=SR, return_tensors="np")["input_values"][0] for b in batch])).to(DEV)
        seq = int(model._get_feat_extract_output_lengths(crop))
        mask = _compute_mask_indices((len(idx), seq), mask_prob=cfg.mask_time_prob if cfg.mask_time_prob > 0.2 else 0.65,
                                     mask_length=cfg.mask_time_length or 10, min_masks=2)
        neg = _sample_negative_indices((len(idx), seq), cfg.num_negatives, mask_time_indices=mask)
        model.set_gumbel_temperature(max(2.0 * 0.999995 ** (step * 40), 0.5))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, mask_time_indices=torch.as_tensor(mask, device=DEV),
                        sampled_negative_indices=torch.as_tensor(neg, device=DEV))
        loss = out.loss / mask.sum() if out.loss is not None else None
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{steps} loss {loss.item():.4f} contrastive {out.contrastive_loss.item() / mask.sum():.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    dst = os.path.join(LOCAL_MODELS, "local__voc2vec_dapt" if a.model == "voc2vec" else f"local__{a.model}_dapt")
    model.save_pretrained(dst, safe_serialization=True)
    fe.save_pretrained(dst)
    print("저장:", dst)


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
    def __init__(self, key, n_dom=None, pool="last"):
        super().__init__()
        from transformers import AutoModel, ASTModel
        name, kind = src(key), MODELS[key][1]
        self.kind, self.pool = kind, pool
        if kind == "ast":
            self.backbone = ASTModel.from_pretrained(name)
        else:
            self.backbone = AutoModel.from_pretrained(name)
            self.backbone.freeze_feature_encoder()
        D = self.backbone.config.hidden_size
        self.lw = nn.Parameter(torch.zeros(self.backbone.config.num_hidden_layers + 1))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, len(CLASSES)))
        self.dom = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_dom or len(DATASETS)))

    def forward(self, x, lam=0.0):
        if self.kind == "ast":
            h = self.backbone(x).pooler_output
        elif self.pool == "wsum":       # 층 가중합(SUPERB 방식): 준언어 정보는 중간 층에 많다
            hs = torch.stack(self.backbone(x, output_hidden_states=True).hidden_states)    # [L, B, T, D]
            h = (hs.mean(2) * torch.softmax(self.lw, 0)[:, None, None]).sum(0)
        else:
            h = self.backbone(x).last_hidden_state.mean(1)
        return self.head(h), self.dom(GRL.apply(h, lam))


class FTData(torch.utils.data.Dataset):
    """학습: 무작위 CROP초 구간(짧으면 반복해 채움) + 음량·잡음 증강. 평가: CROP/2 간격 창 목록."""
    CROP = 8.0

    def __init__(self, m, key, train, aug=False, dom=None):
        self.m, self.train, self.kind = m.reset_index(drop=True), train, MODELS[key][1]
        self.fe = feature_extractor(src(key))
        self.aug = Augment() if (aug and train) else None
        self.dom = dom if dom is not None else [DATASETS.index(d) for d in self.m.dataset]
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
            if self.aug is not None and self.aug.ok:   # 실측 잔향·잡음
                x = self.aug(x, rng)
            elif rng.random() < 0.5:                   # 백색 잡음 SNR 15~40dB
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
        return f, int(r.y), torch.as_tensor(ds_mask([r.dataset])[0]), self.dom[i], i


def finetune(key, m_tr, m_te, dann, epochs, log, adv="dataset", aug=False, pool="last", lr=None):
    """adv: dataset(출처 판별 방해) / group(아기·업로더 판별 방해, 학습에 5개 이상 있는 그룹만)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for p in pd.concat([m_tr, m_te]).path:     # 워커가 fork 전에 공유하도록 미리 적재
        wav(p)
    m_tr = m_tr.reset_index(drop=True)
    if adv == "group":
        vc = m_tr.group.value_counts()
        gid = {g: i for i, g in enumerate(vc[vc >= MIN_GROUP].index)}
        dom, n_dom = [gid.get(g, -100) for g in m_tr.group], max(1, len(gid))
    else:
        dom, n_dom = None, len(DATASETS)
    net = FTNet(key, n_dom, pool).to(DEV)
    bs = 8 if key.endswith("_l") else 16
    tr = FTData(m_tr, key, True, aug=aug, dom=dom)
    sampler = torch.utils.data.WeightedRandomSampler(balance_weights(tr.m), len(tr), replacement=True)
    dl = torch.utils.data.DataLoader(tr, bs, sampler=sampler, num_workers=WORKERS, drop_last=True,
                                     persistent_workers=True)
    lr_bb = lr or (2e-5 if MODELS[key][1] == "ast" else 3e-5)
    opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": lr_bb},
                             {"params": list(net.head.parameters()) + list(net.dom.parameters()) + [net.lw], "lr": 1e-3}],
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
            dl_loss = F.cross_entropy(ld.float(), dom, ignore_index=-100) if (dom >= 0).any() else ld.sum() * 0
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
    dl = torch.utils.data.DataLoader(te, bs, num_workers=WORKERS)
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
    out_dir = os.path.join(RUNS, "ft", a.model + (f"_{a.tag}" if a.tag else ""))
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
                    for d in (a.single_on.split(",") if a.single_on else DATASETS):
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
        P = finetune(a.model, m_tr, m_te, cond in ("dann", "badv"), a.epochs, log,
                     adv="group" if cond == "badv" else "dataset", aug=a.aug, pool=a.pool, lr=a.lr)
        pred_frame(m_te.reset_index(drop=True), P, stage="ft",
                   model=a.model + (f"_{a.tag}" if a.tag else ""), protocol=a.protocol,
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
        h = cls.index(CLASSES.index(target_class(d)))
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
    parts += [pd.read_csv(p) for p in glob.glob(os.path.join(RUNS, "temporal", "*", "preds.csv"))]
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
    p = sp.add_parser("convert")
    p.add_argument("--models", default="", help="변환할 모델 키(쉼표). 비우면 전부")
    sp.add_parser("folds")
    for name in ["extract", "probe"]:
        p = sp.add_parser(name)
        p.add_argument("--model", required=True, choices=list(MODELS))
        p.add_argument("--force", action="store_true")
        p.add_argument("--windows", action="store_true", help="extract: 4초 창별 임베딩(시간 구조 모델용)")
    p = sp.add_parser("background")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p = sp.add_parser("temporal")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seeds", type=int, default=3)
    p = sp.add_parser("dapt")
    p.add_argument("--model", default="voc2vec", choices=list(MODELS))
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=5e-5)
    p = sp.add_parser("robust")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--perms", type=int, default=5)
    p.add_argument("--norm", choices=["none", "ds", "baby", "online"], default="none")
    p = sp.add_parser("finetune")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--protocol", choices=["cv", "lodo"], default="cv")
    p.add_argument("--cond", default="single,mixed,dann")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--aug", action="store_true", help="실측 잔향·잡음 증강(ROOT/aug)")
    p.add_argument("--pool", choices=["last", "wsum"], default="last")
    p.add_argument("--lr", type=float, default=None, help="백본 학습률(기본 AST 2e-5, 그 외 3e-5)")
    p.add_argument("--tag", default="", help="결과 폴더 접미사(ft/<model>_<tag>)")
    p.add_argument("--single-on", dest="single_on", default="", help="single 조건을 돌릴 데이터(쉼표)")
    sp.add_parser("report")
    a = ap.parse_args()
    {"convert": cmd_convert, "folds": cmd_folds, "extract": cmd_extract, "probe": cmd_probe,
     "robust": cmd_robust, "temporal": cmd_temporal, "background": cmd_background, "dapt": cmd_dapt,
     "finetune": cmd_finetune, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
