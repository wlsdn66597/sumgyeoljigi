"""소리 종류 분류(sound_events) 무학습 평가 — YAMNet 그대로, 보유 데이터로 측정.

    python eval_sound_events.py [--thr 0.3] [--out D:/datasets/cry/runs/sound_events]
    (--thr를 주면 모든 버킷에 같은 임계값, 안 주면 sound_events.THRESHOLDS)

평가셋 (정답 버킷)
  cry    Donate-a-Cry 정제본 · Infant cry 'cry'(Donate-a-Cry와 겹치지 않는 것) · ESC-50 crying_baby
  laugh  Cry Sense 'laugh'
  cough  ESC-50 coughing · sneezing   (★ 성인 소리)
  other  Cry Sense 'noise'·'silence' · ESC-50 나머지 44개 범주(숨소리·코골이·생활소음 등)
  adult_laugh  ESC-50 laughing — 아기 웃음이 아니라 따로 보고(지표에서 제외)
  whimper(칭얼거림)는 정답 데이터가 없어 '어디서 얼마나 켜지는지'만 본다.

출처 간 중복: 일부 데이터가 ESC-50에서 가져왔다고 밝히고 있어서, 평균 임베딩
코사인 ≥ --dup-sim 인 다른 출처 클립은 한쪽만 남긴다(우선순위 esc50 > dac > infantcry > crysense).
"""
import argparse
import glob
import hashlib
import os
from collections import Counter

import numpy as np
import pandas as pd

import config
import features
import sound_events as se
from prepare_data import to_wav16k

TRUTHS = ["cry", "laugh", "cough", "other"]
PREDS = se.EVENT_BUCKETS + ["other"]
SOURCE_PRIORITY = ["esc50", "dac", "infantcry", "crysense"]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def build_eval_set(root):
    rows = []
    dac_dir = os.path.join(root, "donateacry-corpus", "donateacry_corpus_cleaned_and_updated_data")
    dac_md5 = set()
    for p in sorted(glob.glob(os.path.join(dac_dir, "*", "*.wav"))):
        dac_md5.add(md5(p))
        rows.append((p, "cry", "dac", os.path.basename(os.path.dirname(p))))

    ic_dir = os.path.join(root, "extra", "infantcry", "Dataset", "cry")
    for p in sorted(glob.glob(os.path.join(ic_dir, "*"))):
        if md5(p) not in dac_md5:
            rows.append((p, "cry", "infantcry", "cry"))

    esc = os.path.join(root, "ESC-50-master")
    meta = pd.read_csv(os.path.join(esc, "meta", "esc50.csv"))
    esc_map = {"crying_baby": "cry", "coughing": "cough", "sneezing": "cough", "laughing": "adult_laugh"}
    for _, r in meta.iterrows():
        rows.append((os.path.join(esc, "audio", r.filename), esc_map.get(r.category, "other"),
                     "esc50", r.category))

    cs = os.path.join(root, "extra", "crysense", "cry")
    for folder, truth in [("laugh", "laugh"), ("noise", "other"), ("silence", "other")]:
        for p in sorted(glob.glob(os.path.join(cs, folder, "*"))):
            rows.append((p, truth, "crysense", folder))
    return pd.DataFrame(rows, columns=["path", "truth", "source", "detail"])


def drop_cross_source_duplicates(df, E, thr):
    """다른 출처끼리 거의 같은 클립이면 우선순위 낮은 쪽을 버린다."""
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    S = En @ En.T
    rank = df.source.map({s: i for i, s in enumerate(SOURCE_PRIORITY)}).to_numpy()
    drop, pairs = set(), Counter()
    for a, b in zip(*np.where(np.triu(S, 1) >= thr)):
        if df.source[a] == df.source[b] or a in drop or b in drop:
            continue
        lo, hi = (a, b) if rank[a] > rank[b] else (b, a)     # lo = 우선순위 낮은 쪽
        drop.add(lo)
        pairs[f"{df.source[lo]}:{df.detail[lo]} ≈ {df.source[hi]}:{df.detail[hi]}"] += 1
    keep = np.array([i not in drop for i in range(len(df))])
    return keep, pairs


def prf(y_true, y_pred, label):
    tp = int(np.sum((y_true == label) & (y_pred == label)))
    fp = int(np.sum((y_true != label) & (y_pred == label)))
    fn = int(np.sum((y_true == label) & (y_pred != label)))
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r and not np.isnan(p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=None, help="모든 버킷 공통 임계값 (기본: 버킷별 THRESHOLDS)")
    ap.add_argument("--dup-sim", type=float, default=0.99)
    ap.add_argument("--out", default=os.path.join(config.DATA_ROOT, "runs", "sound_events"))
    args = ap.parse_args()
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    df = build_eval_set(config.DATA_ROOT)
    print(f"평가 클립 {len(df)}개 — YAMNet 점수 계산(캐시 사용)...")
    _, class_names = features.load_yamnet()
    bidx = se.bucket_index(class_names)
    B, E, top = [], [], []
    for i, r in df.iterrows():
        scores, emb_mean = features.file_scores(to_wav16k(r.path))
        B.append(se.clip_bucket_scores(scores, bidx))
        E.append(emb_mean)
        top.append(class_names[int(scores.mean(axis=0).argmax())] if len(scores) else "-")
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(df)}")
    B, E = np.stack(B), np.stack(E)
    df["top_class"] = top

    keep, pairs = drop_cross_source_duplicates(df, E, args.dup_sim)
    if pairs:
        print(f"출처 간 중복(cos≥{args.dup_sim}) 제외 {int((~keep).sum())}개:", dict(pairs.most_common(8)))
    df, B = df[keep].reset_index(drop=True), B[keep]
    for j, b in enumerate(se.EVENT_BUCKETS):
        df[f"s_{b}"] = B[:, j]
    thr = se.THRESHOLDS if args.thr is None else {b: args.thr for b in se.EVENT_BUCKETS}
    df["pred"] = [se.decide(x, thr) for x in B]

    main_mask = df.truth.isin(TRUTHS).to_numpy()
    d = df[main_mask]
    y, yp = d.truth.to_numpy(), d.pred.to_numpy()
    L = ["# 소리 종류 분류 — YAMNet 무학습 평가", "",
         f"- 임계값 {thr}, 울음 우선 판정(울음 점수가 임계값을 넘으면 울음), 클립 점수 = 약 1초 프레임 중 최댓값",
         f"- 평가 클립: {dict(d.truth.value_counts())} (+ 성인 웃음 {int((df.truth == 'adult_laugh').sum())}개 별도)",
         f"- 출처 간 중복 제외: {int((~keep).sum())}개", ""]

    L += ["## 버킷별 성능", "",
          "| 버킷 | ROC-AUC | AP | 정밀도 | 재현율 | F1 |", "|---|---:|---:|---:|---:|---:|"]
    for j, b in enumerate(["cry", "laugh", "cough"]):
        s = d[f"s_{b}"].to_numpy()
        auc = roc_auc_score(y == b, s)
        ap_ = average_precision_score(y == b, s)
        p, r, f = prf(y, yp, b)
        L.append(f"| {b} | {auc:.3f} | {ap_:.3f} | {p:.3f} | {r:.3f} | {f:.3f} |")
    p, r, f = prf(y, yp, "other")
    L.append(f"| other | - | - | {p:.3f} | {r:.3f} | {f:.3f} |")
    macro = f1_score(y, yp, labels=TRUTHS, average="macro", zero_division=0)
    L += ["", f"4버킷 Macro-F1: **{macro:.3f}**", ""]

    L += ["## 임계값별 재현율 / 정밀도 (모든 버킷에 같은 임계값, 울음 우선)", "", "| 버킷 | " + " | ".join(f"thr {t}" for t in (0.1, 0.2, 0.3, 0.5)) + " |",
          "|---|" + "---:|" * 4]
    for b in ["cry", "laugh", "cough"]:
        cells = []
        for t in (0.1, 0.2, 0.3, 0.5):
            tt = {k: t for k in se.EVENT_BUCKETS}
            pt = np.array([se.decide(x, tt) for x in d[[f"s_{k}" for k in se.EVENT_BUCKETS]].to_numpy()])
            p_, r_, _ = prf(y, pt, b)
            cells.append(f"{r_:.2f} / {p_:.2f}")
        L.append(f"| {b} | " + " | ".join(cells) + " |")

    L += ["", "## 혼동행렬 (행=정답, 열=예측)", "", "| | " + " | ".join(PREDS) + " |", "|---|" + "---:|" * len(PREDS)]
    for t in TRUTHS + ["adult_laugh"]:
        row = df[df.truth == t].pred.value_counts()
        L.append(f"| {t} | " + " | ".join(str(int(row.get(c, 0))) for c in PREDS) + " |")

    L += ["", "## 출처별 울음 재현율", ""]
    for src, g in d[d.truth == "cry"].groupby("source"):
        L.append(f"- {src}: {(g.pred == 'cry').mean():.3f} ({len(g)}개)")

    fa = d[(d.truth == "other") & (d.pred != "other")]
    L += ["", f"## 오경보: other인데 이벤트로 잡힌 것 ({len(fa)}/{int((y == 'other').sum())})", ""]
    for (det, pr), n in fa.groupby(["detail", "pred"]).size().sort_values(ascending=False).head(10).items():
        L.append(f"- {det} → {pr}: {n}개")

    wh = df[df.pred == "whimper"]
    L += ["", f"## 칭얼거림(whimper) 예측 {len(wh)}개의 실제 정답", "", f"- {dict(wh.truth.value_counts())}"]
    L += ["", "## 웃음 폴더 점검: 평균 점수 1위 YAMNet 클래스", "",
          f"- {dict(df[(df.truth == 'laugh')].top_class.value_counts().head(6))}"]
    L += ["", "## 한계", "",
          "- 기침·재채기는 ESC-50 성인 소리로만 평가했다. 아기 기침 성능은 따로 확인이 필요하다.",
          "- 칭얼거림·옹알이는 정답 데이터가 없다.",
          "- 공개 데이터는 근접 녹음이다. 침실 마이크 거리·잔향에서는 점수가 낮아질 수 있다."]

    os.makedirs(args.out, exist_ok=True)
    df.drop(columns=["path"]).assign(file=df.path.map(os.path.basename)).to_csv(
        os.path.join(args.out, "clip_scores.csv"), index=False, encoding="utf-8-sig")
    report = "\n".join(L) + "\n"
    open(os.path.join(args.out, "SOUND_EVENTS_REPORT.md"), "w", encoding="utf-8").write(report)
    print("\n" + report)
    print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()
