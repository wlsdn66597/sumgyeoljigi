"""울음 '이유' 3클래스 분류 헤드 학습 (YAMNet 임베딩 전이학습) + 교차검증.

    python prepare_data.py                 # 먼저 manifest.csv 생성
    python train.py [--aug 2] [--no-pitch] [--folds 5] [--holdout-source crysense]

평가 방식 (이전 버전의 문제를 고친 부분)
  - StratifiedGroupKFold 5겹: 같은 아기는 한 fold에만, 소수 클래스도 모든 fold에 등장.
    (단일 분할에서는 belly_pain이 test에 0개였다)
  - 조기 종료용 검증셋도 group 단위로 따로 뗀다. (이전: Keras validation_split이
    라벨순 정렬 배열의 마지막 10%를 떼어 'tired'가 통째로 학습에서 빠졌다)
  - 기준선 2개와 비교: 최빈 클래스, 로지스틱 회귀.
  - 세그먼트 단위(1.5초, 실시간 추론과 동일)와 클립 단위(세그먼트 확률 평균) 모두 보고.
  - 판정 보류: 최고 확률 < CONF_MIN 이면 보류 → 커버리지와 보류 제외 정확도를 함께 보고.

산출물 (--out, 기본 artifacts/)
  cry_head.keras · cry_head.tflite · labels.json · meta.json
  cv_results.json · CV_REPORT.md · confusion_matrix.png (클립 단위, 전 fold 합산)
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd

import augment
import config
import features


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)


# --- 데이터 적재 ------------------------------------------------------------
def load_clips(args):
    df = pd.read_csv(config.MANIFEST)
    df = df[df.label.isin(config.CLASSES)].reset_index(drop=True)
    if args.holdout_source:
        hold = df[df.source == args.holdout_source].reset_index(drop=True)
        df = df[df.source != args.holdout_source].reset_index(drop=True)
    else:
        hold = df.iloc[0:0]

    print(f"클립 {len(df)}개 임베딩 추출(캐시 사용)...")
    clean, keep = [], []
    for i, r in df.iterrows():
        cry, emb = features.file_frames(r.path)
        X, _ = features.segments(cry, emb)
        if len(X):
            clean.append(X)
            keep.append(i)
    dropped = df.drop(index=keep)
    df = df.loc[keep].reset_index(drop=True)
    if len(dropped):
        print(f"울음 세그먼트 없음(CRY_MIN={config.CRY_MIN})으로 제외: {len(dropped)}개 →",
              dict(dropped.groupby(["source", "label"]).size()))

    aug = [[] for _ in range(len(df))]
    if args.aug:
        n_noise, n_rir = len(augment._noise_files()), len(augment._rir_files())
        print(f"증강 임베딩 추출 (클립당 {args.aug}개, pitch={'on' if args.pitch else 'off'}, "
              f"소음 {n_noise}개 · RIR {n_rir}개{' — 없으면 합성으로 대체' if not (n_noise and n_rir) else ''})...")
        for i, r in df.iterrows():
            for k in range(args.aug):
                rng = np.random.default_rng([config.SEED, i, k])
                # 소음·RIR 파일 구성이 바뀌면 다른 증강이므로 캐시 키에 포함
                tag = f"aug{k}|seed{config.SEED}|pitch{int(args.pitch)}|n{n_noise}r{n_rir}|{r.md5}"
                cry, emb = features.file_frames(
                    r.path, aug=lambda w: augment.random_augment(w, rng, args.pitch), tag=tag)
                X, _ = features.segments(cry, emb)
                if len(X):
                    aug[i].append(X)
    df["group"] = merge_near_duplicates(df, clean, args.dup_sim)
    return df, clean, aug, hold


def merge_near_duplicates(df, clean, thr):
    """다른 group인데 평균 임베딩이 거의 같은 클립(재인코딩·재포장 사본)을 같은 group으로
    묶어 train/test 누수를 막는다 (union-find)."""
    M = np.stack([x.mean(axis=0) for x in clean])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = M @ M.T
    parent = list(range(len(df)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    groups = df.group.tolist()
    merged = 0
    for a, b in zip(*np.where(np.triu(S, 1) >= thr)):
        if groups[a] != groups[b] and find(a) != find(b):
            parent[find(a)] = find(b)
            merged += 1
    first = {}                      # 원래 같은 group(같은 아기)끼리도 하나로
    for i, g in enumerate(groups):
        if g in first:
            ra, rb = find(i), find(first[g])
            if ra != rb:
                parent[ra] = rb
        else:
            first[g] = i
    if merged:
        print(f"유사 중복(cos≥{thr}) 병합: {merged}쌍")
    return [f"g{find(i)}" for i in range(len(df))]


def stack(idx, clean, aug, y_clip, with_aug):
    X, y, cid = [], [], []
    for i in idx:
        for P in [clean[i]] + (aug[i] if with_aug else []):
            X.append(P)
            y += [y_clip[i]] * len(P)
            cid += [i] * len(P)
    return np.concatenate(X), np.array(y), np.array(cid)


# --- 모델 ------------------------------------------------------------------
def build_mlp(n_cls):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(128, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_cls, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def class_weights(y, n_cls):
    from sklearn.utils.class_weight import compute_class_weight
    present = np.unique(y)
    w = compute_class_weight("balanced", classes=present, y=y)
    cw = {int(c): 1.0 for c in range(n_cls)}
    cw.update({int(c): float(v) for c, v in zip(present, w)})
    return cw


def inner_val_split(idx, y_clip, groups, seed):
    """train 클립 안에서 group 단위로 조기 종료용 검증셋(약 20%)을 뗀다."""
    from sklearn.model_selection import StratifiedGroupKFold
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    a, b = next(sg.split(idx, y_clip[idx], np.asarray(groups)[idx]))
    return idx[a], idx[b]


def fit_mlp(Xtr, ytr, Xva, yva, n_cls):
    import tensorflow as tf
    m = build_mlp(n_cls)
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=100, batch_size=64, shuffle=True,
          class_weight=class_weights(ytr, n_cls), verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                      restore_best_weights=True)])
    return m


def fit_logreg(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, C=0.1, class_weight="balanced")).fit(Xtr, ytr)


# --- 지표 ------------------------------------------------------------------
def metrics(y, pred, n_cls):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
    return {"macro_f1": float(f1_score(y, pred, average="macro", labels=range(n_cls), zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(y, pred)),
            "acc": float(accuracy_score(y, pred)),
            "recall": [float(r) for r in recall_score(y, pred, average=None, labels=range(n_cls),
                                                      zero_division=0)]}


def clip_probs(P, cid):
    ids = np.unique(cid)
    return ids, np.stack([P[cid == i].mean(axis=0) for i in ids])


def selective(y, P, thr, major=None):
    """판정 보류 적용 시 커버리지·정확도. 불균형 데이터에선 정확도가 최빈 클래스 비율에
    끌려가므로, 같은 판정 구간에서 '전부 최빈 클래스'로 찍었을 때의 정확도도 함께 낸다."""
    keep = P.max(axis=1) >= thr
    nan = float("nan")
    acc = float((P[keep].argmax(1) == y[keep]).mean()) if keep.any() else nan
    base = float((y[keep] == major).mean()) if keep.any() and major is not None else nan
    return {"coverage": float(keep.mean()), "acc_on_covered": acc, "majority_acc_on_covered": base}


# --- 메인 ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=config.ARTIFACTS)
    ap.add_argument("--aug", type=int, default=config.N_AUG, help="train 클립당 증강 사본 수")
    ap.add_argument("--no-pitch", dest="pitch", action="store_false", help="피치 변환 증강 끄기")
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--holdout-source", default=None, help="이 출처는 CV에서 빼고 최종 모델로만 평가")
    ap.add_argument("--dup-sim", type=float, default=0.995, help="유사 중복 병합 코사인 임계값")
    args = ap.parse_args()
    set_seed(config.SEED)

    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold

    labels = config.CLASSES
    n_cls = len(labels)
    lab2i = {l: i for i, l in enumerate(labels)}

    df, clean, aug, hold = load_clips(args)
    y_clip = df.label.map(lab2i).to_numpy()
    groups = df.group.to_numpy()
    print("\n학습 대상 클립:", dict(df.label.value_counts()), "| group", len(set(groups)))

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=config.SEED)
    folds, cm_total = [], np.zeros((n_cls, n_cls), int)
    for f, (tr, te) in enumerate(sgkf.split(np.zeros(len(df)), y_clip, groups)):
        tr_fit, va = inner_val_split(tr, y_clip, groups, config.SEED + f)
        Xtr, ytr, _ = stack(tr_fit, clean, aug, y_clip, with_aug=True)
        Xva, yva, _ = stack(va, clean, aug, y_clip, with_aug=False)
        Xte, yte, cte = stack(te, clean, aug, y_clip, with_aug=False)

        mlp = fit_mlp(Xtr, ytr, Xva, yva, n_cls)
        P = mlp.predict(Xte, verbose=0)
        Plr = fit_logreg(Xtr, ytr).predict_proba(Xte)
        major = int(np.bincount(ytr, minlength=n_cls).argmax())

        ids, Pc = clip_probs(P, cte)
        _, Pc_lr = clip_probs(Plr, cte)
        yc = y_clip[ids]
        r = {"fold": f, "n_test_clips": int(len(ids)), "n_test_segments": int(len(yte)),
             "test_clips_per_class": np.bincount(yc, minlength=n_cls).tolist(),
             "segment": {"mlp": metrics(yte, P.argmax(1), n_cls),
                         "logreg": metrics(yte, Plr.argmax(1), n_cls),
                         "majority": metrics(yte, np.full_like(yte, major), n_cls)},
             "clip": {"mlp": metrics(yc, Pc.argmax(1), n_cls),
                      "logreg": metrics(yc, Pc_lr.argmax(1), n_cls),
                      "majority": metrics(yc, np.full_like(yc, major), n_cls),
                      "mlp_selective": selective(yc, Pc, config.CONF_MIN, major)}}
        cm_total += confusion_matrix(yc, Pc.argmax(1), labels=range(n_cls))
        folds.append(r)
        print(f"[fold {f}] clip Macro-F1  MLP {r['clip']['mlp']['macro_f1']:.3f} | "
              f"LogReg {r['clip']['logreg']['macro_f1']:.3f} | "
              f"최빈 {r['clip']['majority']['macro_f1']:.3f}  (test {len(ids)} clips)")

    summary = summarize(folds)
    print_summary(summary, labels)

    # 최종 모델: 전체 데이터로 학습 (조기 종료용 group 검증셋만 분리)
    tr_fit, va = inner_val_split(np.arange(len(df)), y_clip, groups, config.SEED)
    Xtr, ytr, _ = stack(tr_fit, clean, aug, y_clip, with_aug=True)
    Xva, yva, _ = stack(va, clean, aug, y_clip, with_aug=False)
    final = fit_mlp(Xtr, ytr, Xva, yva, n_cls)

    holdout = evaluate_holdout(final, hold, labels) if len(hold) else None

    os.makedirs(args.out, exist_ok=True)
    final.save(os.path.join(args.out, "cry_head.keras"))
    export_tflite(final, os.path.join(args.out, "cry_head.tflite"))
    json.dump(labels, open(os.path.join(args.out, "labels.json"), "w"), ensure_ascii=False)
    meta = {"classes": labels, "seg_frames": config.SEG_FRAMES, "cry_min": config.CRY_MIN,
            "conf_min": config.CONF_MIN, "aug": args.aug, "pitch": args.pitch,
            "clips": {k: int(v) for k, v in df.label.value_counts().items()},
            "sources": {k: int(v) for k, v in df.source.value_counts().items()}, "seed": config.SEED}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), ensure_ascii=False, indent=2)
    json.dump({"folds": folds, "summary": summary, "holdout": holdout, "meta": meta},
              open(os.path.join(args.out, "cv_results.json"), "w"), ensure_ascii=False, indent=2)
    save_cm(cm_total, labels, os.path.join(args.out, "confusion_matrix.png"))
    write_report(summary, cm_total, labels, meta, holdout, os.path.join(args.out, "CV_REPORT.md"))
    print(f"\n저장 완료 → {args.out}/")


def summarize(folds):
    out = {}
    for level in ("segment", "clip"):
        out[level] = {}
        for model in ("mlp", "logreg", "majority"):
            vals = {k: [f[level][model][k] for f in folds] for k in ("macro_f1", "balanced_acc", "acc")}
            rec = np.array([f[level][model]["recall"] for f in folds])
            out[level][model] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                 for k, v in vals.items()}
            out[level][model]["recall_mean"] = rec.mean(axis=0).tolist()
    sel = [f["clip"]["mlp_selective"] for f in folds]
    out["clip"]["mlp_selective"] = {
        "coverage": float(np.mean([s["coverage"] for s in sel])),
        "acc_on_covered": float(np.nanmean([s["acc_on_covered"] for s in sel])),
        "majority_acc_on_covered": float(np.nanmean([s["majority_acc_on_covered"] for s in sel]))}
    return out


def print_summary(s, labels):
    print("\n=== 교차검증 요약 (평균 ± 표준편차) ===")
    for level in ("segment", "clip"):
        for model in ("mlp", "logreg", "majority"):
            m = s[level][model]
            rec = " ".join(f"{l}={v:.2f}" for l, v in zip(labels, m["recall_mean"]))
            print(f"{level:7s} {model:8s} Macro-F1 {m['macro_f1']['mean']:.3f}±{m['macro_f1']['std']:.3f} "
                  f"| BalAcc {m['balanced_acc']['mean']:.3f} | recall {rec}")
    sel = s["clip"]["mlp_selective"]
    print(f"판정 보류(conf<{config.CONF_MIN}) 적용 시: 커버리지 {sel['coverage']:.2f}, "
          f"판정한 것의 정확도 {sel['acc_on_covered']:.3f} (같은 구간 최빈 기준선 {sel['majority_acc_on_covered']:.3f})")


def evaluate_holdout(model, hold, labels):
    lab2i = {l: i for i, l in enumerate(labels)}
    ys, ps = [], []
    for _, r in hold.iterrows():
        cry, emb = features.file_frames(r.path)
        X, _ = features.segments(cry, emb)
        if len(X):
            ps.append(model.predict(X, verbose=0).mean(axis=0))
            ys.append(lab2i[r.label])
    if not ys:
        return None
    ys, ps = np.array(ys), np.stack(ps)
    res = {"clips": int(len(ys)), "clip": metrics(ys, ps.argmax(1), len(labels)),
           "selective": selective(ys, ps, config.CONF_MIN, int(np.bincount(ys).argmax()))}
    print(f"\n[holdout] {len(ys)} clips  Macro-F1 {res['clip']['macro_f1']:.3f}")
    return res


def export_tflite(model, path):
    """헤드만 tflite로 (ECO.AI/LiteRT 온디바이스용). 입력: 1024-d 세그먼트 임베딩."""
    import tensorflow as tf
    open(path, "wb").write(tf.lite.TFLiteConverter.from_keras_model(model).convert())


def save_cm(cm, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Clip-level, all CV folds")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write_report(s, cm, labels, meta, holdout, path):
    L = ["# 울음 이유 3클래스 분류 — 교차검증 결과", "",
         f"- 클래스: {', '.join(labels)}",
         f"- 클립 수: {meta['clips']}",
         f"- 출처: {meta['sources']}",
         f"- 평가: StratifiedGroupKFold (같은 아기는 한 fold에만), 증강 {meta['aug']}개/클립, "
         f"pitch={'on' if meta['pitch'] else 'off'}, seed {meta['seed']}", "",
         "| 단위 | 모델 | Macro-F1 | Balanced Acc | Accuracy | "
         + " | ".join(f"recall {l}" for l in labels) + " |",
         "|---|---|---:|---:|---:|" + "---:|" * len(labels)]
    for level in ("segment", "clip"):
        for model in ("mlp", "logreg", "majority"):
            m = s[level][model]
            L.append(f"| {level} | {model} | {m['macro_f1']['mean']:.3f} ± {m['macro_f1']['std']:.3f} | "
                     f"{m['balanced_acc']['mean']:.3f} | {m['acc']['mean']:.3f} | "
                     + " | ".join(f"{v:.2f}" for v in m["recall_mean"]) + " |")
    sel = s["clip"]["mlp_selective"]
    L += ["", f"판정 보류(최고 확률 < {config.CONF_MIN}) 적용 시 커버리지 {sel['coverage']:.2f}, "
              f"판정한 것의 정확도 {sel['acc_on_covered']:.3f} "
              f"(같은 구간을 전부 최빈 클래스로 찍을 때 {sel['majority_acc_on_covered']:.3f})", "",
          "혼동행렬 (클립 단위, MLP, 전 fold 합산; 행=정답, 열=예측)", "",
          "| | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for i, l in enumerate(labels):
        L.append(f"| {l} | " + " | ".join(str(v) for v in cm[i]) + " |")
    if holdout:
        L += ["", f"Holdout 출처 평가: {holdout['clips']} clips, "
                  f"Macro-F1 {holdout['clip']['macro_f1']:.3f}"]
    L += ["", "## 한계", "",
          "- 라벨은 부모가 앱에서 직접 고른 자가 태깅이라 잡음이 크다(정답 자체가 불확실).",
          "- 공개 데이터 녹음 환경(스마트폰 근접)과 실제 침실 마이크 환경이 다르다.",
          "- 울음 이유 분류는 실험적 기능이며, 실사용 판단은 맥락(cry_context)과 함께 힌트로만 쓴다."]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
