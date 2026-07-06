"""울음 '이유 분류' 헤드 학습 (YAMNet 임베딩 전이학습).

실행: python train.py <데이터루트> [--out artifacts] [--aug 2]
산출물: artifacts/cry_head.keras, artifacts/labels.json, artifacts/confusion_matrix.png
"""
import os
import json
import argparse
import numpy as np
import librosa

import features
import augment
import prepare_data

SR = 16000


def load_wav(path):
    w, _ = librosa.load(path, sr=SR, mono=True)
    return w


def build_xy(df, n_aug=0):
    X, y = [], []
    for _, r in df.iterrows():
        X.append(features.embed_file(r.path)); y.append(r.label)   # clean = 캐시 사용
        if n_aug:
            w = load_wav(r.path)                                   # 증강 시에만 로드
            for _ in range(n_aug):
                X.append(features.embed(augment.random_augment(w, SR))); y.append(r.label)
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--aug", type=int, default=2, help="train 클립당 증강 사본 수")
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.utils.class_weight import compute_class_weight

    df = prepare_data.build_index(args.root)
    labels = sorted(df.label.unique())
    lab2i = {l: i for i, l in enumerate(labels)}

    # 화자/녹음(group) 단위 held-out 분할
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(df, groups=df.group))
    df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
    print(f"train {len(df_tr)} clips / test {len(df_te)} clips (그룹 분리)")

    print("임베딩 추출 중... (train, 증강 포함)")
    Xtr, ytr_s = build_xy(df_tr, n_aug=args.aug)
    print("임베딩 추출 중... (test)")
    Xte, yte_s = build_xy(df_te, n_aug=0)
    ytr = np.array([lab2i[s] for s in ytr_s])
    yte = np.array([lab2i[s] for s in yte_s])

    cw = compute_class_weight("balanced", classes=np.arange(len(labels)), y=ytr)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(labels), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, validation_split=0.1, epochs=40, batch_size=32,
              class_weight=class_weight, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])

    pred = model.predict(Xte).argmax(1)
    print("\n=== held-out 평가 ===")
    print(classification_report(yte, pred, target_names=labels, digits=3))
    print("Macro-F1:", round(f1_score(yte, pred, average="macro"), 3))

    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "cry_head.keras"))
    json.dump(labels, open(os.path.join(args.out, "labels.json"), "w"), ensure_ascii=False)
    _save_cm(confusion_matrix(yte, pred), labels, os.path.join(args.out, "confusion_matrix.png"))
    print(f"\n저장 완료 → {args.out}/")


def _save_cm(cm, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout(); fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
