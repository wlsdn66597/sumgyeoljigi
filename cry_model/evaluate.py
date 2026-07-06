"""저장된 모델을 held-out 테스트셋에서 재평가하고 보고서용 지표/그림 생성.

실행: python evaluate.py <데이터루트> [--art artifacts]
동일 seed로 train.py와 같은 그룹 분할을 재현해 test셋만 평가한다.
"""
import argparse
import numpy as np

import features
import prepare_data
from infer import CryModel
from train import load_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--art", default="artifacts")
    args = ap.parse_args()

    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import classification_report, f1_score

    df = prepare_data.build_index(args.root)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(gss.split(df, groups=df.group))
    df_te = df.iloc[te_idx]

    model = CryModel.load(args.art)
    lab2i = {l: i for i, l in enumerate(model.labels)}

    y_true, y_pred = [], []
    for _, r in df_te.iterrows():
        _, reason, _ = model.predict(load_wav(r.path))
        y_true.append(lab2i[r.label]); y_pred.append(lab2i[reason])

    print(classification_report(y_true, y_pred, target_names=model.labels, digits=3))
    print("Macro-F1:", round(f1_score(y_true, y_pred, average="macro"), 3))


if __name__ == "__main__":
    main()
