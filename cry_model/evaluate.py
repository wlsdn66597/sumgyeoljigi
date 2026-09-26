"""저장된 모델을 '학습에 쓰지 않은' 라벨 폴더 데이터로 평가 (예: 직접 녹음한 울음).

    python evaluate.py <폴더> [--art artifacts]
    <폴더>/<라벨>/*.wav   라벨 폴더명은 config.LABEL_MAP으로 3클래스에 매핑된다.

교차검증 성능은 train.py가 보고한다(CV_REPORT.md). 여기서는 새 환경(실제 침실
마이크 등)으로의 일반화를 따로 확인한다.
"""
import argparse
import os

import numpy as np

import config
import features
from infer import CryModel
from train import metrics, selective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--art", default=config.ARTIFACTS)
    args = ap.parse_args()

    model = CryModel.load(args.art)
    lab2i = {l: i for i, l in enumerate(model.labels)}

    ys, ps, skipped = [], [], 0
    for folder in sorted(os.listdir(args.root)):
        label = config.LABEL_MAP.get(folder.lower())
        d = os.path.join(args.root, folder)
        if label is None or label not in lab2i or not os.path.isdir(d):
            continue            # 모델이 학습하지 않은 클래스(예: 2클래스 모델의 sleepy)는 제외
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".wav"):
                continue
            _, p = model.probs(features.load_wav(os.path.join(d, f)))
            if p is None:
                skipped += 1
                continue
            ys.append(lab2i[label])
            ps.append(p)
    if not ys:
        raise SystemExit("평가할 울음 클립이 없습니다.")

    ys, ps = np.array(ys), np.stack(ps)
    m = metrics(ys, ps.argmax(1), len(model.labels))
    sel = selective(ys, ps, model.conf_min, int(np.bincount(ys).argmax()))
    print(f"클립 {len(ys)}개 (울음 미검출 제외 {skipped}개)")
    print(f"Macro-F1 {m['macro_f1']:.3f} | Balanced Acc {m['balanced_acc']:.3f} | Acc {m['acc']:.3f}")
    print("recall:", {l: round(r, 3) for l, r in zip(model.labels, m["recall"])})
    print(f"판정 보류 적용: 커버리지 {sel['coverage']:.2f}, 판정한 것의 정확도 {sel['acc_on_covered']:.3f} "
          f"(같은 구간 최빈 기준선 {sel['majority_acc_on_covered']:.3f})")


if __name__ == "__main__":
    main()
