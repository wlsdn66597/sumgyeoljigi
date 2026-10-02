"""여러 울음 이유 데이터를 하나의 라벨 체계로 묶어 GPU 서버용 학습 세트를 만든다(로컬에서 실행).

    python joint_build.py            # DATA_ROOT/joint/ 에 wav(16kHz 모노) + meta.csv 생성

통합 라벨 (데이터마다 가진 클래스가 다르다 → 학습 때 없는 클래스는 마스킹)
  hunger      DAC hungry, Enes F(먹여서 멈춤), Baidu hungry
  discomfort  DAC belly_pain·burping·discomfort·cold_hot, Enes I, Baidu uncomfortable·diaper
  isolation   Enes S(안아줘서 멈춤), Baidu hug
  sleepy      DAC tired, Baidu sleepy
  (Enes D 통증 4개·모름, Baidu awake 는 대응 클래스가 없어 제외)

그룹(학습·평가 분리 단위)
  DAC: 업로더 UUID, Enes: 아기, Baidu: 아기 정보가 없어 YAMNet 임베딩 유사 중복(cos≥0.99, complete linkage) 묶음
  Baidu 클립 중 DAC와 거의 같은 것(cos≥0.99)은 데이터 간 누수를 막기 위해 뺀다.
"""
import collections
import glob
import os

import numpy as np
import pandas as pd
import soundfile as sf

import config
import features

OUT = os.path.join(config.DATA_ROOT, "joint")
RES = os.path.join(config.DATA_ROOT, "research")
ENES_DIR = os.path.join(RES, "enes", "00_pooled_separate")
CLASSES = ["hunger", "discomfort", "isolation", "sleepy"]
DAC_MAP = {"hungry": "hunger", "discomfort": "discomfort", "sleepy": "sleepy"}
ENES_MAP = {"F": "hunger", "I": "discomfort", "S": "isolation"}
BAIDU_MAP = {"hungry": "hunger", "uncomfortable": "discomfort", "diaper": "discomfort",
             "hug": "isolation", "sleepy": "sleepy"}
DUP_COS = 0.99
MAX_GAP_S = 1.0    # Enes 음절 사이 공백 상한
MAX_SEC = 60.0     # 클립 길이 상한


def _norm(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def _clusters(E, thr=DUP_COS):
    """모든 쌍이 유사도 thr 이상인 묶음(complete linkage). union-find는 연쇄로 359개짜리 거대 묶음이 생겼다."""
    from scipy.cluster.hierarchy import fcluster, linkage
    return list(fcluster(linkage(_norm(E), method="complete", metric="cosine"), 1 - thr, criterion="distance"))


def _write(ds, cid, w):
    rel = f"wav/{ds}/{cid}.wav"
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    w = np.clip(w[: int(MAX_SEC * config.SR)], -1, 1)
    sf.write(p, (w * 32767).astype(np.int16), config.SR, subtype="PCM_16")
    return rel, len(w) / config.SR


def build_dac():
    man = pd.read_csv(config.MANIFEST)
    man = man[man.source.isin(["dac_cleaned", "dac_raw"])]
    rows, E, skipped = [], [], 0
    for r in man.itertuples():
        cry, emb = features.file_frames(r.path)
        X, _ = features.segments(cry, emb)
        if not len(X):             # 원본 버킷 중 울음이 없는 녹음
            skipped += 1
            continue
        cid = f"dac{len(rows):04d}"
        rel, dur = _write("dac", cid, features.load_wav(r.path))
        rows.append({"id": cid, "dataset": "dac", "path": rel, "dur": dur, "group": r.group,
                     "label": DAC_MAP[r.label], "orig_label": r.orig_label, "source": r.source})
        E.append(emb.mean(axis=0))
    print(f"DAC {len(rows)}개 (울음 없는 원본 {skipped}개 제외)")
    return rows, np.stack(E)


def build_enes():
    import librosa
    groups = collections.defaultdict(list)
    for e in os.scandir(ENES_DIR):
        p = e.name[:-4].split("_")
        if len(p) != 8 or "-" not in p[7] or p[4][:1] not in ENES_MAP:
            continue
        s, t = p[7].split("-")
        groups["_".join(p[:7])].append((int(s), int(t), e.name))
    rows = []
    print(f"Enes 울음 {len(groups)}개 재구성 중 (음절을 원래 시각에 배치, 공백 최대 {MAX_GAP_S}s)")
    for i, (key, items) in enumerate(sorted(groups.items())):
        items.sort()
        parts, sr, prev_end = [], None, None
        for s, t, n in items:
            x, sr = sf.read(os.path.join(ENES_DIR, n), dtype="float32")
            x = x.mean(axis=1) if x.ndim > 1 else x
            if prev_end is not None:      # 파일명의 시작·끝은 ms, 각 파일은 앞뒤 50ms 여유 포함
                gap = min(max(s - prev_end - 100, 0) / 1000.0, MAX_GAP_S)
                parts.append(np.zeros(int(gap * sr), np.float32))
            parts.append(x)
            prev_end = t
            if sum(len(q) for q in parts) / sr >= MAX_SEC:
                break
        w = librosa.resample(np.concatenate(parts), orig_sr=sr, target_sr=config.SR)
        p = key.split("_")
        cid = f"enes{i:04d}"
        rel, dur = _write("enes", cid, w)
        rows.append({"id": cid, "dataset": "enes", "path": rel, "dur": dur, "group": p[0],
                     "label": ENES_MAP[p[4][:1]], "orig_label": p[4], "source": key,
                     "cause_parent": p[3], "age": p[2], "date": p[5], "time": p[6]})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(groups)}")
    print(f"Enes {len(rows)}개, 아기 {len(set(r['group'] for r in rows))}명")
    return rows


def build_baidu(dac_E):
    files = sorted(glob.glob(os.path.join(RES, "baidu", "train", "*", "*.wav")))
    files = [f for f in files if os.path.basename(os.path.dirname(f)) in BAIDU_MAP]
    E = np.stack([features.file_frames(f)[1].mean(axis=0) for f in files])
    near_dac = (_norm(E) @ _norm(dac_E).T).max(axis=1) >= DUP_COS
    cl = _clusters(E)
    rows = []
    for k, f in enumerate(files):
        if near_dac[k]:
            continue
        lab = os.path.basename(os.path.dirname(f))
        cid = f"baidu{k:04d}"
        rel, dur = _write("baidu", cid, features.load_wav(f))
        rows.append({"id": cid, "dataset": "baidu", "path": rel, "dur": dur, "group": f"bd{cl[k]}",
                     "label": BAIDU_MAP[lab], "orig_label": lab, "source": os.path.basename(f)})
    n_grp = len(set(r["group"] for r in rows))
    print(f"Baidu {len(rows)}개 (DAC와 거의 같은 {int(near_dac.sum())}개 제외), 유사 중복 묶음 후 그룹 {n_grp}개")
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    dac, dac_E = build_dac()
    rows = dac + build_enes() + build_baidu(dac_E)
    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(OUT, "meta.csv"), index=False)
    print(meta.groupby(["dataset", "label"]).size().unstack(fill_value=0))
    print(f"총 {len(meta)}개, {meta.dur.sum() / 3600:.2f}시간 → {OUT}")


if __name__ == "__main__":
    main()
