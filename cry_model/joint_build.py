"""여러 울음 이유 데이터를 하나의 라벨 체계로 묶어 GPU 서버용 학습 세트를 만든다(로컬에서 실행).

    python joint_build.py            # DATA_ROOT/joint/ 에 wav(16kHz 모노) + meta.csv + meta_unlab.csv + aug/ 생성
                                     # 이미 만든 wav는 다시 쓰지 않는다(같은 순서로 같은 id가 붙는다)

통합 라벨 (데이터마다 가진 클래스가 다르다 → 학습 때 없는 클래스는 마스킹)
  hunger      DAC hungry, Enes F(먹여서 멈춤), Baidu hungry
  discomfort  DAC belly_pain·burping·discomfort·cold_hot, Enes I, Baidu uncomfortable·diaper, Corvin B(집 목욕)
  isolation   Enes S(안아줘서 멈춤), Baidu hug
  sleepy      DAC tired, Baidu sleepy
  pain        Corvin D(병원 예방접종) — 녹음 장소가 함께 바뀌는 지름길이 있다(배경만으로 AUC 0.71)

Kaggle Cry Sense 출처 불명 47개는 뺐다: 37개가 DAC 클립의 재인코딩본(voc2vec 유사도 ≈1.0, 라벨 92% 일치)이라
섞으면 DAC 평가로 정답이 샌다(파일명·md5 중복 검사로는 못 잡았다).

라벨 없는 울음(meta_unlab.csv): 울음 도메인 추가 사전학습용
  Infant cry Dataset 울음, ESC-50 아기 울음, Baidu test·awake, Enes 통증·모름, IFPaLVD 병원 영상 소리
증강(aug/): OpenSLR 28 실측·시뮬레이션 RIR, ESC-50(아기 울음 제외)·OpenSLR 28 잡음

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
CLASSES = ["hunger", "discomfort", "isolation", "sleepy", "pain"]
DAC_MAP = {"hungry": "hunger", "discomfort": "discomfort", "sleepy": "sleepy"}
ENES_MAP = {"F": "hunger", "I": "discomfort", "S": "isolation"}
CORVIN_MAP = {"B": "discomfort", "D": "pain"}
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


def _exists(ds, cid):
    """이미 쓴 wav면 (상대 경로, 길이)를, 없으면 None."""
    rel = f"wav/{ds}/{cid}.wav"
    p = os.path.join(OUT, rel)
    return (rel, sf.info(p).duration) if os.path.exists(p) else None


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
        rel, dur = _exists("dac", cid) or _write("dac", cid, features.load_wav(r.path))
        rows.append({"id": cid, "dataset": "dac", "path": rel, "dur": dur, "group": r.group,
                     "label": DAC_MAP[r.label], "orig_label": r.orig_label, "source": r.source})
        E.append(emb.mean(axis=0))
    print(f"DAC {len(rows)}개 (울음 없는 원본 {skipped}개 제외)")
    return rows, np.stack(E)


def _enes_groups(keep):
    groups = collections.defaultdict(list)
    for e in os.scandir(ENES_DIR):
        p = e.name[:-4].split("_")
        if len(p) != 8 or "-" not in p[7] or not keep(p[4][:1]):
            continue
        s, t = p[7].split("-")
        groups["_".join(p[:7])].append((int(s), int(t), e.name))
    return sorted(groups.items())


def _enes_wave(items):
    """음절 파일을 원래 시각에 배치해 한 울음으로 잇는다(16kHz)."""
    import librosa
    parts, sr, prev_end = [], None, None
    for s, t, n in sorted(items):
        x, sr = sf.read(os.path.join(ENES_DIR, n), dtype="float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        if prev_end is not None:      # 파일명의 시작·끝은 ms, 각 파일은 앞뒤 50ms 여유 포함
            gap = min(max(s - prev_end - 100, 0) / 1000.0, MAX_GAP_S)
            parts.append(np.zeros(int(gap * sr), np.float32))
        parts.append(x)
        prev_end = t
        if sum(len(q) for q in parts) / sr >= MAX_SEC:
            break
    return librosa.resample(np.concatenate(parts), orig_sr=sr, target_sr=config.SR)


def build_enes():
    groups = _enes_groups(lambda c: c in ENES_MAP)
    rows = []
    print(f"Enes 울음 {len(groups)}개 (음절을 원래 시각에 배치, 공백 최대 {MAX_GAP_S}s)")
    for i, (key, items) in enumerate(groups):
        p = key.split("_")
        cid = f"enes{i:04d}"
        rel, dur = _exists("enes", cid) or _write("enes", cid, _enes_wave(items))
        rows.append({"id": cid, "dataset": "enes", "path": rel, "dur": dur, "group": p[0],
                     "label": ENES_MAP[p[4][:1]], "orig_label": p[4], "source": key,
                     "cause_parent": p[3], "age": p[2], "date": p[5], "time": p[6]})
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
        rel, dur = _exists("baidu", cid) or _write("baidu", cid, features.load_wav(f))
        rows.append({"id": cid, "dataset": "baidu", "path": rel, "dur": dur, "group": f"bd{cl[k]}",
                     "label": BAIDU_MAP[lab], "orig_label": lab, "source": os.path.basename(f)})
    n_grp = len(set(r["group"] for r in rows))
    print(f"Baidu {len(rows)}개 (DAC와 거의 같은 {int(near_dac.sum())}개 제외), 유사 중복 묶음 후 그룹 {n_grp}개")
    return rows


def build_corvin():
    files = sorted(glob.glob(os.path.join(RES, "corvin", "*", "OpenData", "cry_sequences_22babies", "*.wav")))
    rows = []
    for k, f in enumerate(files):
        name = os.path.splitext(os.path.basename(f))[0]
        baby, code = name.split("_")[0], name.split("_")[1]
        cid = f"corvin{k:04d}"
        rel, dur = _exists("corvin", cid) or _write("corvin", cid, features.load_wav(f))
        rows.append({"id": cid, "dataset": "corvin", "path": rel, "dur": dur, "group": baby,
                     "label": CORVIN_MAP[code[0]], "orig_label": code, "source": name})
    print(f"Corvin {len(rows)}개, 아기 {len(set(r['group'] for r in rows))}명")
    return rows


def build_unlabeled(labeled_E):
    """라벨 없는 울음. 라벨 있는 클립과 거의 같은 것(cos≥0.99)은 뺀다(추가 사전학습이 평가 클립을 미리 보지 않게)."""
    srcs = []
    srcs += [("infantcry", f) for f in sorted(glob.glob(os.path.join(config.EXTRA_DIR, "infantcry", "**", "cry", "*.wav"),
                                                         recursive=True))]
    srcs += [("esc50_baby", f) for f in sorted(glob.glob(os.path.join(config.DATA_ROOT, "ESC-50-master", "audio", "*-20.wav")))]
    srcs += [("baidu_test", f) for f in sorted(glob.glob(os.path.join(RES, "baidu", "test", "*.wav")))]
    srcs += [("baidu_awake", f) for f in sorted(glob.glob(os.path.join(RES, "baidu", "train", "awake", "*.wav")))]
    srcs += [("ifpalvd", f) for f in sorted(glob.glob(os.path.join(RES, "ifpalvd", "_wav", "*.wav")))]
    rows, dropped = [], 0
    Ln = _norm(labeled_E)
    for k, (src, f) in enumerate(srcs):
        e = features.file_frames(f)[1].mean(axis=0)
        if (Ln @ _norm(e[None]).T).max() >= DUP_COS:
            dropped += 1
            continue
        cid = f"u{k:05d}"
        rel, dur = _exists("unlab", cid) or _write("unlab", cid, features.load_wav(f))
        rows.append({"id": cid, "dataset": src, "path": rel, "dur": dur, "group": f"{src}:{k}",
                     "source": os.path.basename(f)})
    for i, (key, items) in enumerate(_enes_groups(lambda c: c not in ENES_MAP)):   # Enes 통증·모름
        cid = f"uenes{i:04d}"
        rel, dur = _exists("unlab", cid) or _write("unlab", cid, _enes_wave(items))
        rows.append({"id": cid, "dataset": "enes_other", "path": rel, "dur": dur, "group": key.split("_")[0],
                     "source": key})
    u = pd.DataFrame(rows)
    print(f"라벨 없는 울음 {len(u)}개, {u.dur.sum() / 3600:.2f}시간 (라벨 있는 클립과 거의 같은 {dropped}개 제외)")
    print(u.groupby("dataset").dur.agg(["count", "sum"]).round(0))
    return u


def build_aug(n_sim=2000, seed=config.SEED):
    """증강용 RIR·잡음을 16kHz 모노로 aug/ 에 모은다(서버로 옮기기 위해)."""
    import augment
    real, sim = augment._rir_sets()
    rng = np.random.default_rng(seed)
    sim = [sim[i] for i in sorted(rng.choice(len(sim), min(n_sim, len(sim)), replace=False))]
    for kind, files in (("rir_real", real), ("rir_sim", sim), ("noise", augment._noise_files())):
        d = os.path.join(OUT, "aug", kind)
        os.makedirs(d, exist_ok=True)
        for k, f in enumerate(files):
            p = os.path.join(d, f"{k:05d}.wav")
            if not os.path.exists(p):
                sf.write(p, augment._load_first_channel(f), config.SR, subtype="PCM_16")
        print(f"aug/{kind}: {len(files)}개")


def main():
    os.makedirs(OUT, exist_ok=True)
    dac, dac_E = build_dac()
    rows = dac + build_enes() + build_baidu(dac_E) + build_corvin()
    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(OUT, "meta.csv"), index=False)
    print(meta.groupby(["dataset", "label"]).size().unstack(fill_value=0))
    print(f"총 {len(meta)}개, {meta.dur.sum() / 3600:.2f}시간 → {OUT}")
    # 중복 검사는 원본 경로의 YAMNet 캐시로 한다(재포장 가능성이 있는 DAC·Baidu·Corvin 기준)
    src = sorted(f for f in glob.glob(os.path.join(RES, "baidu", "train", "*", "*.wav"))
                 if os.path.basename(os.path.dirname(f)) in BAIDU_MAP)       # awake는 라벨 없는 쪽
    src += sorted(glob.glob(os.path.join(RES, "corvin", "*", "OpenData", "cry_sequences_22babies", "*.wav")))
    labeled_E = np.concatenate([dac_E, np.stack([features.file_frames(f)[1].mean(axis=0) for f in src])])
    build_unlabeled(labeled_E).to_csv(os.path.join(OUT, "meta_unlab.csv"), index=False)
    build_aug()


if __name__ == "__main__":
    main()
