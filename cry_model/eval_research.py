"""연구용 울음 데이터로 검증: 원거리 울음 감지, 원인 분류(아기 간·아기별), 통증 vs 불편.

    python eval_research.py enes    # EnesBabyCries: 가정 1~4m 녹음 울음 감지율 + 원인 분류
    python eval_research.py pain    # Corvin: 통증(예방접종) vs 불편(목욕) + 녹음 장소 지름길 점검
    python eval_research.py extra   # Enes 추가 실험: 깨끗한 라벨, 배고픔 vs 나머지, 리듬·맥락 특징
    python eval_research.py online  # 실제 사용 모사: 아기별 원인 이력이 시간순으로 쌓일 때 맥락 + 이력 결합
    python eval_research.py online_audio  # 위 결합에 대형 모델 소리 확률(joint_train.py 아기 분리 예측)을 더함
    python eval_research.py baidu   # iFLYTEK/Baidu 6클래스로 학습 → EnesBabyCries로 교차 평가(아기 완전 분리)
    python eval_research.py ifpalvd # IFPaLVD: 같은 환경에서 녹음된 울음의 통증 강도(심함 vs 중간) 구분

데이터 (DATA_ROOT/research/)
  enes/00_pooled_separate/  Lockhart-Bouron 외 2023, OSF ru7na. 아기 24명의 가정 녹음을
      음절 단위로 자른 파일. 이름 = 아기_성별_나이회차_원인(부모판단)_원인(울음을멈춘행동)_날짜_시각_구간.
      원인 코드 첫 글자: F 배고픔, I 불편, S 혼자 둠, D 통증 (논문은 '멈춘 행동' 기준 사용)
  corvin/*/OpenData/cry_sequences_22babies/  Corvin 외 2024, Zenodo 11061909 (CC BY 4.0).
      아기22명, 이름 = 아기_B01(불편: 집에서 목욕 중) / 아기_D01(통증: 병원 예방접종)

결과: DATA_ROOT/runs/research/{ENES,PAIN}_REPORT.md
"""
import argparse
import collections
import glob
import os
import re

import numpy as np
import pandas as pd
import soundfile as sf

import config
import features

RES = os.path.join(config.DATA_ROOT, "research")
OUT = os.path.join(config.DATA_ROOT, "runs", "research")
ENES_DIR = os.path.join(RES, "enes", "00_pooled_separate")
CAUSE = {"F": "hunger", "I": "discomfort", "S": "isolation", "D": "pain"}
CAUSES3 = ["discomfort", "hunger", "isolation"]
DET_THR = 0.3            # 실시간 울음 임계값
MAX_SEG_PER_ITEM = 20    # 아기 간 분류 학습 속도를 위해 울음 하나당 세그먼트 수 상한


# --- 데이터 적재 ------------------------------------------------------------
def enes_bouts(max_sec=120):
    """음절 파일을 긴 울음(bout) 단위로 모아 시간순으로 이어 붙이고 YAMNet 프레임을 계산(캐시)."""
    groups = collections.defaultdict(list)
    for n in os.listdir(ENES_DIR):
        p = n[:-4].split("_")
        if len(p) != 8 or "-" not in p[7]:
            continue
        groups["_".join(p[:7])].append((int(p[7].split("-")[0]), n))
    cache_dir = os.path.join(config.CACHE_DIR, "enes_bouts")
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    print(f"EnesBabyCries 긴 울음 {len(groups)}개 (음절을 시간순으로 이어 붙여 YAMNet, 최대 {max_sec}초)")
    for i, (key, items) in enumerate(sorted(groups.items())):
        cp = os.path.join(cache_dir, f"{key}_{max_sec}.npz")
        if os.path.exists(cp):
            z = np.load(cp)
            cry, emb = z["cry"], z["emb"]
        else:
            import librosa
            parts, total, sr = [], 0.0, None
            for _, n in sorted(items):
                x, sr = sf.read(os.path.join(ENES_DIR, n), dtype="float32")
                parts.append(x.mean(axis=1) if x.ndim > 1 else x)
                total += len(parts[-1]) / sr
                if total >= max_sec:
                    break
            w = librosa.resample(np.concatenate(parts), orig_sr=sr, target_sr=config.SR)
            cry, emb = features.frames(w)
            np.savez(cp, cry=cry, emb=emb)
        p = key.split("_")
        out.append({"key": key, "baby": p[0], "age": int(p[2]) if p[2].isdigit() else -1,
                    "cause": CAUSE.get(p[4][:1], "other"), "cause_parent": CAUSE.get(p[3][:1], "other"),
                    "cry": cry, "emb": emb})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(groups)}")
    return out


def corvin_items():
    files = sorted(glob.glob(os.path.join(RES, "corvin", "*", "OpenData", "cry_sequences_22babies", "*.wav")))
    out = []
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        baby, code = name.split("_")[0], name.split("_")[1]
        cry, emb = features.file_frames(p)
        out.append({"key": name, "baby": baby, "pain": int(code.startswith("D")), "cry": cry, "emb": emb})
    return out


def seg_of(item, cry_min=config.CRY_MIN, cap=None):
    X, c = features.segments(item["cry"], item["emb"], cry_min=cry_min)
    if cap and len(X) > cap:
        idx = np.linspace(0, len(X) - 1, cap).astype(int)
        X, c = X[idx], c[idx]
    return X, c


# --- 공통 모델·지표 ---------------------------------------------------------
def logreg():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"))


def leave_baby_out(items, y, Xs, n_cls):
    """아기 한 명씩 빼고 학습 → 그 아기의 울음(item) 단위 확률(세그먼트 평균)."""
    babies = np.array([it["baby"] for it in items])
    P = np.zeros((len(items), n_cls))
    for b in np.unique(babies):
        te = np.where(babies == b)[0]
        tr = np.where(babies != b)[0]
        Xtr = np.concatenate([Xs[i] for i in tr])
        ytr = np.concatenate([[y[i]] * len(Xs[i]) for i in tr])
        m = logreg().fit(Xtr, ytr)
        for i in te:
            P[i] = m.predict_proba(Xs[i]).mean(axis=0)
    return P


def cls_metrics(y, pred, n_cls):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    return {"acc": accuracy_score(y, pred), "bal_acc": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", labels=range(n_cls), zero_division=0)}


# --- EnesBabyCries ------------------------------------------------------------
def run_enes():
    bouts = enes_bouts()
    L = ["# EnesBabyCries 검증 (가정 녹음, 마이크 1~4m)", ""]

    # 1) 원거리 울음 감지율 — 실시간과 같은 1.5초 세그먼트 단위, 임계값 0.3
    def det(items):
        seg_rate, any_det = [], []
        for it in items:
            _, c = features.segments(it["cry"], it["emb"], cry_min=None)
            seg_rate.append(float((c >= DET_THR).mean()) if len(c) else 0.0)
            any_det.append(bool(len(c)) and float(c.max()) >= DET_THR)
        return np.mean(seg_rate), np.mean(any_det), len(items)

    man = pd.read_csv(config.MANIFEST)
    dac = [{"cry": c, "emb": e} for c, e in
           (features.file_frames(p) for p in man[man.source == "dac_cleaned"].path)]
    e_seg, e_any, e_n = det(bouts)
    d_seg, d_any, d_n = det(dac)
    L += ["## 1. 원거리 울음 감지 (임계값 0.3, 1.5초 세그먼트)", "",
          "| 데이터 | 울음 수 | 1.5초 구간 감지율 | 울음 단위 감지율(한 번이라도) |", "|---|---:|---:|---:|",
          f"| Donate-a-Cry (스마트폰 근접) | {d_n} | {d_seg:.3f} | {d_any:.3f} |",
          f"| EnesBabyCries (가정 1~4m) | {e_n} | {e_seg:.3f} | {e_any:.3f} |", ""]
    by_age = collections.defaultdict(list)
    for it in bouts:
        by_age[it["age"]].append(it)
    L += ["나이 회차별(1=생후 15일, 2=1.5개월, 3=2.5개월, 4=3.5개월) 1.5초 구간 감지율: "
          + ", ".join(f"{a}회차 {det(v)[0]:.3f}({len(v)})" for a, v in sorted(by_age.items()) if a > 0), ""]

    # 2) 원인 분류 — 아기 간 (leave-one-baby-out)
    items = [it for it in bouts if it["cause"] in CAUSES3]
    Xs, keep = [], []
    for i, it in enumerate(items):
        X, _ = seg_of(it, cap=MAX_SEG_PER_ITEM)
        if len(X):
            Xs.append(X)
            keep.append(i)
    items = [items[i] for i in keep]
    y = np.array([CAUSES3.index(it["cause"]) for it in items])
    counts = collections.Counter(it["cause"] for it in items)
    print(f"원인 분류: 울음 {len(items)}개 {dict(counts)}, 아기 {len(set(it['baby'] for it in items))}명")

    P = leave_baby_out(items, y, Xs, 3)
    m_plain = cls_metrics(y, P.argmax(1), 3)
    # 변형: 아기별 평균 임베딩을 빼서 '누구 목소리인지'를 지우고 분류 (라벨 없이 그 아기 울음만 있으면 가능)
    babies = np.array([it["baby"] for it in items])
    Xc = []
    for i, it in enumerate(items):
        mu = np.concatenate([Xs[j] for j in np.where(babies == it["baby"])[0]]).mean(axis=0)
        Xc.append(Xs[i] - mu)
    Pc = leave_baby_out(items, y, Xc, 3)
    m_cent = cls_metrics(y, Pc.argmax(1), 3)
    maj = collections.Counter(y).most_common(1)[0][0]
    m_maj = cls_metrics(y, np.full_like(y, maj), 3)
    L += ["## 2. 원인 분류: 아기 간 일반화 (한 아기씩 빼고 학습)", "",
          f"- 울음 {len(items)}개 {dict(counts)}, 아기 {len(set(babies))}명. 라벨 = 울음을 멈춘 부모 행동",
          "", "| 방법 | 정확도 | 균형 정확도 | Macro-F1 |", "|---|---:|---:|---:|",
          f"| 최빈 클래스 | {m_maj['acc']:.3f} | {m_maj['bal_acc']:.3f} | {m_maj['macro_f1']:.3f} |",
          f"| YAMNet + 로지스틱 회귀 | {m_plain['acc']:.3f} | {m_plain['bal_acc']:.3f} | {m_plain['macro_f1']:.3f} |",
          f"| + 아기별 평균 제거 | {m_cent['acc']:.3f} | {m_cent['bal_acc']:.3f} | {m_cent['macro_f1']:.3f} |",
          "", "무작위 균형 정확도 = 0.333. 논문(Lockhart-Bouron 2023)의 기계 분류 정확도는 36%.", ""]

    # 3) 원인 분류 — 아기별 (부모 피드백으로 '우리 아기 전용' 모델을 만든다는 가설 검증)
    E = np.stack([X.mean(axis=0) for X in Xs])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ages = np.array([it["age"] for it in items])
    res = {"loo": [0, 0, 0], "age": [0, 0, 0]}          # [맞음(모델), 맞음(최빈), 전체]
    for b in np.unique(babies):
        idx = np.where(babies == b)[0]
        for mode in ("loo", "age"):
            for i in idx:
                if mode == "loo":
                    tr = idx[idx != i]                                  # 같은 날 녹음이 학습에 섞일 수 있음(낙관적)
                else:
                    tr = idx[ages[idx] != ages[i]]                      # 다른 나이 회차로만 학습(엄격)
                classes = [c for c in range(3) if np.sum(y[tr] == c) >= 2]
                if len(classes) < 2 or y[i] not in classes:
                    continue
                cent = np.stack([E[tr][y[tr] == c].mean(axis=0) for c in classes])
                pred = classes[int(np.argmax(cent @ E[i]))]
                major = collections.Counter(y[tr]).most_common(1)[0][0]
                res[mode][0] += int(pred == y[i])
                res[mode][1] += int(major == y[i])
                res[mode][2] += 1
    L += ["## 3. 원인 분류: 아기별 학습 (같은 아기의 다른 울음으로 학습, 최근접 평균)", "",
          "| 분할 | 평가 울음 | 모델 정확도 | 그 아기 최빈 원인 찍기 |", "|---|---:|---:|---:|"]
    for mode, name in (("loo", "울음 하나씩 빼기 (같은 날 녹음 섞임, 낙관적)"),
                       ("age", "다른 나이 회차로만 학습 (엄격)")):
        ok, mj, n = res[mode]
        L.append(f"| {name} | {n} | {ok / max(n, 1):.3f} | {mj / max(n, 1):.3f} |")

    pain_bouts = [it for it in bouts if it["cause"] == "pain"]
    L += ["", f"통증(D) 긴 울음 {len(pain_bouts)}개: {[it['key'] for it in pain_bouts]}", "",
          "## 한계", "",
          "- 음절 파일을 이어 붙여 만든 울음이라 원래 울음의 쉼·리듬 정보가 빠졌다.",
          "- 원인 라벨은 울음을 멈춘 부모 행동이다(부모 판단과 75% 일치). 정답 자체에 잡음이 있다.",
          "- 아기별 학습은 아기당 울음 수가 적어(6~52개) 추정 오차가 크다."]
    report = "\n".join(L) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "ENES_REPORT.md"), "w", encoding="utf-8").write(report)
    print("\n" + report)
    return bouts


# --- Corvin 통증 vs 불편 -------------------------------------------------------
def run_pain():
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    items = corvin_items()
    y_all = np.array([it["pain"] for it in items])
    L = ["# 통증 vs 불편 울음 (Corvin 2024, 아기 22명)", "",
         f"- 불편(집·목욕) {int((y_all == 0).sum())}개, 통증(병원·예방접종) {int((y_all == 1).sum())}개",
         "- 평가: 한 아기씩 빼고 학습(leave-one-baby-out), 울음 단위 확률 = 세그먼트 평균", ""]

    def evaluate(name, seg_fn):
        its, Xs = [], []
        for it in items:
            X = seg_fn(it)
            if len(X):
                its.append(it)
                Xs.append(X)
        y = np.array([it["pain"] for it in its])
        if len(set(y)) < 2:
            L.append(f"| {name} | {len(its)} | - | - |")
            return None
        P = leave_baby_out(its, y, Xs, 2)[:, 1]
        auc = roc_auc_score(y, P)
        bal = balanced_accuracy_score(y, (P >= 0.5).astype(int))
        L.append(f"| {name} | {len(its)} | {auc:.3f} | {bal:.3f} |")
        return auc

    def background(it):
        """울음 점수가 낮은 프레임(울음 사이 쉼·배경) 평균. 녹음 장소 정보만 담긴다."""
        m = it["cry"] < 0.05
        return it["emb"][m].mean(axis=0, keepdims=True) if m.sum() >= 1 else np.zeros((0, 1024))

    L += ["| 입력 | 울음 수 | ROC-AUC | 균형 정확도 |", "|---|---:|---:|---:|"]
    auc_cry = evaluate("울음 세그먼트 (CRY_MIN 0.1)", lambda it: seg_of(it)[0])
    evaluate("강한 울음만 (0.5 이상)", lambda it: seg_of(it, cry_min=0.5)[0])
    auc_bg = evaluate("**배경만 (울음 점수 0.05 미만 프레임)**", background)
    L += ["", "배경만으로도 구분되면(AUC가 높으면) 모델이 울음이 아니라 녹음 장소(집 vs 병원)를 배운 것이다.", ""]

    # 외부 적용: Corvin 전체로 학습 → EnesBabyCries(가정, 같은 장비) · Donate-a-Cry
    Xtr = np.concatenate([seg_of(it)[0] for it in items if len(seg_of(it)[0])])
    ytr = np.concatenate([[it["pain"]] * len(seg_of(it)[0]) for it in items if len(seg_of(it)[0])])
    model = logreg().fit(Xtr, ytr)

    def pain_prob(item):
        X, _ = seg_of(item)
        return float(model.predict_proba(X)[:, 1].mean()) if len(X) else np.nan

    bouts = enes_bouts()
    by = collections.defaultdict(list)
    for it in bouts:
        by[it["cause"]].append(pain_prob(it))
    L += ["## 외부 데이터 적용 (Corvin 전체로 학습한 모델)", "",
          "| 데이터 | 울음 수 | 통증 확률 평균 | 통증 판정(≥0.5) 비율 |", "|---|---:|---:|---:|"]
    for c in ["pain", "hunger", "discomfort", "isolation"]:
        v = np.array([p for p in by.get(c, []) if not np.isnan(p)])
        if len(v):
            L.append(f"| Enes 가정 · {c} | {len(v)} | {v.mean():.3f} | {(v >= 0.5).mean():.3f} |")
    man = pd.read_csv(config.MANIFEST)
    dac = man[man.source.isin(["dac_cleaned", "dac_raw"])]
    dv = collections.defaultdict(list)
    for _, r in dac.iterrows():
        cry, emb = features.file_frames(r.path)
        p = pain_prob({"cry": cry, "emb": emb})
        if not np.isnan(p):
            dv[r.orig_label].append(p)
    for lab, v in sorted(dv.items()):
        v = np.array(v)
        L.append(f"| Donate-a-Cry · {lab} | {len(v)} | {v.mean():.3f} | {(v >= 0.5).mean():.3f} |")
    L += ["", "Enes는 모두 집에서 녹음됐으므로, 통증 울음(4개)이 다른 원인보다 높게 나오면 장소가 아니라",
          "울음 자체의 통증 단서를 잡았다는 약한 근거가 된다. 표본이 4개뿐이라 결론으로 쓰기는 어렵다.", ""]
    L += ["## 한계", "",
          "- 통증은 병원, 불편은 집에서 녹음돼 장소와 원인이 완전히 겹친다(위 배경 점검 참고).",
          "- 통증 88개(아기당 4개)로 표본이 작다."]
    report = "\n".join(L) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "PAIN_REPORT.md"), "w", encoding="utf-8").write(report)
    print("\n" + report)


# --- Enes 추가 실험 -----------------------------------------------------------
def enes_timing():
    """파일 이름의 구간(ms)만으로 긴 울음별 리듬 특징과 시작 시각을 만든다(오디오 불필요)."""
    groups = collections.defaultdict(list)
    for n in os.listdir(ENES_DIR):
        p = n[:-4].split("_")
        if len(p) != 8 or "-" not in p[7]:
            continue
        a, b = p[7].split("-")
        groups["_".join(p[:7])].append((int(a), int(b)))
    feats = {}
    for key, segs in groups.items():
        segs.sort()
        st = np.array([a for a, _ in segs], float)
        en = np.array([b for _, b in segs], float)
        dur = (en - st) / 1000
        gaps = np.clip((st[1:] - en[:-1]) / 1000, 0, None)
        span = max((en.max() - st.min()) / 1000, 1e-3)
        p = key.split("_")
        t = pd.to_datetime(p[5] + p[6], format="%d%m%Y%H%M", errors="coerce")
        feats[key] = {"n_syl": len(segs), "syl_med": float(np.median(dur)),
                      "syl_iqr": float(np.subtract(*np.percentile(dur, [75, 25]))),
                      "gap_med": float(np.median(gaps)) if len(gaps) else 0.0,
                      "voiced_ratio": float(dur.sum() / span), "span": float(span),
                      "rate": float(len(segs) / span), "time": t}
    return feats


def context_features(bouts, timing, cap_h=12.0):
    """맥락 특징: 시간대, 나이, 같은 아기의 직전 울음·직전 배고픔 울음 이후 경과 시간, 직전 원인.
    직전 울음의 원인은 실제 사용에서 부모 피드백으로 알 수 있는 정보다."""
    rows = {}
    by_baby = collections.defaultdict(list)
    for it in bouts:
        by_baby[it["baby"]].append(it)
    for b, its in by_baby.items():
        its = sorted(its, key=lambda it: (timing[it["key"]]["time"] if pd.notna(timing[it["key"]]["time"])
                                          else pd.Timestamp.min))
        last_t, last_hunger_t, last_cause = None, None, "none"
        for it in its:
            t = timing[it["key"]]["time"]
            h = t.hour + t.minute / 60 if pd.notna(t) else 12.0
            def since(prev):
                if prev is None or pd.isna(t):
                    return cap_h
                return float(min(max((t - prev).total_seconds() / 3600, 0), cap_h))
            rows[it["key"]] = {"hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
                               "age": it["age"], "h_since_prev": since(last_t),
                               "h_since_hunger": since(last_hunger_t),
                               **{f"prev_{c}": float(last_cause == c) for c in CAUSES3}}
            if pd.notna(t):
                last_t = t
                if it["cause"] == "hunger":
                    last_hunger_t = t
            last_cause = it["cause"]
    return rows


def leave_baby_out_tab(X, y, babies, n_cls):
    from sklearn.ensemble import RandomForestClassifier
    P = np.zeros((len(y), n_cls))
    for b in np.unique(babies):
        te, tr = babies == b, babies != b
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                   random_state=config.SEED, n_jobs=-1).fit(X[tr], y[tr])
        P[np.ix_(te, m.classes_)] = m.predict_proba(X[te])
    return P


def run_extra():
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    bouts = enes_bouts()
    timing = enes_timing()
    ctx = context_features(bouts, timing)
    items = [it for it in bouts if it["cause"] in CAUSES3]
    Xs, keep = [], []
    for i, it in enumerate(items):
        X, _ = seg_of(it, cap=MAX_SEG_PER_ITEM)
        if len(X):
            Xs.append(X)
            keep.append(i)
    items = [items[i] for i in keep]
    y = np.array([CAUSES3.index(it["cause"]) for it in items])
    babies = np.array([it["baby"] for it in items])
    L = ["# EnesBabyCries 추가 실험", "",
         f"- 울음 {len(items)}개, 아기 {len(set(babies))}명, 모두 한 아기씩 빼고 학습(leave-one-baby-out)",
         "- 3분류는 균형 정확도(무작위 0.333), 배고픔 vs 나머지는 ROC-AUC(무작위 0.5)", ""]

    # A) 깨끗한 라벨: 부모 판단 == 울음을 멈춘 행동
    clean = np.array([it["cause_parent"] == it["cause"] for it in items])
    P_all = leave_baby_out(items, y, Xs, 3)
    ci = np.where(clean)[0]
    P_clean = leave_baby_out([items[i] for i in ci], y[ci], [Xs[i] for i in ci], 3)
    L += ["## A. 라벨 잡음 줄이기 (부모 판단과 멈춘 행동이 일치한 울음만)", "",
          "| 데이터 | 울음 수 | 균형 정확도 |", "|---|---:|---:|",
          f"| 전체 | {len(y)} | {balanced_accuracy_score(y, P_all.argmax(1)):.3f} |",
          f"| 라벨 일치만 | {len(ci)} | {balanced_accuracy_score(y[ci], P_clean.argmax(1)):.3f} |", ""]

    # B) 배고픔 vs 나머지
    yh = (y == CAUSES3.index("hunger")).astype(int)

    # C) 리듬·맥락 특징
    rkeys = ["n_syl", "syl_med", "syl_iqr", "gap_med", "voiced_ratio", "span", "rate"]
    ckeys = ["hour_sin", "hour_cos", "age", "h_since_prev", "h_since_hunger"] + [f"prev_{c}" for c in CAUSES3]
    R = np.array([[timing[it["key"]][k] for k in rkeys] for it in items], float)
    C = np.array([[ctx[it["key"]][k] for k in ckeys] for it in items], float)
    A = np.stack([X.mean(axis=0) for X in Xs])            # 소리: 울음 단위 평균 임베딩
    from sklearn.decomposition import PCA
    A20 = PCA(n_components=20, random_state=config.SEED).fit_transform(A)
    sets = {"소리 (YAMNet, 로지스틱 회귀)": None, "리듬 (음절 길이·간격·속도)": R, "맥락 (시간대·경과 시간·직전 원인)": C,
            "리듬 + 맥락": np.hstack([R, C]), "소리(PCA 20) + 리듬 + 맥락": np.hstack([A20, R, C])}
    L += ["## B·C. 특징별 비교", "", "| 특징 | 3분류 균형 정확도 | 배고픔 vs 나머지 AUC |", "|---|---:|---:|"]
    for name, X in sets.items():
        if X is None:
            P = P_all
        else:
            P = leave_baby_out_tab(X, y, babies, 3)
        L.append(f"| {name} | {balanced_accuracy_score(y, P.argmax(1)):.3f} | "
                 f"{roc_auc_score(yh, P[:, CAUSES3.index('hunger')]):.3f} |")
    L += ["", "맥락 특징의 '직전 원인'은 실제 사용에서 부모 피드백으로 얻는 정보다.",
          "시각은 녹음 파일의 시작 시각이라 울음 발생 시각과 약간 다를 수 있다.", ""]
    report = "\n".join(L) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "ENES_EXTRA_REPORT.md"), "w", encoding="utf-8").write(report)
    print("\n" + report)


# --- 실제 사용 모사: 부모 피드백(원인 이력)이 시간순으로 쌓이는 상황 -------------------
def run_online():
    """각 아기의 울음을 시간순으로 보며, 그 시점 이전의 정보만 쓴다.
    - 일반 맥락 모델: 다른 아기들로 학습한 리듬+맥락 RF (leave-one-baby-out)
    - 아기 이력: 그 아기의 이전 울음 원인 빈도(부모가 '왜 울었나요?'에 답한 결과에 해당)
    - 결합: 두 확률을 곱해 정규화
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    bouts = [it for it in enes_bouts() if it["cause"] in CAUSES3]
    timing = enes_timing()
    ctx = context_features(enes_bouts(), timing)
    rkeys = ["n_syl", "syl_med", "syl_iqr", "gap_med", "voiced_ratio", "span", "rate"]
    ckeys = ["hour_sin", "hour_cos", "age", "h_since_prev", "h_since_hunger"] + [f"prev_{c}" for c in CAUSES3]
    X = np.array([[timing[b["key"]][k] for k in rkeys] + [ctx[b["key"]][k] for k in ckeys] for b in bouts], float)
    y = np.array([CAUSES3.index(b["cause"]) for b in bouts])
    babies = np.array([b["baby"] for b in bouts])
    Pg = leave_baby_out_tab(X, y, babies, 3)

    tkey = lambda b: timing[b["key"]]["time"] if pd.notna(timing[b["key"]]["time"]) else pd.Timestamp.min
    rows = []
    for bb in np.unique(babies):
        idx = sorted(np.where(babies == bb)[0], key=lambda i: tkey(bouts[i]))
        counts = np.zeros(3)
        for k, i in enumerate(idx):
            prior = (counts + 1.0) / (counts.sum() + 3.0)          # 라플라스 평활
            comb = Pg[i] * prior
            comb = comb / comb.sum()
            rows.append({"k": k, "y": y[i], "g": Pg[i], "p": prior, "c": comb})
            counts[y[i]] += 1                                      # 이 울음 뒤 부모 피드백이 들어온다

    def report(sel, name):
        yy = np.array([r["y"] for r in sel])
        yh = (yy == CAUSES3.index("hunger")).astype(int)
        out = []
        for key, label in (("g", "일반 맥락 모델"), ("p", "아기 이력만"), ("c", "맥락 + 아기 이력")):
            P = np.stack([r[key] for r in sel])
            out.append(f"| {name} | {label} | {len(sel)} | {balanced_accuracy_score(yy, P.argmax(1)):.3f} | "
                       f"{roc_auc_score(yh, P[:, CAUSES3.index('hunger')]):.3f} |")
        return out

    L = ["# 실제 사용 모사: 맥락 + 아기별 원인 이력 (EnesBabyCries)", "",
         "각 아기의 울음을 시간순으로 보며 그 시점 이전 정보만 사용. 아기 이력 = 이전 울음들의 원인 빈도",
         "(부모 피드백으로 쌓인다고 가정). 일반 맥락 모델은 다른 아기들로만 학습.", "",
         "| 평가 구간 | 방법 | 울음 수 | 3분류 균형 정확도 | 배고픔 AUC |", "|---|---|---:|---:|---:|"]
    L += report(rows, "전체")
    L += report([r for r in rows if r["k"] >= 5], "이력 5개 이상 쌓인 뒤")
    L += report([r for r in rows if r["k"] >= 10], "이력 10개 이상 쌓인 뒤")
    L += ["", "무작위: 균형 정확도 0.333, AUC 0.5"]
    report_txt = "\n".join(L) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "ONLINE_REPORT.md"), "w", encoding="utf-8").write(report_txt)
    print("\n" + report_txt)


def run_online_audio():
    """대형 모델 소리 확률을 맥락 + 아기 이력에 더하면 오르는지(실제 사용 모사와 같은 시간순 절차).
    소리 확률은 joint_train.py가 아기 분리 5겹으로 만든 예측(OOF)이라 평가 아기를 본 적 없는 모델의 출력이다.
    입력: OUT/enes_audio_oof.csv (key, stage, model, cond, hunger, discomfort, isolation)"""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    A = pd.read_csv(os.path.join(OUT, "enes_audio_oof.csv"))
    bouts = [it for it in enes_bouts() if it["cause"] in CAUSES3]
    timing = enes_timing()
    ctx = context_features(enes_bouts(), timing)
    rkeys = ["n_syl", "syl_med", "syl_iqr", "gap_med", "voiced_ratio", "span", "rate"]
    ckeys = ["hour_sin", "hour_cos", "age", "h_since_prev", "h_since_hunger"] + [f"prev_{c}" for c in CAUSES3]
    X = np.array([[timing[b["key"]][k] for k in rkeys] + [ctx[b["key"]][k] for k in ckeys] for b in bouts], float)
    y = np.array([CAUSES3.index(b["cause"]) for b in bouts])
    babies = np.array([b["baby"] for b in bouts])
    keys = [b["key"] for b in bouts]
    variants = {"맥락 + 아기 이력 (기존 최고)": leave_baby_out_tab(X, y, babies, 3)}
    audio = {}
    for (stage, model, cond), g in A.groupby(["stage", "model", "cond"]):
        g = g.drop_duplicates("key").set_index("key")
        if not set(keys) <= set(g.index):
            continue
        Pa = g.loc[keys, CAUSES3].values
        audio[f"{stage} {model} {cond}"] = Pa
        variants[f"+ 소리 {stage} {model} {cond}"] = leave_baby_out_tab(np.hstack([X, Pa]), y, babies, 3)

    tkey = lambda b: timing[b["key"]]["time"] if pd.notna(timing[b["key"]]["time"]) else pd.Timestamp.min
    order = {bb: sorted(np.where(babies == bb)[0], key=lambda i: tkey(bouts[i])) for bb in np.unique(babies)}

    def online(Pg):
        """아기 이력(라플라스 평활 원인 빈도)과 곱해 시간순 예측. 반환: 예측 확률, 이력 개수."""
        P, K = np.zeros_like(Pg), np.zeros(len(y), int)
        for bb, idx in order.items():
            counts = np.zeros(3)
            for k, i in enumerate(idx):
                c = Pg[i] * (counts + 1.0) / (counts.sum() + 3.0)
                P[i], K[i] = c / c.sum(), k
                counts[y[i]] += 1
        return P, K

    h = CAUSES3.index("hunger")
    base_P, K = online(variants["맥락 + 아기 이력 (기존 최고)"])
    rng = np.random.default_rng(config.SEED)
    ub = np.unique(babies)
    bi = {b: np.where(babies == b)[0] for b in ub}
    boots = [np.concatenate([bi[b] for b in rng.choice(ub, len(ub))]) for _ in range(1000)]

    def score(P, ix):
        return balanced_accuracy_score(y[ix], P[ix].argmax(1)), roc_auc_score(y[ix] == h, P[ix, h])
    L = ["# 소리(대형 모델) + 맥락 + 아기 이력 (EnesBabyCries, 시간순)", "",
         "- 소리 확률: joint_train.py 아기 분리 5겹 예측. 맥락 RF: 한 아기씩 빼고 학습. 이력: 그 아기의 이전 원인 빈도.",
         "- Δ는 기존 최고(맥락 + 아기 이력) 대비, 괄호는 아기 단위 부트스트랩 95% 구간.", "",
         "| 방법 | 전체 균형 정확도 | 전체 배고픔 AUC | 이력 10개+ 균형 정확도 | Δ균형 정확도(전체) |",
         "|---|---:|---:|---:|---|"]
    for name, Pg in variants.items():
        P, _ = online(Pg)
        ten = np.where(K >= 10)[0]
        b_all, a_all = score(P, np.arange(len(y)))
        b10, _ = score(P, ten)
        d = [score(P, ix)[0] - score(base_P, ix)[0] for ix in boots]
        lo, hi = np.percentile(d, [2.5, 97.5])
        L.append(f"| {name} | {b_all:.3f} | {a_all:.3f} | {b10:.3f} | "
                 f"{b_all - score(base_P, np.arange(len(y)))[0]:+.3f} ({lo:+.2f}~{hi:+.2f}) |")
    L += ["", "## 참고: 소리 확률만 (아기 이력 없이)", "", "| 소리 모델 | 균형 정확도 | 배고픔 AUC |", "|---|---:|---:|"]
    for name, Pa in audio.items():
        b_, a_ = score(Pa, np.arange(len(y)))
        L.append(f"| {name} | {b_:.3f} | {a_:.3f} |")
    _save_report("ONLINE_AUDIO_REPORT.md", L)


# --- iFLYTEK/Baidu 6클래스 → EnesBabyCries 교차 평가 ---------------------------------
BAIDU_MAP = {"hungry": "hunger", "hug": "isolation", "uncomfortable": "discomfort", "diaper": "discomfort"}


def baidu_items():
    files = sorted(glob.glob(os.path.join(RES, "baidu", "**", "train", "*", "*.wav"), recursive=True))
    out = []
    for p in files:
        cry, emb = features.file_frames(p)
        out.append({"key": p, "label": os.path.basename(os.path.dirname(p)), "cry": cry, "emb": emb})
    return out


def run_baidu():
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    items = baidu_items()
    labels6 = sorted(set(it["label"] for it in items))
    L = ["# iFLYTEK/Baidu 울음 6클래스 → EnesBabyCries 교차 평가", "",
         f"- Baidu 학습 데이터 {len(items)}개: {dict(collections.Counter(it['label'] for it in items))}",
         "- 파일명에 아기 정보가 없어서 Baidu 내부 평가는 같은 아기가 학습·평가에 섞일 수 있다(낙관적).", ""]

    # 0) 중복 점검: Baidu 내부, Donate-a-Cry와의 겹침
    E = np.stack([it["emb"].mean(axis=0) for it in items])
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    S = En @ En.T
    np.fill_diagonal(S, 0)
    man = pd.read_csv(config.MANIFEST)
    dac = [features.file_frames(pth)[1].mean(axis=0) for pth in man[man.source == "dac_cleaned"].path]
    D = np.stack(dac)
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    near_dac = int(((En @ D.T).max(axis=1) >= 0.99).sum())
    L += [f"- 내부 유사 중복(cos≥0.99) 클립: {int((S.max(axis=1) >= 0.99).sum())}개, "
          f"Donate-a-Cry와 거의 같은 클립: {near_dac}개", ""]

    # 1) Baidu 내부 6클래스 5겹 (아기 정보 없음 → 낙관적)
    Xs, keep = [], []
    for i, it in enumerate(items):
        X, _ = seg_of(it, cry_min=None, cap=MAX_SEG_PER_ITEM)
        if len(X):
            Xs.append(X)
            keep.append(i)
    items = [items[i] for i in keep]
    y6 = np.array([labels6.index(it["label"]) for it in items])
    P6 = np.zeros((len(items), len(labels6)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=config.SEED).split(np.zeros(len(y6)), y6):
        m = logreg().fit(np.concatenate([Xs[i] for i in tr]), np.concatenate([[y6[i]] * len(Xs[i]) for i in tr]))
        for i in te:
            P6[i] = m.predict_proba(Xs[i]).mean(axis=0)
    L += ["## 1. Baidu 내부 6클래스 (5겹, 아기 분리 불가 → 낙관적)", "",
          f"균형 정확도 {balanced_accuracy_score(y6, P6.argmax(1)):.3f} (무작위 {1 / len(labels6):.3f})", ""]

    # 2) Baidu(3클래스로 매핑)로 학습 → Enes 평가
    mi = [i for i, it in enumerate(items) if it["label"] in BAIDU_MAP]
    yb = np.array([CAUSES3.index(BAIDU_MAP[items[i]["label"]]) for i in mi])
    Xb = np.concatenate([Xs[i] for i in mi])
    ybs = np.concatenate([[yb[k]] * len(Xs[i]) for k, i in enumerate(mi)])
    model = logreg().fit(Xb, ybs)
    bouts = [it for it in enes_bouts() if it["cause"] in CAUSES3]
    eX, ey = [], []
    for it in bouts:
        X, _ = seg_of(it, cap=MAX_SEG_PER_ITEM)
        if len(X):
            eX.append(X)
            ey.append(CAUSES3.index(it["cause"]))
    ey = np.array(ey)
    Pe = np.stack([model.predict_proba(X).mean(axis=0) for X in eX])
    yh = (ey == CAUSES3.index("hunger")).astype(int)
    cm = confusion_matrix(ey, Pe.argmax(1), labels=range(3))
    L += ["## 2. Baidu로 학습 → EnesBabyCries로 평가 (아기·녹음 환경 완전 분리)", "",
          "매핑: hungry→배고픔, hug→혼자 둠(안아주길 원함), uncomfortable·diaper→불편. awake·sleepy는 대응 클래스가 없어 제외.", "",
          f"- Enes 울음 {len(ey)}개: 3분류 균형 정확도 **{balanced_accuracy_score(ey, Pe.argmax(1)):.3f}** (무작위 0.333), "
          f"배고픔 AUC **{roc_auc_score(yh, Pe[:, CAUSES3.index('hunger')]):.3f}** (무작위 0.5)",
          "", "혼동행렬 (행=Enes 정답, 열=예측; " + ", ".join(CAUSES3) + ")", ""]
    L += ["    " + " ".join(f"{v:4d}" for v in row) for row in cm]

    # 3) 반대 방향: Enes로 학습 → Baidu 평가
    m2 = logreg().fit(np.concatenate(eX), np.concatenate([[ey[k]] * len(X) for k, X in enumerate(eX)]))
    Pb = np.stack([m2.predict_proba(Xs[i]).mean(axis=0) for i in mi])
    ybh = (yb == CAUSES3.index("hunger")).astype(int)
    L += ["", "## 3. 반대 방향: Enes로 학습 → Baidu 평가", "",
          f"- Baidu 클립 {len(mi)}개: 균형 정확도 {balanced_accuracy_score(yb, Pb.argmax(1)):.3f}, "
          f"배고픔 AUC {roc_auc_score(ybh, Pb[:, CAUSES3.index('hunger')]):.3f}", "",
          "## 한계", "",
          "- 두 데이터의 원인 정의가 다르다(Baidu 라벨 기준 불명, Enes는 울음을 멈춘 부모 행동).",
          "- Baidu 미러의 원 대회 약관이 불분명해 연구 확인용으로만 사용했다."]
    _save_report("BAIDU_REPORT.md", L)


# --- IFPaLVD: 같은 환경에서의 통증 강도 ---------------------------------------------
def ifpalvd_items():
    root = os.path.join(RES, "ifpalvd")
    lab = pd.read_csv(os.path.join(root, "labels.csv"), sep=";")
    # 공개 아카이브의 파일명은 인도네시아어 'kasus'(=case)로 시작한다. 라벨은 253개 구간 기준이지만
    # 공개된 영상은 (아기, 시술 전/후)마다 10초 구간 1개씩 47개뿐이다.
    vids = {os.path.basename(v).replace("kasus", "case"): v
            for v in glob.glob(os.path.join(root, "**", "*.avi"), recursive=True)}
    wav_dir = os.path.join(root, "_wav")
    os.makedirs(wav_dir, exist_ok=True)
    out, missing = [], 0
    import subprocess
    for _, r in lab.iterrows():
        v = vids.get(r.filename)
        if v is None:
            missing += 1
            continue
        w = os.path.join(wav_dir, os.path.splitext(r.filename)[0].replace(" ", "_") + ".wav")
        if not os.path.exists(w):
            subprocess.run([config.FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", v, "-vn",
                            "-ac", "1", "-ar", str(config.SR), w], capture_output=True)
        if not os.path.exists(w) or os.path.getsize(w) < 1000:
            missing += 1
            continue
        cry, emb = features.file_frames(w)
        case = int(re.search(r"case\s*(\d+)", r.filename).group(1))
        out.append({"key": r.filename, "baby": f"case{case}", "pain": r.pain_level, "cry_label": r.cry_label,
                    "phase": r.filename.split("_")[1], "wav": w, "cry": cry, "emb": emb})
    return out, missing


def run_ifpalvd():
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    import soundfile as sf_
    items, missing = ifpalvd_items()
    L = ["# IFPaLVD: 통증 강도와 울음 (아기 27명, 시술 전후 영상의 소리)", "",
         f"- 사용 클립 {len(items)}개 (영상·오디오 없음 {missing}개)",
         f"- {dict(collections.Counter((it['pain'], it['cry_label']) for it in items))}", ""]

    # 0) 울음 감지 확인: YAMNet 울음 점수로 Cry vs No Cry
    yc = np.array([it["cry_label"] == "Cry" for it in items]).astype(int)
    sc = np.array([float(it["cry"].max()) if len(it["cry"]) else 0.0 for it in items])
    L += [f"## 0. 울음 감지 (YAMNet 최대 울음 점수): Cry vs No Cry AUC {roc_auc_score(yc, sc):.3f}", ""]

    def lbo(sub, yfun, seg_fn, name):
        its, Xs = [], []
        for it in sub:
            X = seg_fn(it)
            if len(X):
                its.append(it)
                Xs.append(X)
        y = np.array([yfun(it) for it in its])
        if len(set(y)) < 2 or len(set(it["baby"] for it in its)) < 3:
            L.append(f"| {name} | {len(its)} | - | - |")
            return
        P = leave_baby_out(its, y, Xs, 2)[:, 1]
        L.append(f"| {name} | {len(its)} ({int(y.sum())}:{int((1 - y).sum())}) | {roc_auc_score(y, P):.3f} | "
                 f"{balanced_accuracy_score(y, (P >= 0.5).astype(int)):.3f} |")

    def background(it):
        m = it["cry"] < 0.05
        return it["emb"][m].mean(axis=0, keepdims=True) if m.sum() >= 1 else np.zeros((0, 1024))

    cry_items = [it for it in items if it["cry_label"] == "Cry" and it["pain"] != "No Pain"]
    severe = lambda it: int(it["pain"] == "Severe Pain")
    L += ["## 1. 울고 있는 클립 중 심한 통증 vs 중간 통증 (한 아기씩 빼고 학습)", "",
          "| 입력 | 클립 (심함:중간) | ROC-AUC | 균형 정확도 |", "|---|---:|---:|---:|"]
    lbo(cry_items, severe, lambda it: seg_of(it)[0], "울음 세그먼트 (YAMNet 임베딩)")
    lbo(cry_items, severe, background, "배경만 (울음 점수 0.05 미만)")

    # 단순 기준선: 음량·울음 점수만으로
    def simple_auc(feat):
        y = np.array([severe(it) for it in cry_items])
        v = np.array([feat(it) for it in cry_items])
        return roc_auc_score(y, v)

    def rms(it):
        x, _ = sf_.read(it["wav"], dtype="float32")
        return float(np.sqrt(np.mean(x ** 2)) + 1e-9)

    def cry_ratio(it):
        return float((it["cry"] >= 0.3).mean()) if len(it["cry"]) else 0.0

    L += ["", "학습 없는 단순 지표 (심함일수록 클 것으로 가정):", "",
          f"- 음량(RMS) AUC {simple_auc(rms):.3f}",
          f"- 울음 비율(울음 점수 0.3 이상 프레임 비율) AUC {simple_auc(cry_ratio):.3f}",
          f"- 최대 울음 점수 AUC {simple_auc(lambda it: float(it['cry'].max()) if len(it['cry']) else 0.0):.3f}", ""]

    pain_cry = [it for it in items if it["cry_label"] == "Cry"]
    L += ["## 2. 울고 있는 클립 중 통증 있음 vs 없음 (표본 불균형 주의)", "",
          "| 입력 | 클립 (통증:없음) | ROC-AUC | 균형 정확도 |", "|---|---:|---:|---:|"]
    lbo(pain_cry, lambda it: int(it["pain"] != "No Pain"), lambda it: seg_of(it)[0], "울음 세그먼트")
    L += ["", "## 한계", "",
          "- 통증 라벨은 FLACC 행동 점수라 '울음' 항목이 점수에 들어간다. 그래서 심함/중간 구분은 사실상 울음 격렬도 추정이다.",
          "- 라벨은 253개 구간 기준이지만 공개 영상은 (아기, 시술 전/후)마다 1개씩 47개뿐이다.",
          "  나머지 구간은 원저자(Yosi Kristian 외)에게 요청해야 한다. 녹음 장소·장비 정보가 없다."]
    _save_report("IFPALVD_REPORT.md", L)


def _save_report(name, L):
    txt = "\n".join(L) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["enes", "pain", "extra", "online", "online_audio", "baidu", "ifpalvd"])
    args = ap.parse_args()
    {"enes": run_enes, "pain": run_pain, "extra": run_extra, "online": run_online,
     "online_audio": run_online_audio, "baidu": run_baidu, "ifpalvd": run_ifpalvd}[args.what]()
