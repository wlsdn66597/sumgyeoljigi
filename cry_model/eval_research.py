"""연구용 울음 데이터로 검증: 원거리 울음 감지, 원인 분류(아기 간·아기별), 통증 vs 불편.

    python eval_research.py enes    # EnesBabyCries: 가정 1~4m 녹음 울음 감지율 + 원인 분류
    python eval_research.py pain    # Corvin: 통증(예방접종) vs 불편(목욕) + 녹음 장소 지름길 점검
    python eval_research.py extra   # Enes 추가 실험: 깨끗한 라벨, 배고픔 vs 나머지, 리듬·맥락 특징
    python eval_research.py online  # 실제 사용 모사: 아기별 원인 이력이 시간순으로 쌓일 때 맥락 + 이력 결합

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["enes", "pain", "extra", "online"])
    args = ap.parse_args()
    {"enes": run_enes, "pain": run_pain, "extra": run_extra, "online": run_online}[args.what]()
