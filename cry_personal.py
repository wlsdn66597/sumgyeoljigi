"""울음 개인화 힌트 (파이에서 실행, numpy만 사용).

부모가 울음이 끝난 뒤 "무엇으로 그쳤나요?"에 답한 기록으로 그 아기에게 맞춰 배운다.

흐름
  1) 울음이 감지되는 동안 1초마다 YAMNet 임베딩(과 그 1초 소리)을 모은다(add_window).
  2) 울음이 시작되고 몇 초 뒤 힌트를 낸다(hint) — 부모가 무엇을 해 볼지 정할 때 쓰는 참고. 1·2순위를 같이 낸다.
  3) 울음이 끝나면(end) 부모에게 묻고, 답을 그 울음의 소리 특징과 함께 저장한다(feedback).
힌트 = 맥락(마지막 수유 후 시간·직전 울음 후 시간·시간대) × 이 아기의 이전 답 빈도 × 이전 울음 소리와의 유사도

소리 비교 공간(cry_model/README.md 8~10차, sim_space.py, sim_kclass.py) — blocks로 고른다
  acoustic  YAMNet 임베딩 → 음향 특징 134개(음높이·크기·eGeMAPS·MFCC) 선형 변환(acoustic_path의 coef)
  mfcc      울음 소리에서 직접 계산한 MFCC 20개의 평균·표준편차(40개). librosa.feature.mfcc와 같은 계산을 numpy로 한다
  yamnet16  예전 YAMNet 16차원(basis)
  블록마다 '아기 안 퍼짐'으로 나눠 같은 비중으로 붙인다(acoustic_path의 spread_*). YAMNet은 '무슨 소리인가'를 가르도록
  학습돼 같은 울음끼리의 음높이·세기·음색 차이에 둔하다. 정답 규칙을 여러 특징으로 만든 가상 세계에서 YAMNet 16차원이
  가장 약했다(9차).
자가 검증(gate)
  - 울음이 끝날 때마다, 그 울음의 답을 받기 전에 '소리 포함'과 '소리 없이(맥락 × 이력)' 예측을 저장해 둔다.
  - 답 5개부터 소리 예측을 기록하고, 최근 20개 중 비교가 8개 이상일 때 소리 포함 예측이 2개 이상 더 맞으면 소리를 섞는다.
  - 실제 Enes 재생에서 소리를 항상 섞으면 이력만보다 나빴다(0.433 → 0.350). 아기마다 다를 수 있어 기기가 스스로 재고 켠다.
  - 확정이 아닌 '쓸수록 맞춰지는 실험적 힌트'로만 보여주고, 확신이 낮거나 이전 답이 적으면 숨긴다(show / show2=False).
행동(actions): 기본 6개(수유·기저귀·재우기·안아주기·트림·온도). 행동이 늘면 1순위 정확도는 내려가지만 무작위 대비 이득은
  비슷하고(10차: 3개 0.52 / 무작위 0.33, 6개 0.30 / 0.17), 1·2순위가 더 의미 있어진다. 드문 행동(온도 등)은 소리보다
  센서 맥락(온습도·수면 상태)이 맡아야 한다.

필요 파일(cry_model/artifacts, 재생성 가능)
  yamnet_emb.tflite          export_yamnet_tflite.py — 점수 + 임베딩 출력
  cry_personal_acoustic.npz  sim_space.py export, sim_kclass.py export — 음향 변환, MFCC 멜 필터·표준화, 블록 퍼짐
  cry_personal_basis.npz     sim_personal.py basis   — YAMNet 16차원 표준화·PCA
  cry_personal_ctx.json      sim_personal.py ctx     — 맥락 로지스틱 회귀 계수(수유 / 기저귀 / 안아주기 3개로 학습)
"""
import base64
import json
import math
import os
import time

import numpy as np

ACTIONS3 = ["feed", "diaper", "hold"]                                   # Enes 라벨과 같은 3개(예전 기본)
ACTIONS6 = ["feed", "diaper", "sleep", "hold", "burp", "temp"]           # 실사용 6개
ACTIONS = ACTIONS6                                                      # 기본은 실사용 6개
EXTRA = ["self", "unknown", "other", "sick"]                            # 기록만 하는 답
LABEL_KO = {"feed": "수유", "diaper": "기저귀", "sleep": "재우기", "hold": "안아주기·자세 바꾸기", "burp": "트림·가스 빼기",
            "temp": "온도·환경 조절", "self": "저절로 그침", "unknown": "모름", "other": "기타", "sick": "아파 보임"}
CAP_H = 12.0
N_FFT, HOP_MFCC, N_MFCC = 1024, 256, 20


def ctx_features(t, last_cry_t, last_feed_t):
    """맥락 특징: 하루 중 시각(sin·cos), 직전 울음 후 시간, 마지막 수유 후 시간(모르면 상한)."""
    lt = time.localtime(t)
    h = lt.tm_hour + lt.tm_min / 60
    since = lambda p: CAP_H if p is None else float(min(max((t - p) / 3600, 0), CAP_H))
    return np.array([math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
                     since(last_cry_t), since(last_feed_t)], float)


# --- MFCC (librosa.feature.mfcc(n_mfcc=20, n_fft=1024, hop_length=256)과 같은 값, numpy만) ---------------
def dct_matrix(n_out=N_MFCC, n_in=128):
    k, n = np.arange(n_out)[:, None], np.arange(n_in)[None, :]
    D = np.sqrt(2 / n_in) * np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    D[0] /= np.sqrt(2)
    return D


def mfcc_frames(x, mel_fb, D=None):
    """한 구간(파이는 1초 창)의 MFCC [프레임, 20]과 프레임 크기(RMS dB). mel_fb = librosa.filters.mel(16000, 1024)."""
    D = dct_matrix(N_MFCC, mel_fb.shape[0]) if D is None else D
    x = np.asarray(x, float)
    xp = np.pad(x, N_FFT // 2)
    n = 1 + len(x) // HOP_MFCC
    fr = xp[np.arange(N_FFT)[None, :] + HOP_MFCC * np.arange(n)[:, None]]
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    S = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
    db = 10 * np.log10(np.maximum(S @ mel_fb.T, 1e-10))
    db = np.maximum(db, db.max() - 80)
    return db @ D.T, 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)


def mfcc_stats(mf, db):
    """소리 나는 프레임(울음 전체 최대 -30dB 이내)의 MFCC 평균·표준편차 40개."""
    act = db > db.max() - 30
    mf = mf[act] if act.sum() >= 3 else mf
    return np.r_[mf.mean(0), mf.std(0)]


def _pack(v, dtype=np.float16):
    return base64.b64encode(np.asarray(v, dtype).tobytes()).decode()


def _unpack(s, dtype=np.float16):
    return np.frombuffer(base64.b64decode(s), dtype).astype(float)


class CryPersonal:
    def __init__(self, basis_path, ctx_path=None, state_path=None, k_min=10, temp=3.0, show_min=0.5,
                 hint_seconds=10, half_life=8, w_sound=1.0, proto_from="all", sound_method="proto", k_nn=3, show2_min=0.6,
                 acoustic_path=None, blocks=None, actions=None, gate=True, gate_n=8, gate_window=20, gate_margin=2,
                 track_from=5):
        """basis_path / acoustic_path: npz 경로 또는 딕셔너리. ctx_path: json 경로 또는 딕셔너리.
        blocks: 소리 비교 공간 묶음("acoustic", "mfcc", "yamnet16"). 없으면 쓸 수 있는 것 중 acoustic + mfcc.
        actions: 학습하는 답 목록(ACTIONS3 / ACTIONS6 또는 직접)."""
        self.actions = list(actions or ACTIONS)
        b = basis_path if isinstance(basis_path, dict) else np.load(basis_path)
        self.mu, self.sd, self.P = b["mean"], b["std"], b["pca"]
        a = None
        if acoustic_path is not None and (isinstance(acoustic_path, dict) or os.path.exists(acoustic_path)):
            a = acoustic_path if isinstance(acoustic_path, dict) else dict(np.load(acoustic_path))
        a = a or {}
        self.ac = tuple(np.asarray(a[k], float) for k in ("x_mean", "x_std", "coef", "intercept")) if "coef" in a else None
        self.mel_fb = np.asarray(a["mel_fb"], float) if "mel_fb" in a else None
        self.mfcc_norm = (np.asarray(a["mfcc_mean"], float), np.asarray(a["mfcc_std"], float)) if "mfcc_mean" in a else None
        self.dct = dct_matrix(N_MFCC, self.mel_fb.shape[0]) if self.mel_fb is not None else None
        ok = {"yamnet16": True, "acoustic": self.ac is not None, "mfcc": self.mfcc_norm is not None}
        want = blocks or [k for k in ("acoustic", "mfcc") if ok[k]] or ["yamnet16"]
        self.blocks = [k for k in want if ok[k]] or ["yamnet16"]
        self.w = {k: 1.0 / float(a.get(f"spread_{k}", 1.0)) for k in ("yamnet16", "acoustic", "mfcc")}
        self.space = "+".join(self.blocks)
        self.ctx = None
        c = ctx_path if isinstance(ctx_path, dict) else (
            json.load(open(ctx_path, encoding="utf-8")) if ctx_path and os.path.exists(ctx_path) else None)
        if c:
            self.ctx = (np.array(c["coef"]), np.array(c["intercept"]), np.array(c["mean"]), np.array(c["std"]),
                        list(c.get("classes", ACTIONS3)))
        self.k_min, self.temp, self.show_min, self.hint_seconds = k_min, temp, show_min, hint_seconds
        self.show2_min = show2_min      # 1·2순위를 함께 보여줄 기준(두 확률의 합). 행동이 많으면 1순위 확률이 낮아진다
        # half_life: 이력(이전 답 빈도)에서 최근 답일수록 크게 반영(답 8개마다 절반). 아기가 크면서 우는 이유
        #   비율이 바뀌어서다(재생 시험: 이력만 0.389 → 0.434). 소리 비교에는 쓰지 않는다
        # w_sound: 소리 유사도를 힌트에 얼마나 반영할지(확률의 거듭제곱). 0이면 계산·기록만 한다
        # proto_from: 이전 울음을 '울음 전체' 평균(all)으로 비교. 처음 몇 초만 쓰면 더 나빴다(0.428 → 0.381)
        self.half_life, self.w_sound, self.proto_from = half_life, w_sound, proto_from
        # sound_method: proto(행동별 평균과 코사인) / knn(가장 비슷한 이전 울음 k개의 가중 투표).
        #   유형 크기를 맞춘 가상 세계에서 proto가 knn보다 일관되게 나았다(sim_space.py)
        self.sound_method, self.k_nn = sound_method, k_nn
        # gate: 최근 gate_window개 답에서 소리 포함 예측이 gate_margin개 이상 더 맞아야 켠다(비교 gate_n개 이상).
        #   track_from: 답이 이만큼 쌓이면 소리 예측을 기록하기 시작한다(힌트에는 k_min부터)
        self.gate, self.gate_n, self.gate_window, self.gate_margin = gate, gate_n, gate_window, gate_margin
        self.track_from = track_from
        self.state_path = state_path
        self.episodes = []   # {"id","t0","t1","e","eh","m","mh","z","zh","action","pred_sound","pred_base"}
        self.last_feed = None
        self.cur = None
        if state_path and os.path.exists(state_path):
            self.load()

    # --- 울음 구간 ---------------------------------------------------------------
    def start(self, t):
        self.cur = {"t0": t, "embs": [], "mf": [], "db": []}

    def add_window(self, emb, audio=None):
        """울음으로 판정된 1초 창의 YAMNet 임베딩(1024)과 그 창의 소리(16kHz, MFCC 블록을 쓸 때)."""
        if self.cur is None:
            return
        self.cur["embs"].append(np.asarray(emb, np.float32))
        if audio is not None and self.mel_fb is not None:
            mf, db = mfcc_frames(audio, self.mel_fb, self.dct)
            self.cur["mf"].append(mf.astype(np.float32))
            self.cur["db"].append(db.astype(np.float32))

    def _cur_mfcc(self, n_windows=None):
        if not self.cur or not self.cur["mf"]:
            return None
        k = slice(None) if n_windows is None else slice(0, n_windows)
        return mfcc_stats(np.concatenate(self.cur["mf"][k]), np.concatenate(self.cur["db"][k]))

    def end(self, t, mfcc=None):
        """울음이 끝났을 때 호출. 저장한 울음 번호를 돌려준다(부모 답을 받을 때 쓴다).
        mfcc: (울음 전체, 처음 hint_seconds초) MFCC 통계를 밖에서 줄 때(시험용). 답을 받기 전에 '소리 포함'·'소리 없이'
        예측을 남겨 자가 검증에 쓴다."""
        if self.cur is None or not self.cur["embs"]:
            self.cur = None
            return None
        e = np.mean(self.cur["embs"], 0)
        eh = np.mean(self.cur["embs"][: self.hint_seconds], 0)
        m, mh = mfcc if mfcc is not None else (self._cur_mfcc(), self._cur_mfcc(self.hint_seconds))
        ep = {"id": len(self.episodes), "t0": self.cur["t0"], "t1": t, "e": _pack(e), "eh": _pack(eh),
              "m": None if m is None else _pack(m, np.float32), "mh": None if mh is None else _pack(mh, np.float32),
              "z": np.round(self._project(e, m), 4).tolist(), "zh": np.round(self._project(eh, mh), 4).tolist(), "action": None,
              "pred_sound": None, "pred_base": None}
        labeled = self._labeled()
        last_cry = self.episodes[-1]["t1"] if self.episodes else None
        base = self._context(self.cur["t0"], last_cry) * self._prior(labeled)
        ep["pred_base"] = self.actions[int(np.argmax(base))]
        if len(labeled) >= self.track_from:
            snd = self._sound(np.array(ep["zh"]), labeled) ** max(self.w_sound, 1.0)
            ep["pred_sound"] = self.actions[int(np.argmax(base * snd))]
        self.episodes.append(ep)
        self.cur = None
        self.save()
        return ep["id"]

    def feedback(self, ep_id, action):
        """부모 답 기록. actions에 없는 답(저절로·모름 등)은 기록만 한다. 수유는 마지막 수유 시각으로도 쓴다."""
        ep = self.episodes[ep_id]
        ep["action"] = action
        if action == "feed":
            self.last_feed = max(self.last_feed or 0, ep["t1"])
        self.save()

    def log_feed(self, t):
        """앱에서 수유를 따로 기록했을 때."""
        self.last_feed = max(self.last_feed or 0, t)
        self.save()

    # --- 힌트 --------------------------------------------------------------------
    def hint(self, t=None, embs=None, mfcc=None):
        """지금 울음의 힌트. embs·mfcc를 안 주면 지금까지 모은 창 중 처음 hint_seconds개를 쓴다."""
        t = t if t is not None else (self.cur["t0"] if self.cur else time.time())
        if embs is None:
            embs = self.cur["embs"][: self.hint_seconds] if self.cur else []
            mfcc = self._cur_mfcc(self.hint_seconds)
        last_cry = self.episodes[-1]["t1"] if self.episodes else None
        p_ctx = self._context(t, last_cry)
        labeled = self._labeled()
        p_prior = self._prior(labeled)
        parts = {"context": p_ctx, "history": p_prior}
        p = p_ctx * p_prior
        check = self.check()
        sound_on = False
        if len(labeled) >= self.k_min and len(embs):
            p_sound = self._sound(self._project(np.mean(embs, 0), mfcc), labeled)
            parts["sound"] = p_sound
            sound_on = bool(self.w_sound) and (not self.gate or check["sound_better"])
            if sound_on:
                p = p * p_sound ** self.w_sound
        p = p / p.sum()
        order = np.argsort(-p)
        top, top2 = int(order[0]), [self.actions[i] for i in order[:2]]   # 먼저 해 볼 1·2순위
        since_feed = None if self.last_feed is None else round((t - self.last_feed) / 3600, 1)
        return {"probs": {a: round(float(v), 3) for a, v in zip(self.actions, p)}, "top": self.actions[top],
                "top_ko": LABEL_KO.get(self.actions[top], self.actions[top]), "confidence": round(float(p[top]), 3),
                "top2": top2, "top2_ko": [LABEL_KO.get(a, a) for a in top2], "top2_conf": round(float(p[order[:2]].sum()), 3),
                "show": bool(len(labeled) >= self.k_min and p[top] >= self.show_min),
                "show2": bool(len(labeled) >= self.k_min and p[order[:2]].sum() >= self.show2_min),
                "n_feedback": len(labeled), "hours_since_feed": since_feed, "sound_on": sound_on,
                "recent": check, "parts": {k: [round(float(x), 3) for x in v] for k, v in parts.items()}}

    def check(self):
        """자가 검증: 최근 답 gate_window개에서 미리 낸 예측이 맞은 수(소리 포함 / 소리 없이)."""
        ev = [e for e in self.episodes if e["action"] in self.actions and e.get("pred_sound") is not None]
        ev = ev[-self.gate_window:]
        hit_s = sum(e["pred_sound"] == e["action"] for e in ev)
        hit_b = sum(e["pred_base"] == e["action"] for e in ev)
        return {"n": len(ev), "hit_sound": int(hit_s), "hit_base": int(hit_b),
                "sound_better": bool(len(ev) >= self.gate_n and hit_s >= hit_b + self.gate_margin)}

    # --- 내부 --------------------------------------------------------------------
    def _labeled(self):
        return [e for e in self.episodes if e["action"] in self.actions]

    def _project(self, emb, m=None):
        emb = np.asarray(emb, float)
        out = []
        for k in self.blocks:
            if k == "acoustic":
                xm, xs, W, b = self.ac
                v = ((emb - xm) / xs) @ W.T + b
            elif k == "mfcc":
                mm, ms = self.mfcc_norm
                v = np.zeros(len(mm)) if m is None else (np.asarray(m, float) - mm) / ms
            else:
                v = ((emb - self.mu) / self.sd) @ self.P
            out.append(v * self.w[k])
        return np.concatenate(out)

    def _weights(self, n):
        if not self.half_life or n == 0:
            return np.ones(n)
        return 0.5 ** ((n - 1 - np.arange(n)) / self.half_life)

    def _prior(self, labeled):
        w = self._weights(len(labeled))
        K = len(self.actions)
        cnt = np.bincount([self.actions.index(e["action"]) for e in labeled], weights=w, minlength=K).astype(float)
        return (cnt + 1) / (cnt.sum() + K)

    def _sound(self, z, labeled):
        return self._knn(z, labeled) if self.sound_method == "knn" else self._proto(z, labeled)

    def _proto(self, z, labeled):
        key = "zh" if self.proto_from == "first" else "z"
        Z = np.array([e.get(key, e["z"]) for e in self.episodes])   # 답이 없는 울음도 '이 아기 평균'에는 쓴다
        mu = Z.mean(0)
        zc = z - mu
        s = np.full(len(self.actions), -1.0)
        for k, a in enumerate(self.actions):
            sel = [e for e in labeled if e["action"] == a]
            if sel:
                pc = np.mean([np.array(e.get(key, e["z"])) - mu for e in sel], axis=0)
                s[k] = float(zc @ pc / (np.linalg.norm(zc) * np.linalg.norm(pc) + 1e-9))
        e = np.exp(self.temp * (s - s.max()))
        return e / e.sum()

    def _knn(self, z, labeled):
        key = "zh" if self.proto_from == "first" else "z"
        Z = np.array([e.get(key, e["z"]) for e in self.episodes])
        mu = Z.mean(0)
        P = np.array([e.get(key, e["z"]) for e in labeled]) - mu
        q = z - mu
        sims = P @ q / (np.linalg.norm(P, axis=1) * np.linalg.norm(q) + 1e-9)
        top = np.argsort(-sims)[: self.k_nn]
        y = np.array([self.actions.index(labeled[i]["action"]) for i in top])
        p = np.bincount(y, weights=np.exp(self.temp * sims[top]), minlength=len(self.actions)) + 0.05
        return p / p.sum()

    def _context(self, t, last_cry):
        """맥락 모델은 수유 / 기저귀 / 안아주기 3개로 학습했다. 그 밖의 행동은 세 확률의 평균(중립)을 준다."""
        K = len(self.actions)
        if self.ctx is None:
            return np.ones(K) / K
        W, b, m, s, classes = self.ctx
        x = (ctx_features(t, last_cry, self.last_feed) - m) / s
        z = W @ x + b
        pc = np.exp(z - z.max())
        pc = pc / pc.sum()
        p = np.array([pc[classes.index(a)] if a in classes else pc.mean() for a in self.actions])
        return p / p.sum()

    # --- 저장 --------------------------------------------------------------------
    def save(self):
        if not self.state_path:
            return
        tmp = self.state_path + ".tmp"
        json.dump({"episodes": self.episodes, "last_feed": self.last_feed, "space": self.space,
                   "actions": self.actions}, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, self.state_path)

    def load(self):
        d = json.load(open(self.state_path, encoding="utf-8"))
        self.episodes, self.last_feed = d["episodes"], d["last_feed"]
        if d.get("space", "yamnet16") != self.space:     # 비교 공간이 바뀌면 저장한 원 임베딩·MFCC로 다시 바꾼다
            dim = len(self._project(np.zeros(1024)))
            for e in self.episodes:
                if "e" in e:
                    m = _unpack(e["m"], np.float32) if e.get("m") else None
                    mh = _unpack(e["mh"], np.float32) if e.get("mh") else None
                    e["z"] = self._project(_unpack(e["e"]), m).tolist()
                    e["zh"] = self._project(_unpack(e["eh"]), mh).tolist()
            self.episodes = [e for e in self.episodes if len(e["z"]) == dim]
        for e in self.episodes:                          # 예전 상태 파일: 예측을 번호로 저장했다
            for k in ("pred_sound", "pred_base"):
                if isinstance(e.get(k), int):
                    e[k] = ACTIONS3[e[k]]
