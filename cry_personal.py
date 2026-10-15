"""울음 개인화 힌트 (파이에서 실행, numpy만 사용).

부모가 울음이 끝난 뒤 "무엇으로 그쳤나요?"에 답한 기록으로 그 아기에게 맞춰 배운다.

흐름
  1) 울음이 감지되는 동안 1초마다 YAMNet 임베딩을 모은다(add_window).
  2) 울음이 시작되고 몇 초 뒤 힌트를 낸다(hint) — 부모가 무엇을 해 볼지 정할 때 쓰는 참고.
  3) 울음이 끝나면(end) 부모에게 묻고, 답을 그 울음의 소리 특징과 함께 저장한다(feedback).
힌트 = 맥락(마지막 수유 후 시간·직전 울음 후 시간·시간대) × 이 아기의 이전 답 빈도 × 이전 울음 소리와의 유사도

근거: cry_model/README.md 3~6차, cry_model/sim_personal.py(이 모듈을 Enes 울음으로 시간순 재생)
  - 파이와 같은 조건(울기 시작 10초, TFLite YAMNet)에서 맥락 + 이력(최근 답 가중) 3분류 0.456, 배고픔 AUC 0.62
    (무작위 0.33 / 0.5). 소리 유사도만으로도 0.43(귀무 0.38)이지만, 맥락 + 이력에 곱하면 오르지 않았다(-0.02).
  - 그래서 기본값은 소리를 힌트에 섞지 않고(w_sound=0) 임베딩만 저장한다. 실제 기기 데이터로 다시 검증한 뒤 켠다.
  - 확정이 아닌 '쓸수록 맞춰지는 실험적 힌트'로만 보여주고, 확신이 낮거나 이전 답이 적으면 숨긴다(show=False).

필요 파일(cry_model/artifacts, 재생성 가능)
  yamnet_emb.tflite        export_yamnet_tflite.py — 점수 + 임베딩 출력
  cry_personal_basis.npz   sim_personal.py basis   — 공개 데이터로 만든 표준화·PCA(라벨 안 씀)
  cry_personal_ctx.json    sim_personal.py ctx     — 맥락 로지스틱 회귀 계수
"""
import json
import math
import os
import time

import numpy as np

ACTIONS = ["feed", "diaper", "hold"]            # 학습하는 답: 수유 / 기저귀·자세 / 안아주기
EXTRA = ["sleep", "burp", "self", "unknown"]     # 기록만 하는 답: 재우기 / 트림 / 저절로 / 모름
LABEL_KO = {"feed": "수유", "diaper": "기저귀·자세", "hold": "안아주기", "sleep": "재우기", "burp": "트림",
            "self": "저절로", "unknown": "모름"}
CAP_H = 12.0


def ctx_features(t, last_cry_t, last_feed_t):
    """맥락 특징: 하루 중 시각(sin·cos), 직전 울음 후 시간, 마지막 수유 후 시간(모르면 상한)."""
    lt = time.localtime(t)
    h = lt.tm_hour + lt.tm_min / 60
    since = lambda p: CAP_H if p is None else float(min(max((t - p) / 3600, 0), CAP_H))
    return np.array([math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
                     since(last_cry_t), since(last_feed_t)], float)


class CryPersonal:
    def __init__(self, basis_path, ctx_path=None, state_path=None, k_min=10, temp=3.0, show_min=0.5,
                 hint_seconds=10, half_life=8, w_sound=0.0, proto_from="all"):
        """basis_path: npz 경로 또는 {"mean","std","pca"} 딕셔너리. ctx_path: json 경로 또는 딕셔너리."""
        b = basis_path if isinstance(basis_path, dict) else np.load(basis_path)
        self.mu, self.sd, self.P = b["mean"], b["std"], b["pca"]
        self.ctx = None
        c = ctx_path if isinstance(ctx_path, dict) else (
            json.load(open(ctx_path, encoding="utf-8")) if ctx_path and os.path.exists(ctx_path) else None)
        if c:
            self.ctx = (np.array(c["coef"]), np.array(c["intercept"]), np.array(c["mean"]), np.array(c["std"]))
        self.k_min, self.temp, self.show_min, self.hint_seconds = k_min, temp, show_min, hint_seconds
        # half_life: 이력(이전 답 빈도)에서 최근 답일수록 크게 반영(답 8개마다 절반). 아기가 크면서 우는 이유
        #   비율이 바뀌어서다(재생 시험: 이력만 0.389 → 0.434). 소리 비교에는 쓰지 않는다(예시가 줄어 0.428 → 0.398)
        # w_sound: 소리 유사도를 힌트에 얼마나 반영할지(확률의 거듭제곱). 0이면 계산만 하고 섞지 않는다
        # proto_from: 이전 울음을 '울음 전체' 평균(all)으로 비교. 처음 몇 초만 쓰면 더 나빴다(0.428 → 0.381)
        self.half_life, self.w_sound, self.proto_from = half_life, w_sound, proto_from
        self.state_path = state_path
        self.episodes = []            # {"id", "t0", "t1", "z", "action"}
        self.last_feed = None
        self.cur = None
        if state_path and os.path.exists(state_path):
            self.load()

    # --- 울음 구간 ---------------------------------------------------------------
    def start(self, t):
        self.cur = {"t0": t, "embs": []}

    def add_window(self, emb):
        """울음으로 판정된 1초 창의 YAMNet 임베딩(1024)."""
        if self.cur is not None:
            self.cur["embs"].append(np.asarray(emb, np.float32))

    def end(self, t):
        """울음이 끝났을 때 호출. 저장한 울음 번호를 돌려준다(부모 답을 받을 때 쓴다)."""
        if self.cur is None or not self.cur["embs"]:
            self.cur = None
            return None
        ep = {"id": len(self.episodes), "t0": self.cur["t0"], "t1": t,
              "z": self._project(np.mean(self.cur["embs"], 0)).tolist(),
              "zh": self._project(np.mean(self.cur["embs"][: self.hint_seconds], 0)).tolist(), "action": None}
        self.episodes.append(ep)
        self.cur = None
        self.save()
        return ep["id"]

    def feedback(self, ep_id, action):
        """부모 답 기록. 수유는 마지막 수유 시각으로도 쓴다."""
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
    def hint(self, t=None, embs=None):
        """지금 울음의 힌트. embs를 안 주면 지금까지 모은 창 중 처음 hint_seconds개를 쓴다."""
        t = t if t is not None else (self.cur["t0"] if self.cur else time.time())
        if embs is None:
            embs = self.cur["embs"][: self.hint_seconds] if self.cur else []
        last_cry = self.episodes[-1]["t1"] if self.episodes else None
        p_ctx = self._context(t, last_cry)
        labeled = [e for e in self.episodes if e["action"] in ACTIONS]
        p_prior = self._prior(labeled)
        parts = {"context": p_ctx, "history": p_prior}
        p = p_ctx * p_prior
        if len(labeled) >= self.k_min and len(embs):
            p_proto = self._proto(self._project(np.mean(embs, 0)), labeled)
            parts["sound"] = p_proto
            if self.w_sound:
                p = p * p_proto ** self.w_sound
        p = p / p.sum()
        top = int(np.argmax(p))
        since_feed = None if self.last_feed is None else round((t - self.last_feed) / 3600, 1)
        return {"probs": {a: round(float(v), 3) for a, v in zip(ACTIONS, p)}, "top": ACTIONS[top],
                "top_ko": LABEL_KO[ACTIONS[top]], "confidence": round(float(p[top]), 3),
                "show": bool(len(labeled) >= self.k_min and p[top] >= self.show_min),
                "n_feedback": len(labeled), "hours_since_feed": since_feed,
                "parts": {k: [round(float(x), 3) for x in v] for k, v in parts.items()}}

    # --- 내부 --------------------------------------------------------------------
    def _project(self, emb):
        return ((np.asarray(emb, float) - self.mu) / self.sd) @ self.P

    def _weights(self, n):
        if not self.half_life or n == 0:
            return np.ones(n)
        return 0.5 ** ((n - 1 - np.arange(n)) / self.half_life)

    def _prior(self, labeled):
        w = self._weights(len(labeled))
        cnt = np.bincount([ACTIONS.index(e["action"]) for e in labeled], weights=w, minlength=3).astype(float)
        return (cnt + 1) / (cnt.sum() + 3)

    def _proto(self, z, labeled):
        key = "zh" if self.proto_from == "first" else "z"
        Z = np.array([e.get(key, e["z"]) for e in self.episodes])   # 답이 없는 울음도 '이 아기 평균'에는 쓴다
        mu = Z.mean(0)
        zc = z - mu
        s = np.full(3, -1.0)
        for k, a in enumerate(ACTIONS):
            sel = [i for i, e in enumerate(labeled) if e["action"] == a]
            if sel:
                pc = np.mean([np.array(labeled[i].get(key, labeled[i]["z"])) - mu for i in sel], axis=0)
                s[k] = float(zc @ pc / (np.linalg.norm(zc) * np.linalg.norm(pc) + 1e-9))
        e = np.exp(self.temp * (s - s.max()))
        return e / e.sum()

    def _context(self, t, last_cry):
        if self.ctx is None:
            return np.ones(3) / 3
        W, b, m, s = self.ctx
        x = (ctx_features(t, last_cry, self.last_feed) - m) / s
        z = W @ x + b
        e = np.exp(z - z.max())
        return e / e.sum()

    # --- 저장 --------------------------------------------------------------------
    def save(self):
        if not self.state_path:
            return
        tmp = self.state_path + ".tmp"
        json.dump({"episodes": self.episodes, "last_feed": self.last_feed}, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, self.state_path)

    def load(self):
        d = json.load(open(self.state_path, encoding="utf-8"))
        self.episodes, self.last_feed = d["episodes"], d["last_feed"]
