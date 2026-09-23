"""학습된 울음 이유 분류기 추론 + 마이크 캡처.

- CryModel: 울음 감지(cry_score) + 이유 3클래스(reason, conf)
    reason: hungry / sleepy / discomfort
            none      — 울음 점수가 낮아 이유를 따지지 않음
            uncertain — 울음은 맞지만 최고 확률 < conf_min (판정 보류)
- Mic:      sounddevice 링버퍼로 최근 N초 파형 반환

학습과 같은 단위로 판정한다: 파형 → YAMNet 프레임 → 연속 SEG_FRAMES 평균 세그먼트
→ 헤드 확률 → 세그먼트 평균. 1.5초 창이면 세그먼트 1개다.
"""
import json
import os
import threading

import numpy as np

import config
import features

SR = config.SR


class CryModel:
    def __init__(self, head, labels, meta=None):
        self.head = head
        self.labels = labels
        meta = meta or {}
        self.seg_frames = meta.get("seg_frames", config.SEG_FRAMES)
        self.cry_min = meta.get("cry_min", config.CRY_MIN)
        self.conf_min = meta.get("conf_min", config.CONF_MIN)

    @classmethod
    def load(cls, art_dir=config.ARTIFACTS):
        import tensorflow as tf
        head = tf.keras.models.load_model(os.path.join(art_dir, "cry_head.keras"))
        labels = json.load(open(os.path.join(art_dir, "labels.json"), encoding="utf-8"))
        meta_path = os.path.join(art_dir, "meta.json")
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else None
        return cls(head, labels, meta)

    def probs(self, wav16k):
        """→ (cry_score, 클래스 확률[n_cls] 또는 None(울음 세그먼트 없음))"""
        cry, emb = features.frames(wav16k)
        cry_score = float(cry.mean()) if len(cry) else 0.0
        X, _ = features.segments(cry, emb, seg=self.seg_frames, cry_min=self.cry_min)
        if not len(X):
            return cry_score, None
        return cry_score, self.head(X).numpy().mean(axis=0)

    def predict(self, wav16k):
        """→ (cry_score, reason, confidence)"""
        cry_score, p = self.probs(wav16k)
        if p is None:
            return cry_score, "none", 0.0
        i = int(p.argmax())
        conf = float(p[i])
        return cry_score, (self.labels[i] if conf >= self.conf_min else "uncertain"), conf

    def predict_file(self, path):
        """wav 파일 분류 (마이크 없이 데모·테스트용)."""
        return self.predict(features.load_wav(path))


class Mic:
    """백그라운드 InputStream으로 최근 N초를 유지하는 링버퍼."""

    def __init__(self, sr=SR, seconds=3.0):
        self.sr = sr
        self.buf = np.zeros(int(sr * seconds), dtype=np.float32)
        self.lock = threading.Lock()
        self._stream = None

    def _cb(self, indata, frames, time_info, status):
        x = indata[:, 0]
        with self.lock:
            n = len(x)
            self.buf = np.roll(self.buf, -n)
            self.buf[-n:] = x

    def start(self):
        import sounddevice as sd
        # 기본(default) 입력장치 인덱스(-1)는 USB 마이크를 여러 번 뽑았다 꽂으면
        # 재열거되면서 불안정해질 수 있다(실측: query_devices에서 0 in으로 보임,
        # PortAudioError: Error querying device -1). 이름으로 직접 찾아 명시 지정한다.
        name_hint = os.getenv("NUNI_MIC_NAME", "AB13X")
        device = None
        for i, d in enumerate(sd.query_devices()):
            if name_hint.lower() in d["name"].lower():
                device = i
                break
        self._stream = sd.InputStream(device=device, channels=1, samplerate=self.sr,
                                      callback=self._cb)
        self._stream.start()

    def get_last(self, seconds=1.5):
        with self.lock:
            return self.buf[-int(self.sr * seconds):].copy()


if __name__ == "__main__":
    import sys
    import time
    model = CryModel.load()
    if len(sys.argv) > 1:                       # python infer.py a.wav b.wav ...
        for p in sys.argv[1:]:
            s, reason, conf = model.predict_file(p)
            print(f"{p}: cry_score={s:.2f} reason={reason}({conf:.2f})")
        sys.exit(0)
    m = Mic()
    m.start()
    print("마이크 추론 시작 (Ctrl+C 종료)")
    while True:
        cry_score, reason, conf = model.predict(m.get_last(1.5))
        print(f"cry_score={cry_score:.2f} reason={reason}({conf:.2f})")
        time.sleep(1)
