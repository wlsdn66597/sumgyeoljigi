"""학습된 울음 분류기 추론 + 마이크 캡처.

- CryModel: 울음 감지(cry_score) + 이유 분류(reason, conf)
- Mic:      sounddevice 링버퍼로 최근 N초 파형 반환
"""
import os
import json
import threading
import numpy as np

import features

SR = 16000


class CryModel:
    def __init__(self, head, labels):
        self.head = head
        self.labels = labels

    @classmethod
    def load(cls, art_dir):
        import tensorflow as tf
        head = tf.keras.models.load_model(os.path.join(art_dir, "cry_head.keras"))
        labels = json.load(open(os.path.join(art_dir, "labels.json"), encoding="utf-8"))
        return cls(head, labels)

    def predict(self, wav16k):
        """→ (cry_score, reason, confidence)"""
        cry_score, emb = features.analyze(wav16k)
        probs = self.head(emb[None, :]).numpy()[0]
        i = int(probs.argmax())
        return cry_score, self.labels[i], float(probs[i])

    def predict_file(self, path):
        """wav 파일 분류 (마이크 없이 데모·테스트용)."""
        import librosa
        w, _ = librosa.load(path, sr=SR, mono=True)
        return self.predict(w)


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
        self._stream = sd.InputStream(channels=1, samplerate=self.sr, callback=self._cb)
        self._stream.start()

    def get_last(self, seconds=1.5):
        with self.lock:
            return self.buf[-int(self.sr * seconds):].copy()


if __name__ == "__main__":
    import time
    m = Mic(); m.start()
    model = CryModel.load("artifacts")
    print("마이크 추론 시작 (Ctrl+C 종료)")
    while True:
        cry_score, reason, conf = model.predict(m.get_last(1.5))
        crying = cry_score > 0.5
        print(f"cry_score={cry_score:.2f} crying={crying} reason={reason}({conf:.2f})")
        time.sleep(1)
