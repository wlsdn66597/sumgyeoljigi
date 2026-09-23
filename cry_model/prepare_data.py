"""여러 출처의 울음 데이터를 하나의 매니페스트(manifest.csv)로 통합.

출처
  1) Donate-a-Cry 정제본   donateacry_corpus_cleaned_and_updated_data/<라벨>/*.wav
  2) Donate-a-Cry 원본 버킷 donateacry-{android,ios}-upload-bucket/*.3gp|*.caf
     — 정제본에 없는 녹음만 추가(정제본과 같은 녹음은 건너뜀). ffmpeg로 16kHz wav 변환.
  3) extra/<출처명>/…/<라벨폴더>/*  Kaggle 등 폴더-클래스 구조 데이터

누수 방지
  - group: 같은 아기(앱 설치 UUID)의 녹음은 같은 group → train/test에 동시에 들어가지 않음.
  - 중복 제거: Donate-a-Cry 파일명 키(UUID+타임스탬프)와 파일 해시(md5)로 재포장본을 걸러냄.
    파일명이 바뀐 재인코딩 사본은 train.py에서 임베딩 유사도로 한 번 더 묶는다.

실행: python prepare_data.py            (DATA_ROOT는 config / 환경변수 NUNI_CRY_DATA)
출력: DATA_ROOT/manifest.csv  (path,label,orig_label,source,group,key,md5)
"""
import hashlib
import os
import re
import shutil
import subprocess
from collections import Counter

import pandas as pd

import config

AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".3gp", ".caf", ".webm"}
DAC_RE = re.compile(
    r"^(?P<uuid>[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})"
    r"-(?P<ts>\d+)-[\d.]+-(?P<sex>[mf])-(?P<age>\d+)-(?P<code>[a-z]{2})\.\w+$")
# Donate-a-Cry 파일명 코드 → 라벨 이름 (매니페스트의 orig_label을 폴더명과 같은 체계로)
CODE_NAME = {"hu": "hungry", "ti": "tired", "bp": "belly_pain", "bu": "burping",
             "dc": "discomfort", "ch": "cold_hot", "lo": "lonely", "sc": "scared", "dk": "unknown"}


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def have_ffmpeg():
    return shutil.which(config.FFMPEG) is not None or os.path.isfile(config.FFMPEG)


def to_wav16k(src):
    """wav가 아니면 16kHz 모노 wav로 변환해 경로 반환(이미 변환돼 있으면 재사용). 실패 시 None."""
    if src.lower().endswith(".wav"):
        return src
    os.makedirs(config.CONVERTED_DIR, exist_ok=True)
    tag = hashlib.md5(src.encode()).hexdigest()[:8]
    dst = os.path.join(config.CONVERTED_DIR, f"{os.path.splitext(os.path.basename(src))[0]}_{tag}.wav")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    r = subprocess.run([config.FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", src,
                        "-ac", "1", "-ar", str(config.SR), dst], capture_output=True)
    return dst if r.returncode == 0 and os.path.exists(dst) else None


def _walk_audio(root):
    for d, _, files in os.walk(root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in AUDIO_EXT:
                yield os.path.join(d, f)


class Builder:
    def __init__(self):
        self.rows, self.keys, self.md5s = [], set(), set()
        self.skipped = Counter()

    def add(self, path, orig_label, source, group, key=None):
        label = config.LABEL_MAP.get(orig_label)
        if label is None:
            self.skipped[f"라벨 제외({orig_label})"] += 1
            return
        if key and key in self.keys:
            self.skipped["중복(파일명 키)"] += 1
            return
        digest = md5_file(path)
        if digest in self.md5s:
            self.skipped["중복(파일 해시)"] += 1
            return
        wav = to_wav16k(path)
        if wav is None:
            self.skipped["변환 실패"] += 1
            return
        if key:
            self.keys.add(key)
        self.md5s.add(digest)
        self.rows.append({"path": wav, "label": label, "orig_label": orig_label,
                          "source": source, "group": group, "key": key or "", "md5": digest})


def add_donateacry(b, root):
    cleaned = os.path.join(root, "donateacry_corpus_cleaned_and_updated_data")
    if os.path.isdir(cleaned):
        for p in _walk_audio(cleaned):
            m = DAC_RE.match(os.path.basename(p))
            folder = os.path.basename(os.path.dirname(p)).lower()
            if m:
                b.add(p, folder, "dac_cleaned", "dac:" + m["uuid"].lower(),
                      key="dac:" + m["uuid"].lower() + "-" + m["ts"])
            else:
                b.add(p, folder, "dac_cleaned", "dac_file:" + os.path.basename(p))
    if not have_ffmpeg():
        print("[!] ffmpeg 없음 → 원본 버킷(.3gp/.caf)은 건너뜀 (NUNI_FFMPEG로 경로 지정 가능)")
        return
    for bucket in ("donateacry-android-upload-bucket", "donateacry-ios-upload-bucket"):
        d = os.path.join(root, bucket)
        if not os.path.isdir(d):
            continue
        for p in _walk_audio(d):
            m = DAC_RE.match(os.path.basename(p))
            if not m:
                b.skipped["파일명 해석 불가"] += 1
                continue
            b.add(p, CODE_NAME.get(m["code"], m["code"]), "dac_raw", "dac:" + m["uuid"].lower(),
                  key="dac:" + m["uuid"].lower() + "-" + m["ts"])


def add_extra(b, extra_dir):
    """extra/<출처명>/…/<라벨폴더>/파일. 라벨은 파일의 상위 폴더명."""
    if not os.path.isdir(extra_dir):
        return
    for src in sorted(os.listdir(extra_dir)):
        sroot = os.path.join(extra_dir, src)
        if not os.path.isdir(sroot):
            continue
        for p in _walk_audio(sroot):
            folder = os.path.basename(os.path.dirname(p)).lower().replace(" ", "_")
            m = DAC_RE.match(os.path.basename(p))
            if m:   # Donate-a-Cry 재포장본이면 같은 키·같은 아기 group으로 취급
                b.add(p, folder, src, "dac:" + m["uuid"].lower(),
                      key="dac:" + m["uuid"].lower() + "-" + m["ts"])
            else:   # 출처 불명 파일: 아기 식별 불가 → 파일 단위 group (유사도 병합은 train.py)
                b.add(p, folder, src, f"{src}:{os.path.relpath(p, sroot)}")


def build_manifest(root=None):
    root = root or config.DATA_ROOT
    b = Builder()
    add_donateacry(b, os.path.join(root, "donateacry-corpus"))
    add_extra(b, os.path.join(root, "extra"))
    df = pd.DataFrame(b.rows)
    if df.empty:
        raise SystemExit(f"[prepare_data] '{root}' 아래에서 사용할 오디오를 찾지 못했습니다. "
                         "python download_data.py cry 를 먼저 실행하세요.")
    return df, b.skipped


def summarize(df, skipped):
    print(pd.crosstab(df.source, df.label, margins=True))
    print("\n원 라벨별:", dict(Counter(df.orig_label)))
    print("고유 group(아기/파일):", df.group.nunique())
    if skipped:
        print("제외:", dict(skipped))


if __name__ == "__main__":
    df, skipped = build_manifest()
    df.to_csv(config.MANIFEST, index=False, encoding="utf-8")
    summarize(df, skipped)
    print(f"\n저장 → {config.MANIFEST}")
