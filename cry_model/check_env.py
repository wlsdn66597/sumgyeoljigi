"""학습 전 환경 점검. 실행: python check_env.py"""
import os
import shutil
import sys

import config


def ok(msg): print("  [OK]  " + msg)
def bad(msg): print("  [!!]  " + msg)


def main():
    print("Python:", sys.version.split()[0])

    for mod in ["numpy", "tensorflow", "tensorflow_hub", "librosa",
                "soundfile", "sklearn", "matplotlib", "pandas"]:
        try:
            __import__(mod)
            ok(f"import {mod}")
        except Exception as e:
            bad(f"import {mod} 실패 → pip install {mod}  ({e})")

    if shutil.which(config.FFMPEG) or os.path.isfile(config.FFMPEG):
        ok(f"ffmpeg: {config.FFMPEG}")
    else:
        bad("ffmpeg 없음 → Donate-a-Cry 원본 버킷(.3gp/.caf)을 쓸 수 없음 (NUNI_FFMPEG로 경로 지정)")

    try:
        import sounddevice as sd
        ok(f"sounddevice: 오디오 장치 {len(sd.query_devices())}개 (마이크 실시간 추론용)")
    except Exception as e:
        bad(f"sounddevice 미설치/미검출 (파일 추론만 하면 무시 가능): {e}")

    try:
        import features
        _, classes = features.load_yamnet()
        ok(f"YAMNet 로드 성공 (AudioSet 클래스 {len(classes)}개)")
    except Exception as e:
        bad(f"YAMNet 로드 실패 (인터넷 필요): {e}")

    print(f"\n데이터 위치: {config.DATA_ROOT}")
    for sub, name in [("donateacry-corpus", "Donate-a-Cry"), ("ESC-50-master", "소음(ESC-50)"),
                      ("RIRS_NOISES", "잔향(OpenSLR 28)"), ("extra", "추가 데이터(Kaggle 등)")]:
        d = os.path.join(config.DATA_ROOT, sub)
        (ok if os.path.isdir(d) else bad)(f"{name}: {'있음' if os.path.isdir(d) else '없음'}")
    if os.path.exists(config.MANIFEST):
        import pandas as pd
        df = pd.read_csv(config.MANIFEST)
        ok(f"manifest: {len(df)}개 {dict(df.label.value_counts())}")
    else:
        bad("manifest 없음 → python prepare_data.py")


if __name__ == "__main__":
    main()
