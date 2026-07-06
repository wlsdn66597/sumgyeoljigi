"""학습 전 환경 점검. 실행: python check_env.py"""
import sys
import os

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

    try:
        import sounddevice as sd
        n = len(sd.query_devices())
        ok(f"sounddevice: 오디오 장치 {n}개 (마이크 실시간 추론용)")
    except Exception as e:
        bad(f"sounddevice 미설치/미검출 (파일 추론만 하면 무시 가능): {e}")

    try:
        import features
        y, classes = features.load_yamnet()
        ok(f"YAMNet 로드 성공 (AudioSet 클래스 {len(classes)}개)")
        assert any("cry" in c.lower() for c in classes)
        ok("'Baby cry' 계열 클래스 확인")
    except Exception as e:
        bad(f"YAMNet 로드 실패 (인터넷 필요): {e}")

    if os.path.isdir(config.DATA_ROOT):
        labels = [d for d in os.listdir(config.DATA_ROOT)
                  if os.path.isdir(os.path.join(config.DATA_ROOT, d))]
        n = sum(len(os.listdir(os.path.join(config.DATA_ROOT, l))) for l in labels)
        ok(f"데이터: {len(labels)}개 클래스 {labels}, 약 {n}개 파일")
    else:
        bad(f"데이터 폴더 없음: {config.DATA_ROOT}  → python download_data.py")

    for d, name in [(config.NOISE_DIR, "소음(ESC-50/MUSAN)"), (config.RIR_DIR, "RIR")]:
        state = "있음" if os.path.isdir(d) and os.listdir(d) else "없음(합성 fallback)"
        print(f"  [--]  증강 {name}: {state}")


if __name__ == "__main__":
    main()
