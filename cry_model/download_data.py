"""데이터 자동 준비 (best-effort).

  python download_data.py cry     # Donate-a-Cry 울음 데이터 -> data/
  python download_data.py noise   # ESC-50 소음 -> noise/ (증강용)

네트워크/환경에 따라 실패할 수 있으며, 그 경우 아래 수동 안내를 따르면 된다.
"""
import os
import sys
import subprocess
import shutil
import zipfile
import urllib.request

import config

DONATE_REPO = "https://github.com/gveres/donateacry-corpus.git"
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"


def get_cry():
    tmp = os.path.join(config.BASE, "_donateacry")
    if not os.path.isdir(tmp):
        print("git clone donateacry-corpus ...")
        subprocess.run(["git", "clone", "--depth", "1", DONATE_REPO, tmp], check=True)
    # cleaned 데이터 폴더 탐색 (라벨별 하위폴더 구조)
    src = None
    for root, dirs, _ in os.walk(tmp):
        if any(d in dirs for d in ["hungry", "tired", "belly_pain", "discomfort", "burping"]):
            src = root
            break
    if not src:
        raise SystemExit("라벨 폴더를 찾지 못했습니다. 저장소 구조를 확인하세요.")
    os.makedirs(config.DATA_ROOT, exist_ok=True)
    for label in os.listdir(src):
        sp = os.path.join(src, label)
        if os.path.isdir(sp):
            shutil.copytree(sp, os.path.join(config.DATA_ROOT, label), dirs_exist_ok=True)
    print("완료 → data/  (python prepare_data.py data 로 확인)")


def get_noise():
    os.makedirs(config.NOISE_DIR, exist_ok=True)
    zpath = os.path.join(config.BASE, "esc50.zip")
    print("ESC-50 내려받는 중 (약 600MB)...")
    urllib.request.urlretrieve(ESC50_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(config.BASE)
    audio = os.path.join(config.BASE, "ESC-50-master", "audio")
    for f in os.listdir(audio):
        shutil.copy(os.path.join(audio, f), config.NOISE_DIR)
    print("완료 → noise/  (증강에 자동 사용됨)")


HELP = """
[수동 안내]
울음 데이터 (Donate-a-Cry):
  1) https://github.com/gveres/donateacry-corpus 다운로드
  2) cleaned 폴더의 라벨별 하위폴더(hungry/ tired/ ...)를 cry_model/data/ 로 복사

증강용 소음 (선택, ESC-50):
  https://github.com/karoldvl/ESC-50 의 audio/*.wav 를 cry_model/noise/ 로 복사

추가 울음 데이터(권장, 표본 보강):
  Kaggle 'Infant Cry Audio Corpus' 등 검색해 같은 라벨 폴더 구조로 합치기
"""

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if what == "cry":
            get_cry()
        elif what == "noise":
            get_noise()
        else:
            print(HELP)
    except Exception as e:
        print(f"[자동 다운로드 실패] {e}")
        print(HELP)
