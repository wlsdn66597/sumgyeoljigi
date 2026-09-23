"""데이터 자동 준비 (best-effort). 모두 DATA_ROOT(환경변수 NUNI_CRY_DATA) 아래에 받는다.

  python download_data.py cry     # Donate-a-Cry (정제본 + 원본 업로드 버킷, ~100MB, ODbL)
  python download_data.py noise   # ESC-50 (~600MB, CC BY-NC) — 증강 시 crying_baby는 자동 제외
  python download_data.py rir     # OpenSLR 28 룸 임펄스 응답·배경소음 (~1.3GB, Apache 2.0)
  python download_data.py all

로그인·약관 동의가 필요한 데이터는 직접 받아 extra/<출처명>/ 에 풀면 된다(아래 안내).
"""
import os
import subprocess
import sys
import urllib.request
import zipfile

import config

DONATE_REPO = "https://github.com/gveres/donateacry-corpus.git"
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
RIR_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"


def _fetch_zip(url, marker):
    if os.path.isdir(os.path.join(config.DATA_ROOT, marker)):
        print(f"이미 있음 → {marker}/")
        return
    os.makedirs(config.DATA_ROOT, exist_ok=True)
    zpath = os.path.join(config.DATA_ROOT, os.path.basename(url))
    print(f"내려받는 중: {url}")
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(config.DATA_ROOT)
    os.remove(zpath)
    print(f"완료 → {marker}/")


def get_cry():
    dst = os.path.join(config.DATA_ROOT, "donateacry-corpus")
    if os.path.isdir(dst):
        print("이미 있음 → donateacry-corpus/")
        return
    os.makedirs(config.DATA_ROOT, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", DONATE_REPO, dst], check=True)
    print("완료 → donateacry-corpus/  (python prepare_data.py 로 매니페스트 생성)")


def get_noise():
    _fetch_zip(ESC50_URL, "ESC-50-master")


def get_rir():
    _fetch_zip(RIR_URL, "RIRS_NOISES")


HELP = f"""
[데이터 위치] {config.DATA_ROOT}   (환경변수 NUNI_CRY_DATA로 변경)

[직접 받아야 하는 데이터 — 로그인/약관 동의 필요]
  Kaggle 'Baby Cry Pattern Archive (Cry Sense)'  kaggle.com/datasets/mennaahmed23/baby-cry
  Kaggle 'Infant cry Dataset'                    kaggle.com/datasets/sanmithasadhish/infant-cry-dataset
  → zip을 받아 extra/<출처명>/ 에 풀기 (예: extra/crysense/, extra/infantcry/)
    라벨은 파일의 상위 폴더명으로 읽고(config.LABEL_MAP), Donate-a-Cry 재포장본은
    prepare_data.py가 파일명·해시로, train.py가 임베딩 유사도로 중복을 걸러낸다.

[원본 버킷 변환] .3gp/.caf 는 ffmpeg가 필요하다 (PATH 또는 NUNI_FFMPEG).
"""

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    steps = {"cry": [get_cry], "noise": [get_noise], "rir": [get_rir],
             "all": [get_cry, get_noise, get_rir]}.get(what)
    if not steps:
        print(HELP)
        sys.exit(0)
    for step in steps:
        try:
            step()
        except Exception as e:
            print(f"[자동 다운로드 실패] {step.__name__}: {e}")
    print(HELP)
