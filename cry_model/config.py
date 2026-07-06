"""공통 설정 (경로·하이퍼파라미터). 모든 스크립트가 이 값을 공유한다."""
import os

BASE = os.path.dirname(__file__)

SR = 16000
DATA_ROOT = os.path.join(BASE, "data")        # 라벨별 폴더 (hungry/, tired/ ...)
ARTIFACTS = os.path.join(BASE, "artifacts")   # 학습 산출물
CACHE_DIR = os.path.join(BASE, ".emb_cache")  # YAMNet 임베딩 캐시(.npy)
NOISE_DIR = os.path.join(BASE, "noise")       # (선택) ESC-50/MUSAN 소음 wav
RIR_DIR = os.path.join(BASE, "rir")           # (선택) 실측 룸 임펄스 응답 wav

N_AUG = 2          # train 클립당 증강 사본 수
TEST_SIZE = 0.2    # held-out 비율
SEED = 42
CRY_THRESHOLD = 0.5
