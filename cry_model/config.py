"""공통 설정 (경로·라벨 체계·하이퍼파라미터). 모든 스크립트가 이 값을 공유한다."""
import os

BASE = os.path.dirname(__file__)

SR = 16000
SEED = 42

# --- 경로 ---------------------------------------------------------------
# 원본 데이터는 용량·라이선스 문제로 저장소 밖(예: D:\datasets\cry)에 둔다.
#   DATA_ROOT/
#     donateacry-corpus/          git clone (정제본 + 원본 업로드 버킷)
#     ESC-50-master/              증강용 소음 (crying_baby 제외하고 사용)
#     RIRS_NOISES/                OpenSLR 28 — 실측/시뮬레이션 룸 임펄스 응답
#     extra/<출처명>/<라벨>/*.wav  Kaggle 등 폴더-클래스 구조 추가 데이터
#     _converted/                 .3gp/.caf → 16kHz wav 변환 결과(자동 생성)
DATA_ROOT = os.getenv("NUNI_CRY_DATA", os.path.join(BASE, "data"))
EXTRA_DIR = os.path.join(DATA_ROOT, "extra")
CONVERTED_DIR = os.path.join(DATA_ROOT, "_converted")
MANIFEST = os.path.join(DATA_ROOT, "manifest.csv")
ARTIFACTS = os.path.join(BASE, "artifacts")        # 학습 산출물
CACHE_DIR = os.path.join(DATA_ROOT, ".emb_cache")  # YAMNet 프레임 임베딩 캐시(.npz)
FFMPEG = os.getenv("NUNI_FFMPEG", "ffmpeg")

# --- 라벨 체계: 3클래스 ---------------------------------------------------
# 공개 데이터의 세부 라벨은 표본이 너무 적고(burping 8개 등) 부모 자가 태깅이라
# 잡음이 크다. 시스템 토픽(audio/cry의 cls)과 같은 3클래스로 묶는다.
#   hungry     : 배고픔
#   sleepy     : 졸림·피곤
#   discomfort : 신체 불편 전반 (복통·트림 필요(가스)·일반 불편·덥거나 추움)
# lonely/scared(정서)와 dk(모름), 웃음·소음 등 비울음은 학습에서 제외한다.
CLASSES = ["discomfort", "hungry", "sleepy"]
LABEL_MAP = {
    # Donate-a-Cry 파일명 코드
    "hu": "hungry", "ti": "sleepy",
    "bp": "discomfort", "bu": "discomfort", "dc": "discomfort", "ch": "discomfort",
    # 폴더명 (Donate-a-Cry 정제본 · Kaggle 계열)
    "hungry": "hungry", "hunger": "hungry",
    "tired": "sleepy", "tiredness": "sleepy", "sleepy": "sleepy",
    "belly_pain": "discomfort", "burping": "discomfort", "discomfort": "discomfort",
    "cold_hot": "discomfort",
}

# --- 세그먼트 (학습·추론 입력 길이 일치) ------------------------------------
# YAMNet은 0.96초 창을 0.48초 간격으로 프레임화한다. 실시간 추론 창 1.5초는
# 프레임 2개가 되므로, 학습도 '연속 2프레임 평균 임베딩' 단위로 한다.
SEG_FRAMES = 2
CRY_MIN = 0.10     # 세그먼트 평균 울음 점수가 이 미만이면 울음 아님 → 학습에서 제외
                   # (0.1: 기존 울음 감지 평가에서 F1 최고였던 임계값. 0.15면 정제본도 52개 탈락)

# --- 학습 ---------------------------------------------------------------
N_FOLDS = 5
MIN_CLIPS = 50     # 사용 가능 클립이 이 수 미만인 클래스는 학습에서 뺀다 (현재 sleepy 27개 → 제외)
N_AUG = 2          # train 클립당 증강 사본 수
CONF_MIN = 0.5     # 3클래스 이상일 때 판정 보류 임계값 (2클래스는 train.py가 0.7 사용, meta.json에 기록)
