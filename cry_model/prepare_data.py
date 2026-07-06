"""데이터셋 인덱싱.

Donate-a-Cry Corpus (github.com/gveres/donateacry-corpus) 기준:
  root/
    hungry/   *.wav
    tired/    *.wav
    discomfort/ ...
파일명 앞 토큰(앱 설치 UUID)을 group으로 사용 → 같은 아기가 train/test에
동시에 들어가는 누수를 방지(화자/녹음 단위 held-out).
"""
import os
import glob
import pandas as pd


def build_index(root: str) -> pd.DataFrame:
    rows = []
    for label in sorted(os.listdir(root)):
        d = os.path.join(root, label)
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.wav")):
            name = os.path.basename(p)
            group = name.split("-")[0] if "-" in name else os.path.splitext(name)[0]
            rows.append({"path": p, "label": label, "group": group})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"[prepare_data] '{root}' 아래에서 wav를 찾지 못했습니다.")
    return df


if __name__ == "__main__":
    import sys
    df = build_index(sys.argv[1])
    print(df.groupby("label").size())
    print("총 클립:", len(df), "| 고유 그룹:", df.group.nunique())
