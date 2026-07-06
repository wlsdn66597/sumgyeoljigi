# README Results Section

## 하드웨어 없이 재현 가능한 항목

- Python compile/import smoke test
- Synthetic raw radar-like signal 기반 BPM 추정
- Synthetic apnea/motion smoke test
- Controlled scenario 기반 fusion 평가
- Streamlit/dashboard 관련 import smoke
- 기존 reports의 YAMNet/Donate-a-Cry 울음 분석 결과 요약

## 실행 명령어

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
streamlit run app.py
```

## 정량 결과

`bash run_all.sh` 실행 후 다음 파일을 확인한다.

| 항목 | 산출물 |
|---|---|
| Radar DSP | `results/RADAR_DSP_RESULT.md` |
| Fusion evaluation | `results/FUSION_EVALUATION_RESULT.md` |
| Run summary | `results/RESULTS.md`, `results/RESULTS_FINAL.md` |

## 한계

- Radar DSP는 synthetic radar-like signal 기준이며 실제 60GHz 레이더 하드웨어 검증이 아니다.
- Synthetic apnea/motion smoke test는 실환경 무호흡/움직임 감지 검증이 아니다.
- 실제 영유아 환경에서 검증하지 않았다.
- Donate-a-Cry 5-class reason classifier는 class imbalance로 실사용 성능이 부족하다.
- YAMNet cry detection 결과는 synthetic negative 기준 smoke test이며 실제 생활 소음 성능을 보장하지 않는다.
