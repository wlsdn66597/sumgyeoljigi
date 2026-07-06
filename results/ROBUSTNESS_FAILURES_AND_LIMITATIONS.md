# Robustness and Insights Failures, Skips, and Limitations

## 1. 실패한 명령
| 명령 | 실패 여부 | 원인 | 영향 | 대응 |
|---|---|---|---|---|
| `pip install -r requirements.txt` | 실패 아님 | `/home/user/.cache/pip` 쓰기 권한 경고 발생. 패키지는 이미 충족됨 | 실행에는 영향 없음 | 로그에 기록. sudo 사용하지 않음 |
| `python -m compileall .` | 실패 아님 | 없음 | 없음 | 성공으로 기록 |
| `bash run_all.sh` | 실패 아님 | 없음 | 없음 | 성공으로 기록 |
| `python eval_fusion_noisy.py` | 실패 아님 | 없음 | 없음 | 성공으로 기록 |
| `python eval_radar_realistic.py` | 실패 아님 | 없음 | 없음 | 성공으로 기록 |
| `python eval_cry_context.py` | 실패 아님 | 없음 | 없음 | 성공으로 기록 |
| `python eval_sleep_rhythm.py` | 실패 아님 | 없음 | 없음 | 성공으로 기록 |
| `python cloud.py` | 실패 아님 | Matplotlib 기본 설정 디렉터리 `/home/user/.config/matplotlib` 쓰기 불가 경고. `/tmp` 임시 캐시 사용 | 리포트/PNG 생성에는 영향 없음 | 로그에 기록. sudo나 홈 디렉터리 수정 없이 진행 |

## 2. 스킵한 항목
| 항목 | 이유 | 영향 | 후속 작업 |
|---|---|---|---|
| git commit / git push | 사용자 지시: 이번 작업에서는 commit/push 금지 | 변경사항은 로컬 작업트리에만 남음 | 사용자가 직접 stage/commit/push |
| 실제 Streamlit 대시보드 시동/스크린샷 | 이번 요청 범위가 강건성/인사이트 검증 md 정리이며, `run_all.sh`는 import smoke만 수행 | UI 표시 검증은 포함되지 않음 | 필요 시 별도 수동 캡처 |
| 실제 60GHz 레이더 연결 | 하드웨어 없는 synthetic/controlled 검증 범위 | 실물 센서 성능 주장 불가 | 2차 데모에서 실물 통합 후 재검증 |
| 실제 영유아 장기 데이터 검증 | 데이터/IRB/현장 수집 범위 밖 | 실사용/임상 성능 주장 불가 | 실제 데이터 수집 및 윤리 절차 후 검증 |
| 오래 걸리는 ML 학습 | 사용자 지시상 대용량/장시간 ML 학습 자동 실행 금지, 이번 변경의 핵심 평가가 아님 | 울음 이유 분류 성능 재학습 결과 없음 | 별도 승인 후 분리 실행 |

## 3. 문서 기대값과 실제 실행값 차이
| 항목 | 문서 기대값 | 실제 실행값 | 판단 |
|---|---:|---:|---|
| Monte Carlo 반복 횟수 | 2000 | 2000 | 일치 |
| 정상/위험 trial 수 | 문서에 고정값 없음 | 정상 1402, 실제 위험 598 | 실제 실행값 기록 |
| 단일 센서 오탐률 | 약 57.6% | 57.6% (807건) | 일치 |
| 융합 오탐률 | 약 2.7% | 2.7% (38건) | 일치 |
| 미탐 | 0 | single_sensor 0, full_fusion 0 | 일치 |
| naive radar MAE | 15.0회/분 | 15.0회/분 | 일치 |
| realistic radar MAE | 0.5회/분 | 0.5회/분 | 일치 |
| Cry-context 습도 패턴 | 울음 시 평균 습도 저하 | 울음 시 29.5%, 평상시 37.0%, 차이 -7.5 | 일치 |
| Cry-context 최다 울음 시간대 | 자주 우는 시간대 검출 | 6시간대 22회 | 실제 실행값 기록 |
| Sleep rhythm 평균 각성 | N회 형태 기대 | 14.0회 | 실제 실행값 기록 |
| Sleep rhythm 최다 각성 시간대 | X시경 형태 기대 | 3시경, 30회 | 실제 실행값 기록 |
| `results/RESULTS.md` noisy fusion 요약 | 상세값 기반 요약 | 58% -> 3% | 상세 57.6% -> 2.7%의 반올림. 충돌 아님 |
| Cloud report 울음-맥락 | 울음-맥락 섹션 포함, 습도 field 반영 | 섹션 포함. 이번 cloud synthetic report에는 시간대 인사이트 출력 | 코드상 humidity는 record/analyze에 반영. 출력 인사이트는 threshold 조건에 따라 달라짐 |
| `run_all.sh` 단계 표기 | 16단계 기대 | 실행 블록 16개, 앞 4개 echo는 `[1/12]`~`[4/12]` | 실행은 성공. 라벨 표기 수정 권장 |

## 4. 현재 검증의 한계
- synthetic signal 기준이다.
- controlled scenario 기준이다.
- 실제 60GHz 레이더 미검증이다.
- 실제 영유아 데이터 미검증이다.
- 실제 생활 소음/환경 인과 검증이 아니다.
- 의료/진단 목적 검증이 아니다.
- 울음 이유 분류 실사용 성능을 주장할 수 없다.
- 울음-맥락 결과는 상관/인사이트이지 원인 진단이 아니다.
- 수면 리듬 결과는 synthetic 5박 sequence 기반이며 실제 장기 코칭 효과 검증이 아니다.
- 환경/클라우드 리포트는 소프트웨어 흐름 검증이며 실제 IoT 자동제어 검증이 아니다.

## 5. 보고서/README에서 피해야 할 표현
- 실제 영유아 대상 성능 검증 완료
- 실제 60GHz 레이더 성능 입증
- 실제 생활 환경에서 융합 오탐률 입증
- 의료기기 수준 무호흡 탐지
- 울음 원인 진단
- 습도 저하가 울음의 원인이라고 단정
- 실사용 가능한 수면 리듬 코칭 검증 완료
- 실제 IoT 자동제어 완료
- 임상적으로 안전성이 입증된 무호흡 감지
