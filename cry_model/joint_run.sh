#!/bin/bash
# 울음 이유 혼합 학습 전체 실행 (GPU 서버). 로컬에서 joint_build.py로 만든 joint/ 폴더를 ~/cry/data/joint 에 둔다.
#   cd ~/cry && setsid nohup ./joint_run.sh > joint_run.log 2>&1 < /dev/null &
# 다른 사람이 GPU를 쓰는 중이면 빌 때까지 기다린다. 이미 끝난 결과는 건너뛴다.
cd ~/cry
PY=.venv/bin/python
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
wait_gpu() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do
    echo "[$(date +%H:%M)] GPU 사용 중 - 대기"; sleep 120
  done
}
run() { for i in 1 2 3; do wait_gpu; "$@" && return 0; echo "실패, 재시도 $i: $*"; sleep 60; done; return 1; }

$PY joint_train.py convert
[ -f data/joint/folds.csv ] || $PY joint_train.py folds
for m in ast wavlm_bp wavlm_l hubert_l whisper_s clap; do      # 1단계: 동결 프로브
  run $PY joint_train.py extract --model $m
  [ -f data/runs/probe/$m/preds.csv ] || run $PY joint_train.py probe --model $m
  [ -f data/runs/probe/$m/robust.csv ] || run $PY joint_train.py robust --model $m --repeats 5 --perms 5
done
for m in ast wavlm_bp; do                                        # 2단계: 전체 미세조정
  run $PY joint_train.py finetune --model $m --protocol cv --cond single,mixed,dann
  run $PY joint_train.py finetune --model $m --protocol lodo --cond mixed,dann
done
$PY joint_train.py report
echo ALL_DONE
