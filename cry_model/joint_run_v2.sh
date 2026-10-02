#!/bin/bash
# 2차 실험(데이터 확장 + 연구 기반 기법) 전체 실행 (GPU 서버)
#   로컬 joint_build.py 결과를 ~/cry/data/joint2 에 두고:  setsid nohup ./joint_run_v2.sh > joint_run_v2.log 2>&1 < /dev/null &
# 순서: ② 동결 프로브(+① 아기별 정규화) → ④ 울음 추가 사전학습(DAPT) → ⑤ 시간 구조 모델 → ③ 안정화 미세조정 → 보고서
cd ~/cry
PY=.venv/bin/python
export CRY_JOINT=~/cry/data/joint2 CRY_RUNS=~/cry/data/runs2 HF_HUB_OFFLINE=1
run() { for i in 1 2 3; do "$@" && return 0; echo "실패, 재시도 $i: $*"; sleep 60; done; return 1; }
probe_all() {   # 프로브와 반복·귀무 비교(정규화 없음 / 아기별)는 GPU를 적게 써서 뒤에서 병렬로
  local m=$1; mkdir -p $CRY_RUNS/probe/$m
  [ -f $CRY_RUNS/probe/$m/robust_baby.csv ] && return 0     # 이미 끝난 모델은 건너뜀
  run $PY joint_train.py probe --model $m > $CRY_RUNS/probe/$m/probe.log 2>&1
  for n in none baby; do
    run $PY joint_train.py robust --model $m --norm $n --repeats 5 --perms 5 > $CRY_RUNS/probe/$m/robust_$n.log 2>&1
  done
}
# .bin만 있는 모델은 safetensors 사본으로(목록 조회가 필요해 이 단계만 온라인)
HF_HUB_OFFLINE=0 $PY joint_train.py convert --models voc2vec,voc2vec_hubert,w2v2_l,unispeech_l,wavlm_bp,wavlm_l,hubert_l,clap
[ -f $CRY_JOINT/folds.csv ] || $PY joint_train.py folds

# ② 동결 프로브: 발성 전용(voc2vec) + 1차 6종
for m in voc2vec voc2vec_hubert clap wavlm_l wavlm_bp hubert_l whisper_s ast; do
  run $PY joint_train.py extract --model $m     # 추출(GPU 무거움)은 하나씩
  probe_all $m &                                # 프로브는 뒤에서 계속
done

# ④ 울음 추가 사전학습(Enes 라벨 울음 제외) → 같은 프로브
[ -f ~/cry/models/local__voc2vec_dapt/model.safetensors ] || run $PY joint_train.py dapt --model voc2vec --steps 6000
run $PY joint_train.py extract --model voc2vec_dapt
probe_all voc2vec_dapt &

# Bonafos 2025 계열 대형 모델
for m in w2v2_l unispeech_l; do run $PY joint_train.py extract --model $m; probe_all $m & done

# Corvin 통증 대조: 조용한 구간(배경)만으로도 맞히면 녹음 장소 지름길
for m in wavlm_l ast voc2vec; do [ -f $CRY_RUNS/CORVIN_BG_$m.md ] || $PY joint_train.py background --model $m; done

# ⑤ 시간 구조 모델과 ③ 미세조정은 GPU를 나눠 병렬로
(for m in voc2vec voc2vec_dapt wavlm_l; do
  run $PY joint_train.py extract --model $m --windows && run $PY joint_train.py temporal --model $m --epochs 40 --seeds 3 > $CRY_RUNS/temporal_$m.log 2>&1
done) &

# ③ 안정화 미세조정: 실측 잔향·잡음 증강, 층 가중합, 낮은 학습률, 아기(그룹) 구분 방해
for m in voc2vec_dapt voc2vec; do
  run $PY joint_train.py finetune --model $m --protocol cv --cond single,mixed,badv --single-on enes,corvin \
      --aug --pool wsum --lr 1e-5 --epochs 15 --tag v2
  run $PY joint_train.py finetune --model $m --protocol lodo --cond mixed --aug --pool wsum --lr 1e-5 --epochs 15 --tag v2
done
wait
$PY joint_train.py report
echo ALL_DONE
