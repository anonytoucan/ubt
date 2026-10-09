#!/bin/bash
# sim (bge-m3) and judge (Qwen2.5-72B-Instruct-AWQ) for every system in the paper's tables, on the rows of
# scripts/eval/extra_rows_build.py: sim on GPU 0, then judge on two tensor-parallel-2 workers.
#   M=rlvr nohup scripts/eval/extra_metrics_chain.sh > $UBT_WORK/evals/rlvr/extra/chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
source scripts/eval/env_vllm_sm120.sh
export TOKENIZERS_PARALLELISM=false
M=${M:-rlvr}
X=$UBT_WORK/evals/$M/extra
EV=$UBT_WORK/data/main_eval/eval.jsonl
REAL=$UBT_WORK/evals/$M/real/real.jsonl
PY=${PY:-python}
log() { echo "[$(date -u +%T)] $*"; }
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)" -eq 0 ]; }
until gpus_free; do sleep 60; done
log "GPUs free; sim"
for n in ${M}_V_agn ${M}_G_agn ${M}_V_given liblouis gpt55 ${M}_real_given ${M}_real_inferred; do
  CUDA_VISIBLE_DEVICES=0 $PY scripts/eval/u1_extra_metrics.py --rows $X/rows_$n.jsonl --eval $EV --eval $REAL --metrics sim \
      --out $X/$n.sim.json > $X/$n.sim.log 2>&1 || log "FAILED: sim $n"
done
log "judge"
( for n in ${M}_V_agn ${M}_V_given gpt55; do
    CUDA_VISIBLE_DEVICES=0,1 $PY scripts/eval/u1_extra_metrics.py --rows $X/rows_$n.jsonl --eval $EV --metrics judge --judge-tp 2 \
        --out $X/$n.judge.json > $X/$n.judge.log 2>&1 || log "FAILED: judge $n"
  done ) &
( for n in ${M}_G_agn liblouis; do
    CUDA_VISIBLE_DEVICES=2,3 $PY scripts/eval/u1_extra_metrics.py --rows $X/rows_$n.jsonl --eval $EV --metrics judge --judge-tp 2 \
        --out $X/$n.judge.json > $X/$n.judge.log 2>&1 || log "FAILED: judge $n"
  done ) &
wait
log "done"
