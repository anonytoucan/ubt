#!/bin/bash
# Local LLM judge (Qwen2.5-72B-Instruct-AWQ) for the real-corpora table: the final model (code inferred and given), the
# two adaptation runs, LibLouis backward and the sealed NIKL test (outputs kept in its own directory).
# Two tensor-parallel-2 workers start once the GPUs are free; a rerun skips existing metric files.
#   nohup scripts/eval/real_judge_chain.sh > $UBT_WORK/evals/real_judge_chain.log 2>&1 &
# Intervals: scripts/paper/table4_ci.py.
set -uo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
source scripts/eval/env_vllm_sm120.sh
export TOKENIZERS_PARALLELISM=false
PY=${PY:-python}
R=$UBT_WORK
X=$R/evals/rlvr/extra
log() { echo "[$(date -u '+%m-%d %T')] $*"; }
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)" -eq 0 ]; }
judge() {   # judge <gpus> <rows> <eval> <out>
  [ -s "$4" ] && return 0
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/eval/u1_extra_metrics.py --rows "$2" --eval "$3" --metrics judge --judge-tp 2 \
      --out "$4" > "${4%.json}.log" 2>&1 && log "judge $4" || log "FAILED: judge $4"
}
until gpus_free; do sleep 30; done
log "GPUs free"
( judge 0,1 $R/nikl_test/run/rows_sim_inferred.jsonl $R/nikl_test/run/real.jsonl $R/nikl_test/run/judge_inferred.json ) &
( judge 2,3 $X/rows_rlvr_real_inferred.jsonl $R/evals/rlvr/real/real.jsonl $X/rlvr_real_inferred.judge.json
  judge 2,3 $X/rows_rlvr_real_given.jsonl $R/evals/rlvr/real/real.jsonl $X/rlvr_real_given.judge.json
  for d in $R/evals/rlvr_bllm_adapt/real $R/evals/rlvr_vision_adapt/real $R/evals/liblouis/real $R/nikl_test/liblouis; do
    judge 2,3 $d/rows_sim_given.jsonl $d/real.jsonl $d/judge_given.json
  done ) &
wait
log "done"
