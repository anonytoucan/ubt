#!/bin/bash
# Convention adaptation on Vision-Braille: a new LoRA on the final model (400 steps, batch 128, lr 5e-5) trained on the
# train split from scripts/real_corpora/vision_adapt_build.py, merged, then read on the unadapted row's test documents (code given
# and inferred) and scored by scripts/real_corpora/real_corpora_score.py. Waits for the NIKL chain and free GPUs.
#   nohup scripts/real_corpora/vision_adapt_chain.sh > $UBT_WORK/ckpt/vision_adapt_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=$UBT_WORK/models/rlvr
export OUT=$UBT_WORK/ckpt/vision_adapt_rlvr MAX_RESTARTS=3 PORT=29519
MERGED=$UBT_WORK/models/rlvr_vision_adapt
E=$UBT_WORK/evals/rlvr_vision_adapt/real
log() { echo "[$(date -u +%T)] $*"; }
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)" -eq 0 ]; }
log "waiting for the NIKL chain and free GPUs"
while pgrep -f nikl_test_chain.sh > /dev/null || ! gpus_free; do sleep 60; done
mkdir -p $OUT $E
if [ ! -d $OUT/final ]; then
  log "train"
  unset NCCL_P2P_DISABLE
  scripts/train/sft_supervise.sh --data $UBT_WORK/data/vision_adapt --model $BASE --tokenizer $BASE \
      --max-steps 400 --lr 5e-5 --max-len 4096 --per-device-batch 2 --grad-accum 16 --save-steps 100 --eval-steps 100 \
      --dev-n 500 --ckpt-threshold 1000 --report-to none > $OUT/supervise.log 2>&1 || { log "FAILED: train"; exit 1; }
fi
if [ ! -s $MERGED/merged_from.json ]; then
  log "merge"
  python scripts/eval/greedy_eval.py merge --base $BASE --ckpt $OUT/final --tokenizer $BASE --out $MERGED \
      > $OUT/merge.log 2>&1 || { log "FAILED: merge"; exit 1; }
fi
source scripts/eval/env_vllm_sm120.sh
python -c "
import json
rows=[l for l in open('$UBT_WORK/evals/rlvr/real/real.jsonl',encoding='utf-8') if json.loads(l)['group']=='vision']
open('$E/real.jsonl','w',encoding='utf-8').writelines(rows); print(len(rows),'vision rows')"
gen() {   # gen <rows file> <output prefix> [extra bon_gen.py flags]
  local rows=$1 pre=$2
  shift 2
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/eval/bon_gen.py gen --model "$MERGED" --slice "$rows" \
        --out "$pre.$i.jsonl" --shard "$i/4" "$@" > "$pre.$i.log" 2>&1 &
  done
  wait
  [ "$(cat "$pre".*.jsonl | wc -l)" -eq "$(wc -l < "$rows")" ] || { log "FAILED: $pre incomplete"; return 1; }
}
log "gen given";    gen $E/real.jsonl $E/gen_given --prefill-code || exit 1
log "gen inferred"; gen $E/real.jsonl $E/gen_inferred || exit 1
log "score"
python scripts/real_corpora/real_corpora_score.py --dir $E > $E/score.log 2>&1 || { log "FAILED: score"; exit 1; }
log "done"
