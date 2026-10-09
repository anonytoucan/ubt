#!/bin/bash
# Best-of-20 pools (main, noise, zeroshot) on the final SFT model on 4 GPUs, then CPU scoring with bon_eval.py score.
#   nohup scripts/eval/bon_sft_chain.sh > $UBT_WORK/bon_sft/chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
M=$UBT_WORK/models/sft
D=$UBT_WORK/bon_data
W=$UBT_WORK/bon_sft/run
export TOKENIZERS_PARALLELISM=false
log() { echo "[$(date -u +%T)] $*"; }
source scripts/eval/env_vllm_sm120.sh
mkdir -p $W
for s in main noise zeroshot; do
  log "gen $s (4 GPUs, n=20, tau 0.8)"
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/eval/bon_gen.py gen --model $M --slice $D/$s.jsonl \
        --out $W/gen_$s.$i.jsonl --shard $i/4 > $W/gen_$s.$i.log 2>&1 &
  done
  wait
  [ "$(cat $W/gen_$s.*.jsonl | wc -l)" -eq "$(wc -l < $D/$s.jsonl)" ] || { log "FAILED: gen $s incomplete"; exit 1; }
done
log "score (CPU)"
python scripts/eval/bon_eval.py score --data $D --dir $W > $W/score.log 2>&1 || { log "FAILED: score"; exit 1; }
grep -v -iE "warning|prebuilt" $W/score.log | tail -40
log "done"
