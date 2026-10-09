#!/bin/bash
# BrailleLLM few-shot decoding without training: best-of-20 of the SFT model on the 1,000 test documents with k=3
# in-domain examples prefilled (random or cell-retrieved, from scripts/real_corpora/bllm_adapt_build.py), then scoring.
# Starts once no process holds a GPU.
#   nohup scripts/real_corpora/bllm_fewshot_chain.sh > $UBT_WORK/bllm/chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
M=${MODEL:-$UBT_WORK/models/sft}
D=$UBT_WORK/data/bllm
W=$UBT_WORK/bllm/run
log() { echo "[$(date -u +%T)] $*"; }
mkdir -p $W
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
source scripts/eval/env_vllm_sm120.sh
for mode in rand ret; do
  log "gen few-shot $mode (4 GPUs, n=20, tau 0.8)"
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/eval/bon_gen.py gen --model $M --slice $D/fewshot_$mode.jsonl \
        --out $W/gen_fs_$mode.$i.jsonl --shard $i/4 > $W/gen_fs_$mode.$i.log 2>&1 &
  done
  wait
done
for mode in rand ret; do
  log "score few-shot $mode (CPU)"
  python scripts/real_corpora/bllm_fewshot_score.py --pools $W --mode $mode > $W/score_$mode.log 2>&1 \
      || log "FAILED: score $mode (see $W/score_$mode.log)"
done
log "done"
