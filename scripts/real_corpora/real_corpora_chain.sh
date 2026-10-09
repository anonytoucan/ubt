#!/bin/bash
# Best-of-20 pools of the final SFT checkpoint on the real corpora (Bible, BrailleLLM, Vision-Braille; NIKL stays
# sealed), code inferred and code given (gold <code> prefilled).
#   bash scripts/real_corpora/real_corpora_chain.sh   -> $UBT_WORK/real_corpora/run/gen_{inferred,given}.<i>.jsonl
set -e
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
source scripts/eval/env_vllm_sm120.sh
M=$UBT_WORK/models/sft
D=$UBT_WORK/real_corpora
W=$D/run
mkdir -p $W
cat $D/bible.jsonl $D/braillellm.jsonl $D/vision.jsonl > $W/real.jsonl
for cond in inferred given; do
  extra=""; [ "$cond" = given ] && extra="--prefill-code"
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/eval/bon_gen.py gen --model $M --slice $W/real.jsonl \
      --out $W/gen_$cond.$i.jsonl --shard $i/4 $extra > $W/gen_$cond.$i.log 2>&1 &
  done
  wait
  echo "$cond done $(date -u +%T)"
done
echo DONE
