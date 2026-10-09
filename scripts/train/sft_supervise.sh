#!/bin/bash
# Run train_sft_u1.py to completion, resuming from the newest checkpoint after a crash such as a rare OOM.
# Gives up after MAX_RESTARTS consecutive failures without progress.
#   OUT=<run dir> WANDB_RUN_ID=<id> nohup scripts/train/sft_supervise.sh <train_sft_u1.py args without --out/--resume> &
set -uo pipefail
: "${OUT:?set OUT}"
MAX_RESTARTS=${MAX_RESTARTS:-4}
PORT=${PORT:-29516}
cd "$(dirname "$0")/../.."
fails=0
while :; do
  ck=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*-//' | sort -n | tail -1)
  res=(); [ -n "$ck" ] && res=(--resume "$OUT/checkpoint-$ck")
  echo "[supervise] $(date -u +%F' '%T) launch (resume: ${ck:-none})"
  torchrun --nproc_per_node 4 --master_port "$PORT" scripts/train/train_sft_u1.py --out "$OUT" "${res[@]}" "$@" \
      >> "$OUT/train.log" 2>&1 < /dev/null
  rc=$?
  if [ -d "$OUT/final" ] && [ $rc -eq 0 ]; then echo "[supervise] $(date -u +%T) finished"; exit 0; fi
  new_ck=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*-//' | sort -n | tail -1)
  if [ "$new_ck" = "$ck" ]; then fails=$((fails + 1)); else fails=0; fi
  echo "[supervise] $(date -u +%T) exit $rc at checkpoint ${new_ck:-none} (failures without progress: $fails)"
  tail -n 300 "$OUT/train.log" | grep -E "OutOfMemoryError|CUDA out of memory|Error" | tail -3
  [ $fails -ge "$MAX_RESTARTS" ] && { echo "[supervise] giving up"; exit 1; }
  sleep 60
done
