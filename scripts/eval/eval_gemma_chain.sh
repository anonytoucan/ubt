#!/bin/bash
# Gemma-2-9B backbone row of Table 1: merge the QLoRA checkpoint, best-of-20 on the core split (20,071 documents, code
# inferred) with vLLM at Gemma-2's 8,192 context, scoring, then sim and judge of the verified and greedy selections.
#   NAME=gemma ADAPTER=$UBT_WORK/models/gemma_adapter TOK=<adapter>/tok \
#     nohup scripts/eval/eval_gemma_chain.sh > $UBT_WORK/evals/<NAME>/chain.log 2>&1 &
# Rerunning the same command resumes at the first missing stage.
set -uo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
: "${NAME:?set NAME}" "${ADAPTER:?set ADAPTER}" "${TOK:?set TOK}"
O=$UBT_WORK/evals/$NAME
M=$UBT_WORK/models/${NAME}_merged
D=$UBT_WORK/bon_data
EV=$UBT_WORK/data/main_eval/eval.jsonl
PY=${PY:-python}
export TOKENIZERS_PARALLELISM=false
source scripts/eval/env_vllm_sm120.sh
log() { echo "[$(date -u '+%m-%d %T')] $*"; }
mkdir -p $O/bon $O/extra
if [ ! -f $M/merge_info.json ]; then
  log "merge $ADAPTER -> $M"
  CUDA_VISIBLE_DEVICES="" $PY scripts/train/gemma_merge.py --adapter "$ADAPTER" --tokenizer "$TOK" --out $M > $O/merge.log 2>&1 \
      || { log "FAILED: merge"; exit 1; }
fi
N=$(wc -l < $D/core.jsonl)
if [ "$(cat $O/bon/gen_core.*.jsonl 2>/dev/null | wc -l)" -ne "$N" ]; then
  log "gen core ($N documents x 20, code inferred)"
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i $PY scripts/eval/bon_gen.py gen --model $M --slice $D/core.jsonl --out $O/bon/gen_core.$i.jsonl \
        --shard $i/4 --max-model-len 8192 > $O/bon/gen_core.$i.log 2>&1 &
  done
  wait
  [ "$(cat $O/bon/gen_core.*.jsonl | wc -l)" -eq "$N" ] || { log "FAILED: gen incomplete"; exit 1; }
fi
log "score"
$PY scripts/eval/bon_eval.py score --data $D --dir $O/bon --slices core > $O/bon/score.log 2>&1 || { log "FAILED: score"; exit 1; }
grep -A3 "^| group" $O/bon/score.log | head -3
log "sim / judge rows"
$PY scripts/eval/extra_rows_build.py --model $NAME --out $O/extra --pools-only > $O/extra/rows.log 2>&1 || { log "FAILED: rows"; exit 1; }
for n in ${NAME}_V_agn ${NAME}_G_agn; do
  [ -s $O/extra/$n.sim.json ] && continue
  CUDA_VISIBLE_DEVICES=0 $PY scripts/eval/u1_extra_metrics.py --rows $O/extra/rows_$n.jsonl --eval $EV --metrics sim \
      --out $O/extra/$n.sim.json > $O/extra/$n.sim.log 2>&1 || log "FAILED: sim $n"
done
log "judge"
k=0
for n in ${NAME}_V_agn ${NAME}_G_agn; do
  gpus=$([ $k -eq 0 ] && echo 0,1 || echo 2,3); k=$((k + 1))
  [ -s $O/extra/$n.judge.json ] && continue
  CUDA_VISIBLE_DEVICES=$gpus $PY scripts/eval/u1_extra_metrics.py --rows $O/extra/rows_$n.jsonl --eval $EV --metrics judge \
      --judge-tp 2 --out $O/extra/$n.judge.json > $O/extra/$n.judge.log 2>&1 || log "FAILED: judge $n" &
done
wait
log "done"
