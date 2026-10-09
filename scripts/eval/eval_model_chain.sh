#!/bin/bash
# Full evaluation of one checkpoint on every reported set (4 GPUs, then CPU scoring), with the SFT model's settings:
# n=20 (greedy + 19 samples, tau 0.8, top-p 0.95), verified selection lambda=2.
#   bon     main + noise + zeroshot, code inferred                  -> bon_eval.py score
#   prefill core, gold <code> prefilled                             -> bon_eval.py score --slices core
#   forced  mixed + intra + k2p, gold <code> at every segment       -> bon_eval.py score --slices main
#   math    math_eval                                               -> math_baselines.py bon
#   real    Bible + BrailleLLM + Vision-Braille, inferred and given -> real_corpora_score.py
#   kodisc  Korean plain-sentence pairs                             -> kodisc_eval.py score
#   bllm    BrailleLLM few-shot, random and retrieved examples      -> bllm_fewshot_score.py
#   NAME=topup MODEL=$UBT_WORK/models/topup [SKIP_MAIN_FROM=<dir with gen_main.*.jsonl>] \
#     nohup scripts/eval/eval_model_chain.sh > $UBT_WORK/evals/<NAME>/chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
: "${NAME:?set NAME}" "${MODEL:?set MODEL}"
O=$UBT_WORK/evals/$NAME
D=$UBT_WORK/bon_data
export TOKENIZERS_PARALLELISM=false
log() { echo "[$(date -u +%T)] $*"; }
source scripts/eval/env_vllm_sm120.sh
gen() {   # gen <rows file> <output prefix> [extra bon_gen.py flags]
  local rows=$1 pre=$2
  shift 2
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/eval/bon_gen.py gen --model "$MODEL" --slice "$rows" \
        --out "$pre.$i.jsonl" --shard "$i/4" "$@" > "$pre.$i.log" 2>&1 &
  done
  wait
  [ "$(cat "$pre".*.jsonl | wc -l)" -eq "$(wc -l < "$rows")" ] || { log "FAILED: $pre incomplete"; return 1; }
}
mkdir -p $O/bon $O/prefill $O/forced $O/math $O/real $O/kodisc $O/bllm
log "model $MODEL -> $O"
if [ -n "${SKIP_MAIN_FROM:-}" ]; then
  for f in "$SKIP_MAIN_FROM"/gen_main.*.jsonl; do ln -sf "$f" $O/bon/; done
  log "main pool linked from $SKIP_MAIN_FROM"
else
  log "gen main"; gen $D/main.jsonl $O/bon/gen_main || exit 1
fi
log "gen noise";    gen $D/noise.jsonl $O/bon/gen_noise || exit 1
log "gen zeroshot"; gen $D/zeroshot.jsonl $O/bon/gen_zeroshot || exit 1
log "gen prefill (code given, core)"; gen $D/core.jsonl $O/prefill/gen_core --prefill-code || exit 1
log "gen code given on every segment (multi-segment)"; gen $D/multi.jsonl $O/forced/gen_main --force-codes || exit 1
log "gen math";     gen $UBT_WORK/data/math_eval/eval.jsonl $O/math/gen_math || exit 1
log "gen real corpora (inferred, given)"
cp $UBT_WORK/real_corpora/run/real.jsonl $O/real/real.jsonl
gen $O/real/real.jsonl $O/real/gen_inferred || exit 1
gen $O/real/real.jsonl $O/real/gen_given --prefill-code || exit 1
log "gen Korean plain pairs"; gen $UBT_WORK/kodisc_plain/data/kodisc.jsonl $O/kodisc/gen_kodisc || exit 1
log "gen BrailleLLM few-shot"
for m in rand ret; do gen $UBT_WORK/data/bllm/fewshot_$m.jsonl $O/bllm/gen_fs_$m || exit 1; done
log "GPU done; scoring (CPU)"
python scripts/eval/bon_eval.py score --data $D --dir $O/bon > $O/bon/score.log 2>&1 || log "FAILED: bon score"
python scripts/eval/bon_eval.py score --data $D --dir $O/prefill --slices core > $O/prefill/score.log 2>&1 \
    || log "FAILED: prefill score"
python scripts/eval/bon_eval.py score --data $D --dir $O/forced --slices main > $O/forced/score.log 2>&1 \
    || log "FAILED: forced score"
python scripts/eval/math_baselines.py bon --rows $UBT_WORK/data/math_eval/eval.jsonl \
    --gen "$O/math/gen_math.*.jsonl" --out $O/math/bon_score.json > $O/math/score.log 2>&1 || log "FAILED: math"
python scripts/real_corpora/real_corpora_score.py --dir $O/real > $O/real/score.log 2>&1 || log "FAILED: real"
python scripts/eval/kodisc_eval.py score --data $UBT_WORK/kodisc_plain/data --dir $O/kodisc \
    > $O/kodisc/score.log 2>&1 || log "FAILED: kodisc"
for m in rand ret; do
  python scripts/real_corpora/bllm_fewshot_score.py --pools $O/bllm --mode $m > $O/bllm/score_$m.log 2>&1 \
      || log "FAILED: bllm $m"
done
log "done"
