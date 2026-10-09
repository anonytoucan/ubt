#!/bin/bash
# Braille edition of the paper with the final model (run scripts/braille_edition/braille_edition.py build first): best-of-50 with the
# code given, unverified sentences again at n=100, scoring, then the TikZ pages. Waits for free GPUs.
#   nohup scripts/braille_edition/braille_edition_chain.sh > $UBT_WORK/braille_edition/rlvr/chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
export TOKENIZERS_PARALLELISM=false
MODEL=${MODEL:-$UBT_WORK/models/rlvr}
D=${D:-$UBT_WORK/braille_edition/rlvr}
log() { echo "[$(date -u +%T)] $*"; }
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)" -eq 0 ]; }
log "waiting for the NIKL and Vision adaptation chains and free GPUs"
while pgrep -f "nikl_test_chain.sh|vision_adapt_chain.sh" > /dev/null || ! gpus_free; do sleep 60; done
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
log "gen n=50 ($(wc -l < $D/rows.jsonl) rows)"; gen $D/rows.jsonl $D/gen --prefill-code --n 50 || exit 1
log "score"; python scripts/braille_edition/braille_edition.py score --out $D > $D/score.log 2>&1 || { log "FAILED: score"; exit 1; }
if [ -s $D/rerun_rows.jsonl ]; then
  log "gen n=100 on $(wc -l < $D/rerun_rows.jsonl) unverified sentences"
  gen $D/rerun_rows.jsonl $D/rerun --prefill-code --n 100 || exit 1
  python scripts/braille_edition/braille_edition.py score --out $D --rerun-glob "$D/rerun.*.jsonl" > $D/score.log 2>&1 \
      || { log "FAILED: score (rerun)"; exit 1; }
fi
if [ -n "${RERUN_UNITS:-}" ] && [ -s $D/rerun_unit_rows.jsonl ]; then   # optional, off by default
  log "gen n=${RERUN_N:-100} on $(wc -l < $D/rerun_unit_rows.jsonl) unverified units"
  gen $D/rerun_unit_rows.jsonl $D/rerun_units --prefill-code --n ${RERUN_N:-100} || exit 1
  RG=(); ls $D/rerun.*.jsonl >/dev/null 2>&1 && RG=(--rerun-glob "$D/rerun.*.jsonl")   # the glob stays unexpanded
  python scripts/braille_edition/braille_edition.py score --out $D "${RG[@]}" --rerun-units-glob "$D/rerun_units.*.jsonl" > $D/score.log 2>&1 \
      || { log "FAILED: score (unit rerun)"; exit 1; }
fi
log "render"
N=$(python -c "import json; print(json.load(open('$D/edition.cells.json'))['n_pages'])")
python scripts/braille_edition/render_braille_tikz.py --cells $D/edition.cells.json --marks $D/marks.json \
    --text $D/edition.text.txt --out $D/braille_edition_all.tex --pages 1-$N > $D/render.log 2>&1 || log "FAILED: render"
log "done"
