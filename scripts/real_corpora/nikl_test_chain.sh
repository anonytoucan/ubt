#!/bin/bash
# The sealed NIKL test, run once with the final model; everything stays under $O, outside git.
# Rows from scripts/real_corpora/nikl_test_build.py; pools as for every real corpus (greedy + 19 samples, code inferred and given),
# scored by scripts/real_corpora/real_corpora_score.py, then bge-m3 sim of the verified outputs.
#   MODEL=$UBT_WORK/models/rlvr nohup scripts/real_corpora/nikl_test_chain.sh > $UBT_WORK/nikl_test/chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
export TOKENIZERS_PARALLELISM=false
source scripts/eval/env_vllm_sm120.sh
MODEL=${MODEL:-$UBT_WORK/models/rlvr}
O=$UBT_WORK/nikl_test/run
log() { echo "[$(date -u +%T)] $*"; }
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
[ -s $O/real.jsonl ] || { log "FAILED: no rows (scripts/real_corpora/nikl_test_build.py)"; exit 1; }
log "model $MODEL -> $O ($(wc -l < $O/real.jsonl) documents)"
done_pool() { [ "$(cat "$1".*.jsonl 2>/dev/null | wc -l)" -eq "$(wc -l < $O/real.jsonl)" ]; }   # rerun: keep finished pools
done_pool $O/gen_inferred || { log "gen inferred"; gen $O/real.jsonl $O/gen_inferred || exit 1; }
done_pool $O/gen_given || { log "gen given"; gen $O/real.jsonl $O/gen_given --prefill-code || exit 1; }
log "score"
python scripts/real_corpora/real_corpora_score.py --dir $O > $O/score.log 2>&1 || { log "FAILED: score"; exit 1; }
log "sim"
python - <<'PY'
import json, os, sys
sys.path.insert(0, "src")
from ubt.wandb_utils import parse_target
O = os.path.join(os.environ["UBT_WORK"], "nikl_test/run")
rows = {}
for l in open(f"{O}/per_doc.jsonl", encoding="utf-8"):
    d = json.loads(l)
    v = d["pol"]["V"]
    hyp = "\n".join(t for _, t in parse_target(v["text"]))
    rows.setdefault(d["cond"], []).append({"id": d["id"], "hyp_text": hyp, "group": "nikl_fc" if d["f_consistent"] else "nikl_nonfc", "set": v["kind"]})
for cond, rs in rows.items():
    with open(f"{O}/rows_sim_{cond}.jsonl", "w", encoding="utf-8") as fh:
        for r in rs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print({c: len(r) for c, r in rows.items()})
PY
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)" -eq 0 ]; }
while pgrep -f vision_adapt_chain.sh > /dev/null || ! gpus_free; do sleep 60; done   # bge-m3 needs a GPU
for cond in given inferred; do
  CUDA_VISIBLE_DEVICES=0 python scripts/eval/u1_extra_metrics.py --rows $O/rows_sim_$cond.jsonl --eval $O/real.jsonl \
      --metrics sim --out $O/sim_$cond.json > $O/sim_$cond.log 2>&1 || log "FAILED: sim $cond"
done
log "done"
