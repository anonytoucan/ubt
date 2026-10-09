#!/bin/bash
# RLVR stage from the top-up model. Waits for the top-up evaluation chain to leave the GPUs, then
#   1 mining  top-up model, G=4 at T=1.0 on the prompt pool (4 GPUs) -> scripts/train/rlvr_mine.py -> pool_mined
#   2 RL      trl vllm-serve on GPU 0 + GRPO/DAPO on GPUs 1-3, fresh LoRA r64, 400 steps (3 attempts, resuming)
#   3 curve   dev greedy of the top-up model and every checkpoint (report only, not used for selection)
#   4 merge   step-400 LoRA -> $UBT_WORK/models/rlvr
#   5 eval    scripts/eval/eval_model_chain.sh NAME=rlvr (same settings as SFT and top-up)
#   WAIT_LOG=$UBT_WORK/evals/topup/chain.log nohup scripts/train/rlvr_chain.sh \
#       > $UBT_WORK/rlvr_run/chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
source scripts/eval/env_vllm_sm120.sh
export TOKENIZERS_PARALLELISM=false RLVR_DATA=$UBT_DATASETS/data_u1_v2
M=$UBT_WORK/models/topup
D=$UBT_WORK/data/rlvr
O=$UBT_WORK/rlvr_run
PY=${PY:-python}
PORT=8011
log() { echo "[$(date -u +%T)] $*"; }
tree_pids() { echo "$1"; for c in $(pgrep -P "$1"); do tree_pids "$c"; done; }
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)" -eq 0 ]; }
mkdir -p $O/mine $O/curve

if [ -n "${WAIT_LOG:-}" ]; then
  log "waiting for 'GPU done' in $WAIT_LOG"
  until grep -qE "GPU done|FAILED" "$WAIT_LOG" 2>/dev/null; do sleep 30; done
fi
until gpus_free; do sleep 30; done
log "GPUs free"

# 1 difficulty mining
if [ ! -s $D/pool_mined.jsonl ]; then
  log "mining: top-up model, G=4, T=1.0, $(wc -l < $D/pool.jsonl) prompts"
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i $PY scripts/train/rlvr_sample.py --model $M --prompts $D/pool.jsonl \
        --out $O/mine/samples.$i.jsonl --n 4 --temperature 1.0 --top-p 1.0 --max-tokens 1024 --shard $i/4 \
        --seed 145 > $O/mine/samples.$i.log 2>&1 &
  done
  wait
  $PY scripts/train/rlvr_mine.py --pool $D/pool.jsonl --samples "$O/mine/samples.*.jsonl" --out $D/pool_mined.jsonl \
      > $O/mine/mine.log 2>&1 || { log "FAILED: mining (see $O/mine)"; exit 1; }
fi
log "mined pool: $(wc -l < $D/pool_mined.jsonl) prompts"

# 2 RL
SP=
serve() {
  CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model $M --max-model-len 3072 --gpu-memory-utilization 0.85 \
      --port $PORT > $O/vllm_serve.$1.log 2>&1 &
  SP=$!
  for i in $(seq 1 240); do
    # TRL >= 1.13 serves vLLM's /health route; /health/ answers 307
    curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200 \
        && { log "vLLM server up (pid $SP, ~$((i * 5))s)"; return 0; }
    kill -0 $SP 2>/dev/null || { log "FAILED: vLLM server exited (see $O/vllm_serve.$1.log)"; return 1; }
    sleep 5
  done
  log "FAILED: vLLM server not up in 20 min"; stop_server; return 1
}
stop_server() {
  local pids
  pids=$(tree_pids "$SP")
  kill $pids 2>/dev/null; sleep 10; kill -9 $pids 2>/dev/null
  for _ in $(seq 1 60); do gpus_free && return 0; sleep 5; done
  log "WARNING: GPU memory still in use 5 min after the run"
}
for attempt in 1 2 3; do
  [ -d $O/run/final ] && break
  serve $attempt || exit 1
  log "train attempt $attempt"
  CUDA_VISIBLE_DEVICES=1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      WANDB_PROJECT=${WANDB_PROJECT:-ubt} WANDB_DIR=$UBT_WORK/wandb WANDB_SILENT=true \
      accelerate launch --num_processes 3 --mixed_precision bf16 scripts/train/train_rlvr.py \
      --model $M --train $D/pool_mined.jsonl --out $O/run --max-steps 400 --per-device-bs 4 --grad-accum 16 \
      --num-generations 8 --vllm-port $PORT --save-steps 50 --report-to wandb --run-name rlvr \
      --resume > $O/train.$attempt.log 2>&1
  rc=$?
  stop_server
  log "train exit $rc"
  [ $rc -eq 0 ] && break
  ls -d $O/run/checkpoint-* > /dev/null 2>&1 || { log "FAILED: training died before its first checkpoint"; exit 1; }
done
[ -d $O/run/final ] || { log "FAILED: no final adapter"; exit 1; }

# 3 dev curve (report only)
log "dev curve"
items=(base $(ls -d $O/run/checkpoint-* | sort -t- -k2 -n))
lane() {
  local g=$1 k name
  for ((k = g; k < ${#items[@]}; k += 4)); do
    name=$(basename "${items[$k]}")
    local lora=()
    [ "${items[$k]}" != base ] && lora=(--lora "${items[$k]}")
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train/rlvr_sample.py --model $M "${lora[@]}" --prompts $D/dev.jsonl \
        --out $O/curve/$name.jsonl --n 1 --temperature 0 --max-tokens 1024 > $O/curve/$name.log 2>&1
  done
}
for g in 0 1 2 3; do lane $g & done
wait
$PY - "$O/curve" <<'EOF' > $O/curve/summary.txt
import glob, json, os, sys
rows = []
for p in glob.glob(os.path.join(sys.argv[1], "*.jsonl")):
    name = os.path.basename(p)[:-6]
    step = 0 if name == "base" else int(name.split("-")[1])
    xs = [json.loads(line)["samples"][0] for line in open(p, encoding="utf-8")]
    n = len(xs)
    rows.append((step, n, 100 * sum(min(1.0, x["cer"]) for x in xs) / n, 100 * sum(x["exact"] for x in xs) / n,
                 100 * sum(x["feasible"] for x in xs) / n, 100 * sum(x["conf_ok"] for x in xs) / n,
                 sum(x["reward"] for x in xs) / n))
print("step  docs  CER%(clip)  exact%  feasible%  code-ok%  reward")
for r in sorted(rows):
    print("%4d %5d %10.3f %7.2f %10.2f %9.2f %7.4f" % r)
EOF
cat $O/curve/summary.txt

# 4 merge the step-400 adapter
if [ ! -s $UBT_WORK/models/rlvr/merged_from.json ]; then
  log "merge"
  $PY scripts/eval/greedy_eval.py merge --base $M --ckpt $O/run/final --tokenizer $M \
      --out $UBT_WORK/models/rlvr > $O/merge.log 2>&1 || { log "FAILED: merge"; exit 1; }
fi

# 5 every paper set
log "eval chain (NAME=rlvr)"
mkdir -p $UBT_WORK/evals/rlvr
NAME=rlvr MODEL=$UBT_WORK/models/rlvr scripts/eval/eval_model_chain.sh \
    > $UBT_WORK/evals/rlvr/chain.log 2>&1
log "done"
