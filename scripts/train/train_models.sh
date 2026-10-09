#!/bin/bash
# Training recipes of every model in the paper, one stage per call (4 GPUs unless noted):
#   scripts/train/train_models.sh tokenizer   cell-token tokenizer (256 braille cells as single tokens)
#   scripts/train/train_models.sh sft         SFT of Qwen2.5-7B on the fixed split          -> models/sft
#   scripts/train/train_models.sh topup       continued SFT on weak codes, from the SFT adapter -> models/topup
#   scripts/train/train_models.sh rlvr        RLVR from the top-up model (rlvr_chain.sh) -> models/rlvr
#   scripts/train/train_models.sh byt5        ByT5-base full fine-tuning (G GPUs)            -> models/byt5
#   scripts/train/train_models.sh gemma       Gemma-2-9B QLoRA with cell tokens (G GPUs)     -> models/gemma_adapter
#   scripts/train/train_models.sh ablation-a  cell-token ablation, original tokenizer, stops at step 400
#   scripts/train/train_models.sh ablation-b  cell-token ablation, cell tokens, same wall clock as arm A
#   scripts/train/train_models.sh ablation-eval
# Inputs (README, "Pipeline"): $UBT_WORK/data/fixed_split (scripts/data/fixed_split.py), $UBT_WORK/data/topup (top-up
# set), $UBT_WORK/data/rlvr/{pool,dev}.jsonl (scripts/train/rlvr_build_prompts.py). REPORT_TO=wandb logs to W&B.
set -euo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TOK=$UBT_WORK/models/qwen25_cell_tokenizer
REPORT_TO=${REPORT_TO:-none}
G=${G:-4}
SFT_ARGS=(--epochs 1 --max-len 4096 --per-device-batch 2 --grad-accum 16 --dev-n 500)
CELL_ARGS=(--tokenizer "$TOK" --braille-vocab --new-token-lr-mult 10)

case "${1:-}" in
  tokenizer)
    python scripts/train/build_cell_tokenizer.py --model Qwen/Qwen2.5-7B --out "$TOK"
    python scripts/train/build_cell_tokenizer.py --model google/gemma-2-9b --out "$UBT_WORK/models/gemma2_cell_tokenizer" --keep-single
    ;;
  sft)
    OUT=$UBT_WORK/ckpt/sft scripts/train/sft_supervise.sh --data "$UBT_WORK/data/fixed_split" "${SFT_ARGS[@]}" "${CELL_ARGS[@]}" \
        --save-steps 250 --eval-steps 250 --ckpt-threshold 1000 --report-to "$REPORT_TO"
    python scripts/eval/greedy_eval.py merge --ckpt "$UBT_WORK/ckpt/sft/final" --tokenizer "$TOK" --out "$UBT_WORK/models/sft"
    ;;
  topup)
    OUT=$UBT_WORK/ckpt/topup scripts/train/sft_supervise.sh --data "$UBT_WORK/data/topup" "${SFT_ARGS[@]}" "${CELL_ARGS[@]}" \
        --save-steps 250 --eval-steps 250 --ckpt-threshold 1000 --init-adapter "$UBT_WORK/ckpt/sft/final" --lr 5e-5 \
        --report-to "$REPORT_TO"
    python scripts/eval/greedy_eval.py merge --ckpt "$UBT_WORK/ckpt/topup/final" --tokenizer "$TOK" --out "$UBT_WORK/models/topup"
    ;;
  rlvr)
    scripts/train/rlvr_chain.sh
    ;;
  byt5)
    # effective batch ~64 documents: per-device 1 x grad-accum x G
    torchrun --nproc_per_node "$G" scripts/train/train_byt5_u1.py --data "$UBT_WORK/data/fixed_split" --out "$UBT_WORK/ckpt/byt5" \
        --model google/byt5-base --epochs 1 --per-device-batch 1 --grad-accum $((64 / G)) --weights-dtype fp32 \
        --report-to "$REPORT_TO"
    ln -sfn "$UBT_WORK/ckpt/byt5/final" "$UBT_WORK/models/byt5"
    ;;
  gemma)
    # effective batch 128 documents; eager attention for Gemma-2's logit soft-capping
    torchrun --nproc_per_node "$G" scripts/train/train_sft_u1.py --data "$UBT_WORK/data/fixed_split" --out "$UBT_WORK/ckpt/gemma" \
        --model google/gemma-2-9b --tokenizer "$UBT_WORK/models/gemma2_cell_tokenizer" --braille-vocab --new-token-lr-mult 10 \
        --attn-impl eager --qlora --lora-r 32 --lora-alpha 64 --max-len 4096 --epochs 1 --per-device-batch 1 \
        --grad-accum $((128 / G)) --report-to "$REPORT_TO"
    ln -sfn "$UBT_WORK/ckpt/gemma/final" "$UBT_WORK/models/gemma_adapter"
    ;;
  ablation-a)
    torchrun --nproc_per_node 4 scripts/train/train_sft_u1.py --data "$UBT_WORK/data/fixed_split" --out "$UBT_WORK/ckpt/ablation_A" \
        "${SFT_ARGS[@]}" --save-steps 100 --eval-steps 50 --ckpt-threshold 1400 --stop-at-step 400 --report-to "$REPORT_TO"
    ;;
  ablation-b)
    # T = wall-clock seconds arm A needed for its first 400 steps
    T=$(python -c "import json; h=json.load(open('$UBT_WORK/ckpt/ablation_A/checkpoint-400/trainer_state.json'))['log_history']; \
print([r['elapsed_sec'] for r in h if r.get('step') == 400 and 'elapsed_sec' in r][0])")
    torchrun --nproc_per_node 4 scripts/train/train_sft_u1.py --data "$UBT_WORK/data/fixed_split" --out "$UBT_WORK/ckpt/ablation_B" \
        "${SFT_ARGS[@]}" "${CELL_ARGS[@]}" --save-steps 100 --eval-steps 50 --ckpt-threshold 1400 \
        --stop-at-seconds "$T" --stop-min-step 400 --report-to "$REPORT_TO"
    ;;
  ablation-eval)
    A=$UBT_WORK/ckpt/ablation_A B=$UBT_WORK/ckpt/ablation_B M=$UBT_WORK/models E=$UBT_WORK/data/ablation_eval R=$UBT_WORK/ablation
    SB=$(ls -d "$B"/checkpoint-* | sed 's/.*-//' | sort -n | tail -1)
    mkdir -p "$R"
    python scripts/eval/greedy_eval.py build --out "$E"
    python scripts/eval/greedy_eval.py merge --ckpt "$A/checkpoint-400" --tokenizer Qwen/Qwen2.5-7B --out "$M/ablation_A_s400"
    python scripts/eval/greedy_eval.py merge --ckpt "$B/checkpoint-$SB" --tokenizer "$TOK" --out "$M/ablation_B_s$SB"
    source scripts/eval/env_vllm_sm120.sh
    for i in 0 1; do
      CUDA_VISIBLE_DEVICES=$i python scripts/eval/greedy_eval.py gen --model "$M/ablation_A_s400" --eval "$E" \
          --out "$R/gen_A.$i.jsonl" --shard "$i/2" &
      CUDA_VISIBLE_DEVICES=$((i + 2)) python scripts/eval/greedy_eval.py gen --model "$M/ablation_B_s$SB" --eval "$E" \
          --out "$R/gen_B.$i.jsonl" --shard "$i/2" &
    done
    wait
    python scripts/eval/greedy_eval.py score --eval "$E" --gen "A=$R/gen_A.*.jsonl" --gen "B=$R/gen_B.*.jsonl" --out "$R/report.json"
    python scripts/eval/greedy_eval.py decide --run-a "$A" --run-b "$B" --out "$R/decide.json"
    ;;
  *)
    sed -n '2,13p' "$0"; exit 1
    ;;
esac
