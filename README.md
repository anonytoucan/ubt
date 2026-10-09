# UBT: universal braille-to-text translation

Code for *Universal Braille-to-Text Translation*: one model reads braille in 156 codes without being told the code.
Training data come from forward LibLouis translation; the model is Qwen2.5-7B with the braille cells as single tokens,
trained with SFT and RLVR and decoded with verified best-of-n.

## Layout

```
src/ubt/                   library: translation, data builder (u1/), scoring, RLVR reward
scripts/liblouis/          patched liblouis build and Korean 2024 table checks
scripts/data/              data-u1 build and verification, training splits
scripts/train/             tokenizer, SFT, top-up, RLVR, ByT5 and Gemma
scripts/eval/              greedy and best-of-n evaluation, extra metrics
scripts/real_corpora/      real-corpus test sets (Bible, BrailleLLM, Vision-Braille, NIKL) and adaptation
scripts/baselines/         LibLouis back-translation and GPT-5.5 few-shot
scripts/paper/             paper tables, figures and confidence intervals
scripts/braille_edition/   braille edition of the paper
configs/                   data configuration
patches/                   patched liblouis and the Korean 2024 tables (patches/README.md)
resources/                 Korean Braille Standard 2024 example sentences
tests/                     unit tests
```

## Setup

```
scripts/liblouis/build_liblouis.sh   # patched liblouis 3.38 + Korean 2024 tables; export the printed UBT_LOUIS_* paths
pip install -e ".[dev,experiments]"
```

Nemeth math needs Java and latex2nemeth 1.1.3 (`UBT_LATEX2NEMETH_JAR`). Large files go to `work/` and `datasets/`
(`UBT_WORK`, `UBT_DATASETS`). Run scripts from the repository root.

## Pipeline

**Data**

```
python scripts/data/build_unified.py --out datasets/data_u1 --scale 1.0 --override-gates
python scripts/data/u1_retranscribe.py --data datasets/data_u1 --out datasets/data_u1_iso
python scripts/data/u1_fix_s1b_overlap.py datasets/data_u1_iso datasets/data_u1_s1b
python scripts/data/u1_retranscribe.py --data datasets/data_u1_s1b --out datasets/data_u1_ko \
    --ko-tables-dir work/ko2024_tables --ko-table-file ko-2020-g2.ctb --only-group script:Hangul
python scripts/data/u1_nikl_out_of_train.py datasets/data_u1_ko datasets/data_u1_v2
python scripts/data/verify_unified.py --data datasets/data_u1_v2
```

Training sets: `scripts/data/fixed_split.py`, `scripts/data/u1_topup_build.py`, `scripts/train/rlvr_build_prompts.py`.

**Models**

```
scripts/train/train_models.sh tokenizer | sft | topup | rlvr | byt5 | gemma | ablation-a | ablation-b | ablation-eval
```

**Evaluation**

```
scripts/eval/bon_sft_chain.sh                                       # SFT model
NAME=rlvr MODEL=work/models/rlvr scripts/eval/eval_model_chain.sh   # every evaluation set
```

Baselines: `scripts/baselines/`, `scripts/eval/eval_byt5_chain.sh`, `scripts/eval/eval_gemma_chain.sh`.
Paper tables and figures: `scripts/paper/`.

## Data note

The NIKL Korean-Braille parallel corpus is licence-bound; it is used locally for evaluation only and is not included.

## Tests

```
python -m pytest tests -q
```
