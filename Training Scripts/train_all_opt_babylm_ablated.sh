#!/usr/bin/env bash
set -euo pipefail
############################################
# USER SETTINGS
############################################
export HF_HOME=/workspace/hf_cache

DATASET="znhoughton/babylm-150m-ablated"
TOKENIZER_NAME="opt-babylm-100m-bpe"
BLOCK_SIZE=1024
VOCAB_SIZE=8192
# TARGET: 20M tokens per checkpoint
TOKENS_PER_CHECKPOINT=20000000
SAVE_TOTAL_LIMIT=1
SEED=964
############################################
# STEP 0: Train tokenizer ONCE
# Reuses the same tokenizer as the baseline
# (vocabulary is identical; only training
#  data differs).
############################################
TOKENIZER_PATH="models/${TOKENIZER_NAME}"
if [ -d "${TOKENIZER_PATH}" ]; then
  echo "=== Tokenizer already exists at ${TOKENIZER_PATH}, skipping ==="
else
  echo "=== Training tokenizer ==="
  python tokenizer_and_config.py \
    --base_model facebook/opt-125m \
    --model_name ${TOKENIZER_NAME} \
    --train_file ${DATASET} \
    --from_iterator \
    --bpe \
    --vocab ${VOCAB_SIZE} \
    --hidden_size 768 \
    --attention_heads 12 \
    --layers 12 \
    --intermediate_size 3072 \
    --max_len ${BLOCK_SIZE}
fi
############################################
# FUNCTION: train one OPT model
# Args: MODEL_SIZE BASE_MODEL HIDDEN HEADS
#       LAYERS FFN BATCH GRAD_ACCUM LR N_GPUS WARMUP
############################################
train_opt () {
    MODEL_SIZE=$1
    BASE_MODEL=$2
    HIDDEN=$3
    HEADS=$4
    LAYERS=$5
    FFN=$6
    BATCH=$7
    GRAD_ACCUM=$8
    LR=$9
    N_GPUS=${10}
    WARMUP_STEPS=${11}

    # Build CUDA_VISIBLE_DEVICES string (e.g. "0,1" or "0,1,2,3")
    CUDA_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))

    # Calculate tokens per step and save_steps
    TOKENS_PER_STEP=$((BLOCK_SIZE * BATCH * GRAD_ACCUM * N_GPUS))
    SAVE_STEPS=$((TOKENS_PER_CHECKPOINT / TOKENS_PER_STEP))

    MODEL_NAME="opt-babylm-${MODEL_SIZE}-ablated-20eps"
    MODEL_PATH="models/${MODEL_NAME}"
    RUN_DIR="/tmp/runs/${MODEL_NAME}_${SEED}-20eps"

    echo "============================================================"
    echo "=== Training ${MODEL_NAME} ==="
    echo "=== GPUs: ${N_GPUS} | Tokens/step: ${TOKENS_PER_STEP} ==="
    echo "=== Save every ${SAVE_STEPS} steps (${TOKENS_PER_CHECKPOINT} tokens) ==="
    echo "============================================================"

    # Build config (cheap, safe to re-run)
    python tokenizer_and_config.py \
        --base_model ${BASE_MODEL} \
        --model_name ${MODEL_NAME} \
        --train_file ${DATASET} \
        --from_iterator \
        --bpe \
        --vocab ${VOCAB_SIZE} \
        --hidden_size ${HIDDEN} \
        --attention_heads ${HEADS} \
        --layers ${LAYERS} \
        --intermediate_size ${FFN} \
        --max_len ${BLOCK_SIZE}

    # Train
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${N_GPUS} train_autoreg.py \
        --model_type opt \
        --config_name ${MODEL_PATH} \
        --tokenizer_name ${TOKENIZER_PATH} \
        --dataset_name ${DATASET} \
        --do_train \
        --bf16 \
        --block_size ${BLOCK_SIZE} \
        --per_device_train_batch_size ${BATCH} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --optim adamw_torch_fused \
        --learning_rate ${LR} \
        --warmup_steps ${WARMUP_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --save_total_limit ${SAVE_TOTAL_LIMIT} \
        --save_only_model \
        --logging_steps 10 \
        --report_to tensorboard \
        --num_train_epochs 20 \
        --seed ${SEED} \
        --output_dir ${RUN_DIR} \
        --push_to_hub \
        --hub_model_id znhoughton/${MODEL_NAME}-seed${SEED} \
        --hub_strategy checkpoint \
        --gradient_checkpointing \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --torch_compile \
        --ddp_find_unused_parameters False \
        --overwrite_output_dir

    echo "=== Finished training ${MODEL_NAME} ==="
    # IMPORTANT: free disk before next model
    echo "=== Deleting local run directory ${RUN_DIR} ==="
    rm -rf "${RUN_DIR}"
}


############################################
# OPT-125M
# 2 GPUs: tokens/step = 1024 × 400 × 1 × 2 = 819,200
# total steps ≈ 3660; warmup = 366 (10%)
# save_steps = 20M / 819,200 ≈ 24 steps
############################################
train_opt \
  125m \
  facebook/opt-125m \
  768 \
  12 \
  12 \
  3072 \
  400 \
  1 \
  3e-4 \
  2 \
  366

############################################
# OPT-350M
# 2 GPUs: tokens/step = 1024 × 200 × 1 × 2 = 409,600
# total steps ≈ 7320; warmup = 732 (10%)
# save_steps = 20M / 409,600 ≈ 48 steps
############################################
train_opt \
  350m \
  facebook/opt-350m \
  1024 \
  16 \
  24 \
  4096 \
  200 \
  1 \
  1e-4 \
  2 \
  732

############################################
# OPT-1.3B - 4x A100 80GB (Flash Attention)
# 4 GPUs: tokens/step = 1024 × 250 × 1 × 4 = 1,024,000
# total steps ≈ 2928; warmup = 293 (10%)
# save_steps = 20M / 1,024,000 ≈ 19 steps
############################################
train_opt \
  1.3b \
  facebook/opt-1.3b \
  2048 \
  32 \
  24 \
  8192 \
  250 \
  1 \
  1e-4 \
  4 \
  293
