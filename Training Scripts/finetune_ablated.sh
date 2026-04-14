#!/usr/bin/env bash
set -euo pipefail
############################################
# Fine-tune ablated models on the sentences
# that were removed during corpus ablation.
#
# Usage:
#   bash finetune_ablated.sh
#
# Prerequisites:
#   • The ablated models must already be trained and pushed to HuggingFace:
#       znhoughton/opt-babylm-{size}-ablated-20eps-seed{SEED}
#   • The removed-sentences dataset must be pushed to HuggingFace:
#       znhoughton/babylm-some-binoms-ablated
#     (created by ablate_corpus.py --push-removed-to-hub znhoughton/babylm-some-binoms-ablated)
############################################

export HF_HOME=/workspace/hf_cache

REMOVED_DATASET="znhoughton/binom-ablation-finetune-corpus"
TOKENIZER_NAME="opt-babylm-100m-bpe"
BLOCK_SIZE=1024
WARMUP_STEPS=200
SEED=964

TOKENIZER_PATH="models/${TOKENIZER_NAME}"
VOCAB_SIZE=8192

############################################
# STEP 0: Generate tokenizer if not present
############################################
if [ -d "${TOKENIZER_PATH}" ]; then
  echo "=== Tokenizer already exists at ${TOKENIZER_PATH}, skipping ==="
else
  echo "=== Generating tokenizer ==="
  python tokenizer_and_config.py \
    --base_model facebook/opt-125m \
    --model_name ${TOKENIZER_NAME} \
    --train_file znhoughton/babylm-150m-ablated \
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
# FUNCTION: fine-tune one ablated OPT model
############################################
finetune_opt () {
    MODEL_SIZE=$1
    BATCH=$2
    GRAD_ACCUM=$3
    LR=$4

    BASE_MODEL_ID="znhoughton/opt-babylm-${MODEL_SIZE}-ablated-20eps-seed${SEED}"
    FINETUNE_NAME="opt-babylm-${MODEL_SIZE}-ablated-finetuned-20eps"
    RUN_DIR="/tmp/runs/${FINETUNE_NAME}_${SEED}"

    echo "============================================================"
    echo "=== Fine-tuning ${FINETUNE_NAME} ==="
    echo "=== Base model : ${BASE_MODEL_ID} ==="
    echo "=== Dataset    : ${REMOVED_DATASET} ==="
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_autoreg.py \
        --model_type opt \
        --model_name_or_path ${BASE_MODEL_ID} \
        --tokenizer_name ${TOKENIZER_PATH} \
        --dataset_name ${REMOVED_DATASET} \
        --do_train \
        --bf16 \
        --gradient_checkpointing \
        --block_size ${BLOCK_SIZE} \
        --per_device_train_batch_size ${BATCH} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LR} \
        --warmup_steps ${WARMUP_STEPS} \
        --num_train_epochs 1 \
        --save_strategy epoch \
        --save_total_limit 1 \
        --save_only_model \
        --logging_steps 10 \
        --report_to tensorboard \
        --seed ${SEED} \
        --output_dir ${RUN_DIR} \
        --push_to_hub \
        --hub_model_id znhoughton/${FINETUNE_NAME}-seed${SEED} \
        --hub_strategy end \
        --ddp_find_unused_parameters False

    echo "=== Finished fine-tuning ${FINETUNE_NAME} ==="
    echo "=== Deleting local run directory ${RUN_DIR} ==="
    rm -rf "${RUN_DIR}"
}

############################################
# OPT-125M
############################################
finetune_opt \
  125m \
  64 \
  1 \
  3e-5

############################################
# OPT-350M
############################################
finetune_opt \
  350m \
  32 \
  1 \
  1e-5

############################################
# OPT-1.3B
############################################
finetune_opt \
  1.3b \
  16 \
  1 \
  1e-5
