#!/bin/bash

# Usage: ./run_local_sdpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo"

# Default to ToolUse dataset
# DATA_PATH="datasets/tooluse"
# DATA_PATH="datasets/sciknoweval/biology"
# DATA_PATH="datasets/sciknoweval/chemistry"
# DATA_PATH="datasets/sciknoweval/physics"
# DATA_PATH="datasets/sciknoweval/material"

DATA_PATH="datasets/lcb_v6"
# DATA_PATH="datasets/G-OPD-Training-Data/Eurus"
# DATA_PATH="datasets/gsm8k"
# DATA_PATH="datasets/G-OPD-Training-Data/DeepMath-103K_test_aime2025"
# DATA_PATH="datasets/G-OPD-Training-Data/DeepMath-103K_test_aime2024"
# DATA_PATH="datasets/G-OPD-Training-Data/opd-math"
# Optional validation override. Useful when training on one dataset but validating
# on a separate benchmark parquet, e.g. Humaneval+/MBPP+.
# VAL_DATA_PATH="datasets/humanevalplus/humanevalplus/test.parquet"
# VAL_DATA_PATH="datasets/gsm8k/test.parquet"
# VAL_DATA_PATH="datasets/mbppplus/test.parquet"
# VAL_DATA_PATH="${VAL_DATA_PATH:-}"

# Hyperparameters (from experiments/run_sdpo_all.sh)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
SEED=${SEED:-42}
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
INCLUDE_PRIMARY_SOLUTION=${INCLUDE_PRIMARY_SOLUTION:-True}
FAILURE_SOLUTION_CONDITION=${FAILURE_SOLUTION_CONDITION:-when_no_solution}
SUMMARY_SOURCE=${SUMMARY_SOURCE:-success}
# MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
# MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"


ROLLOUT_MAX_MODEL_LEN=4096


INCLUDE_ANOTHER_SOLUTION=True
INCLUDE_FAILURE_SOLUTION=True
SUMMARIZE_SOLUTIONS=False
SUMMARY_FROM_ALL=False
SUMMARY_K=8

MAX_ACTOR_CKPT_TO_KEEP=2
MAX_CRITIC_CKPT_TO_KEEP=2

# Local-safe default: 1 GPU unless user explicitly pins devices.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ "${CUDA_VISIBLE_DEVICES}" = "-1" ]; then
        VISIBLE_GPUS=0
    else
        VISIBLE_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^\s*$/d' | wc -l)
    fi
else
    VISIBLE_GPUS=1
fi

if [ "${VISIBLE_GPUS}" -lt 1 ]; then
    echo "No visible GPUs detected. This script requires at least 1 GPU."
    exit 1
fi

# Allow overriding from shell, but clamp to visible devices to avoid vLLM init assertion.
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$VISIBLE_GPUS}
if [ "${N_GPUS_PER_NODE}" -gt "${VISIBLE_GPUS}" ]; then
    echo "N_GPUS_PER_NODE (${N_GPUS_PER_NODE}) > visible GPUs (${VISIBLE_GPUS}); clamping to ${VISIBLE_GPUS}."
    export N_GPUS_PER_NODE=${VISIBLE_GPUS}
fi

ROLLOUT_TP_SIZE=1
if [ "${ROLLOUT_TP_SIZE}" -gt "${VISIBLE_GPUS}" ]; then
    echo "ROLLOUT_TP_SIZE (${ROLLOUT_TP_SIZE}) > visible GPUs (${VISIBLE_GPUS}); clamping to ${VISIBLE_GPUS}."
    ROLLOUT_TP_SIZE=${VISIBLE_GPUS}
fi

if [ "${ROLLOUT_BACKEND}" = "vllm" ] && [[ "${MODEL_PATH}" == Qwen/Qwen3.5-* ]]; then
    cat <<'EOF'
Unsupported local combo detected:
- rollout backend: vllm
- model: Qwen3.5

This repo's async rollout path uses the vLLM V1 engine, and vllm==0.8.5.post1 does not support
Qwen3.5 architectures here. Use one of these instead:
- MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 ./run_local_sdpo.sh
- MODEL_PATH=Qwen/Qwen2.5-3B-Instruct ./run_local_sdpo.sh
- upgrade vllm to a version that supports Qwen3.5 in this async path
- switch to sglang if your environment supports it: ROLLOUT_BACKEND=sglang ./run_local_sdpo.sh
EOF
    exit 1
fi

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_sdpo"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Bootstrap the canonical GSM8K parquet layout when requested.
if [ "$DATA_PATH" = "datasets/gsm8k" ]; then
    GSM8K_TRAIN_FILE="$PROJECT_ROOT/$DATA_PATH/train.parquet"
    GSM8K_TEST_FILE="$PROJECT_ROOT/$DATA_PATH/test.parquet"

    if [ ! -f "$GSM8K_TRAIN_FILE" ] || [ ! -f "$GSM8K_TEST_FILE" ]; then
        echo "GSM8K parquet files not found under $PROJECT_ROOT/$DATA_PATH. Generating them from openai/gsm8k..."
        mkdir -p "$PROJECT_ROOT/$DATA_PATH"
        python "$PROJECT_ROOT/examples/data_preprocess/gsm8k.py" --local_save_dir "$PROJECT_ROOT/$DATA_PATH"
    fi
fi

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}
export WANDB_ENTITY="safety"

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="Success${INCLUDE_ANOTHER_SOLUTION}Fail${INCLUDE_FAILURE_SOLUTION}-${DATA_PATH##*/}-SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip_adv_high${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"
CKPT_DIR="/project/flame/wyu3/mopd/${EXP_NAME}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-/project/flame/wyu3/mopd/rollout/${EXP_NAME}}"

ARGS=(
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.seed=$SEED"
  "data.max_prompt_length=2048"
  "trainer.group_name=SDPO-local"
  "trainer.project_name=sdpo_base"
  "trainer.logger=[console,wandb]"
  "trainer.val_before_train=True"
  "trainer.test_freq=5"
  "trainer.save_freq=50"
  "trainer.default_local_dir=$CKPT_DIR"
  "trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP"
  "trainer.max_critic_ckpt_to_keep=$MAX_CRITIC_CKPT_TO_KEEP"
  "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
  "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
  "actor_rollout_ref.rollout.name=$ROLLOUT_BACKEND"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
  "actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_MODEL_LEN"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "actor_rollout_ref.model.use_remove_padding=False"
  "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2"
  "+critic.model.override_config.attn_implementation=flash_attention_2"
  "actor_rollout_ref.actor.optim.lr=$LR"
  "actor_rollout_ref.actor.data_loader_seed=$SEED"
  "critic.data_loader_seed=$SEED"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
  "algorithm.rollout_correction.rollout_is=token"
  "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS}"
  "actor_rollout_ref.actor.self_distillation.alpha=$ALPHA"
  "actor_rollout_ref.actor.self_distillation.include_primary_solution=${INCLUDE_PRIMARY_SOLUTION}"
  "actor_rollout_ref.actor.self_distillation.include_another_solution=$INCLUDE_ANOTHER_SOLUTION"
  "actor_rollout_ref.actor.self_distillation.include_failure_solution=$INCLUDE_FAILURE_SOLUTION"
  "actor_rollout_ref.actor.self_distillation.failure_solution_condition=${FAILURE_SOLUTION_CONDITION}"
  "actor_rollout_ref.actor.self_distillation.summarize_solutions=$SUMMARIZE_SOLUTIONS"
  "actor_rollout_ref.actor.self_distillation.summary_source=${SUMMARY_SOURCE}"
  "actor_rollout_ref.actor.self_distillation.summary_from_all=$SUMMARY_FROM_ALL"
  "actor_rollout_ref.actor.self_distillation.summary_k=$SUMMARY_K"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.rollout.val_kwargs.n=8"
)

if [ -n "$VAL_DATA_PATH" ]; then
  ARGS+=("data.val_files=['$PROJECT_ROOT/$VAL_DATA_PATH']")
fi

if [ -n "$ROLLOUT_DATA_DIR" ]; then
  ARGS+=("trainer.rollout_data_dir=$ROLLOUT_DATA_DIR")
fi

echo "----------------------------------------------------------------"
echo "Starting Local SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Validation data: ${VAL_DATA_PATH:-${DATA_PATH}/test.parquet}"
echo "Model: $MODEL_PATH"
echo "Rollout backend: $ROLLOUT_BACKEND"
echo "Seed: $SEED"
echo "Checkpoint dir: $CKPT_DIR"
echo "Summary from all: $SUMMARY_FROM_ALL"
echo "Rollout data dir: ${ROLLOUT_DATA_DIR:-disabled}"
echo "Resolved GPUs: visible=${VISIBLE_GPUS}, trainer.n_gpus_per_node=${N_GPUS_PER_NODE}, rollout.tp=${ROLLOUT_TP_SIZE}"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" "${ARGS[@]}"
