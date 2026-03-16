#!/bin/bash

# Usage: ./run_local_sdpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo"

# Default to ToolUse dataset
# DATA_PATH="datasets/tooluse"
# DATA_PATH="datasets/sciknoweval/biology"
DATA_PATH="datasets/sciknoweval/chemistry"
# DATA_PATH="datasets/sciknoweval/physics"
# DATA_PATH="datasets/sciknoweval/material"

# DATA_PATH="datasets/lcb_v6"
# DATA_PATH="datasets/G-OPD-Training-Data/Eurus"
# DATA_PATH="datasets/G-OPD-Training-Data/DeepMath-103K_test_aime2025"

# Hyperparameters (from experiments/run_sdpo_all.sh)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
SEED=${SEED:-42}
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
# MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
ROLLOUT_MAX_MODEL_LEN=4096


INCLUDE_ANOTHER_SOLUTION=True
INCLUDE_FAILURE_SOLUTION=True
SUMMARIZE_SOLUTIONS=False
SUMMARY_FROM_ALL=False   # 新增
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

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_sdpo"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}
export WANDB_ENTITY="safety"

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="107228-0-TF-${DATA_PATH##*/}-SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip_adv_high${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"
CKPT_DIR="/project/flame/wyu3/mopd/${EXP_NAME}"

ARGS=(
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.seed=$SEED"
  "data.max_prompt_length=2048"
  "trainer.group_name=SDPO-local"
  "trainer.project_name=sdpo_base"
  "trainer.logger=[console,wandb]"
  "trainer.test_freq=5"
  "trainer.save_freq=50"
  "trainer.default_local_dir=$CKPT_DIR"
  "trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP"
  "trainer.max_critic_ckpt_to_keep=$MAX_CRITIC_CKPT_TO_KEEP"
  "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
  "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
  "actor_rollout_ref.rollout.name=vllm"
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
  "actor_rollout_ref.actor.self_distillation.include_another_solution=$INCLUDE_ANOTHER_SOLUTION"
  "actor_rollout_ref.actor.self_distillation.include_failure_solution=$INCLUDE_FAILURE_SOLUTION"
  "actor_rollout_ref.actor.self_distillation.summarize_solutions=$SUMMARIZE_SOLUTIONS"
  "actor_rollout_ref.actor.self_distillation.summary_k=$SUMMARY_K"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.rollout.val_kwargs.n=8"
)

echo "----------------------------------------------------------------"
echo "Starting Local SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Seed: $SEED"
echo "Checkpoint dir: $CKPT_DIR"
echo "Resolved GPUs: visible=${VISIBLE_GPUS}, trainer.n_gpus_per_node=${N_GPUS_PER_NODE}, rollout.tp=${ROLLOUT_TP_SIZE}"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" "${ARGS[@]}"
