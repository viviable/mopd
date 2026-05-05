from argparse import ArgumentParser
from datetime import datetime
import os

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.entities import JupyterLabJobService, TensorBoardJobService, VsCodeJobService
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# =========================== CONFIGURATION ===========================
DATA_ROOT_PATH = "UI/2026-04-28_173802_UTC/opd_hf"
DATA_SUBDIR = "lcb_v6"
DATASTORE_NAME = "workspaceblobstore"
TRAIN_FILE = "train.parquet"
VAL_FILE = "test.parquet"

STUDENT_MODEL_PATH = "Qwen/Qwen3-4B"
TEACHER_MODEL_PATH = "Qwen/Qwen3-14B"

TEACHER_ROLLOUT_N = 1
TEACHER_TEMPERATURE = 0.6
TEACHER_TOP_P = 0.95
MAX_PROMPT_LENGTH = 1024
MAX_RESPONSE_LENGTH = 1024
MAX_MODEL_LEN = 4096
MAX_BATCHED_TOKENS = 4096
MAX_NUM_SEQS = 64
GPU_MEMORY_UTILIZATION = 0.5
TEACHER_TP = 8

SFT_TRAIN_BATCH_SIZE = 32
SFT_MICRO_BATCH_SIZE_PER_GPU = 1
SFT_MAX_LENGTH = 2048
SFT_LR = 1e-5
SFT_TOTAL_EPOCHS = 3
SFT_TEST_FREQ = 5
SFT_SAVE_FREQ = 5

EVAL_ROLLOUT_N = 8
EVAL_BATCH_SIZE = 32
EVAL_TEMPERATURE = 0.0
EVAL_TOP_P = 1.0
EVAL_DO_SAMPLE = "False"
EVAL_TP = 1

INSTANCE_TYPE = "Singularity.ND96_H100_v5"
INSTANCE_COUNT = 1
GPUS_PER_NODE = 8
ENVIRONMENT = "verl-grpo-xtab:7"
EXPERIMENT_NAME = "verl-gkd"

DEFAULT_SUBSCRIPTION_ID = "d0c05057-7972-46ff-9bcf-3c932250155e"
DEFAULT_RESOURCE_GROUP = "SingularityH100"
DEFAULT_WORKSPACE = "H100CentralUS"
DEFAULT_CLUSTER = "h100centralusvc"
DEFAULT_UAI = (
    "/subscriptions/d0c05057-7972-46ff-9bcf-3c932250155e/"
    "resourceGroups/SingularityH100/providers/Microsoft.ManagedIdentity/"
    "userAssignedIdentities/singularityh100"
)
# ====================================================================


def _read_token_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cluster", type=str, default=DEFAULT_CLUSTER)
    parser.add_argument("--resource_group", type=str, default=DEFAULT_RESOURCE_GROUP)
    parser.add_argument("--workspace", type=str, default=DEFAULT_WORKSPACE)
    parser.add_argument("--subscription_id", type=str, default=DEFAULT_SUBSCRIPTION_ID)
    parser.add_argument("--suffix", type=str, default="azure_gkd")
    parser.add_argument("--data_subdir", type=str, default=DATA_SUBDIR)
    parser.add_argument("--student_model", type=str, default=STUDENT_MODEL_PATH)
    parser.add_argument("--teacher_model", type=str, default=TEACHER_MODEL_PATH)
    parser.add_argument("--teacher_rollout_n", type=int, default=TEACHER_ROLLOUT_N)
    parser.add_argument("--teacher_tp", type=int, default=TEACHER_TP)
    parser.add_argument("--environment", type=str, default=ENVIRONMENT)
    args = parser.parse_args()

    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print("DefaultAzureCredential failed, fallback to InteractiveBrowserCredential:", ex)
        credential = InteractiveBrowserCredential()

    ml_client = MLClient(
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
        credential=credential,
    )

    tokens_dir = os.path.join(PROJECT_ROOT, "tokens")
    hf_token = _read_token_file(os.path.join(tokens_dir, ".hf.txt"))
    if not hf_token:
        print("No HF token found, continuing without it")

    wandb_token = os.environ.get("WANDB_API_KEY", "").strip() or _read_token_file(
        os.path.join(tokens_dir, ".wandb.txt")
    )
    if not wandb_token:
        print("No W&B token found, continuing without it")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resolved_data_path = DATA_ROOT_PATH.rstrip("/")
    if args.data_subdir:
        resolved_data_path = f"{resolved_data_path}/{args.data_subdir.strip('/')}"

    student_name = args.student_model.replace("/", "-")
    teacher_name = args.teacher_model.replace("/", "-")
    exp_name = (
        f"GKD-train{SFT_TRAIN_BATCH_SIZE}-rollout{args.teacher_rollout_n}-lr{SFT_LR}"
        f"-{teacher_name}-to-{student_name}-{args.suffix}"
    )

    vc_cluster = (
        f"/subscriptions/{args.subscription_id}"
        f"/resourceGroups/{args.resource_group}"
        f"/providers/Microsoft.MachineLearningServices/virtualclusters/{args.cluster}"
    )

    data_input_uri = (
        f"azureml://subscriptions/{args.subscription_id}"
        f"/resourcegroups/{args.resource_group}"
        f"/workspaces/{args.workspace}"
        f"/datastores/{DATASTORE_NAME}/paths/{resolved_data_path}/"
    )

    output_uri = (
        f"azureml://subscriptions/{args.subscription_id}"
        f"/resourcegroups/{args.resource_group}"
        f"/workspaces/{args.workspace}"
        f"/datastores/workspaceblobstore/paths/cheng/gkd_models/{timestamp}/"
    )

    command_str = f"""
set -euo pipefail

export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:${{PYTHONPATH:-}}
export USER=${{USER:-$(whoami)}}
export WANDB_ENTITY=safety
export DISABLE_VERSION_CHECK=1
export USE_AZURE_VERL_TRAINING=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export TASK={resolved_data_path}

GEN_DIR=/tmp/gkd_teacher_rollouts
mkdir -p "$GEN_DIR"
TEACHER_TRAIN="$GEN_DIR/train_teacher.parquet"
TEACHER_VAL="$GEN_DIR/val_teacher.parquet"
SFT_TRAIN="$GEN_DIR/train_sft.parquet"
SFT_VAL="$GEN_DIR/val_sft.parquet"
CKPT_DIR="${{outputs.model_dir}}/{resolved_data_path}/{exp_name}"
export TEACHER_TRAIN TEACHER_VAL SFT_TRAIN SFT_VAL CKPT_DIR

echo "Mounted dataset directory: ${{inputs.dataset}}"
echo "Resolved datastore path: {resolved_data_path}"
echo "Teacher model: {args.teacher_model}"
echo "Student model: {args.student_model}"
echo "Teacher rollout n: {args.teacher_rollout_n}"
echo "Checkpoint directory: $CKPT_DIR"
ls -la "${{inputs.dataset}}" || true
mkdir -p "$CKPT_DIR"

echo "Generating teacher rollouts for train split"
python -m verl.trainer.main_generation_server \\
  --config-name ppo_trainer \\
  "data.train_files=[${{inputs.dataset}}/{TRAIN_FILE}]" \\
  "data.prompt_key=prompt" \\
  "+data.output_path=$TEACHER_TRAIN" \\
  "trainer.nnodes=1" \\
  "trainer.n_gpus_per_node={GPUS_PER_NODE}" \\
  "actor_rollout_ref.model.path={args.teacher_model}" \\
  "actor_rollout_ref.model.use_shm=False" \\
  "actor_rollout_ref.rollout.name=vllm" \\
  "actor_rollout_ref.rollout.n={args.teacher_rollout_n}" \\
  "actor_rollout_ref.rollout.tensor_model_parallel_size={args.teacher_tp}" \\
  "actor_rollout_ref.rollout.temperature={TEACHER_TEMPERATURE}" \\
  "actor_rollout_ref.rollout.top_p={TEACHER_TOP_P}" \\
  "actor_rollout_ref.rollout.response_length={MAX_RESPONSE_LENGTH}" \\
  "actor_rollout_ref.rollout.prompt_length={MAX_PROMPT_LENGTH}" \\
  "actor_rollout_ref.rollout.max_model_len={MAX_MODEL_LEN}" \\
  "actor_rollout_ref.rollout.max_num_batched_tokens={MAX_BATCHED_TOKENS}" \\
  "actor_rollout_ref.rollout.max_num_seqs={MAX_NUM_SEQS}" \\
  "actor_rollout_ref.rollout.gpu_memory_utilization={GPU_MEMORY_UTILIZATION}" \\
  "actor_rollout_ref.rollout.load_format=safetensors" \\
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"

ray stop --force || true

echo "Generating teacher rollouts for validation split"
python -m verl.trainer.main_generation_server \\
  --config-name ppo_trainer \\
  "data.train_files=[${{inputs.dataset}}/{VAL_FILE}]" \\
  "data.prompt_key=prompt" \\
  "+data.output_path=$TEACHER_VAL" \\
  "trainer.nnodes=1" \\
  "trainer.n_gpus_per_node={GPUS_PER_NODE}" \\
  "actor_rollout_ref.model.path={args.teacher_model}" \\
  "actor_rollout_ref.model.use_shm=False" \\
  "actor_rollout_ref.rollout.name=vllm" \\
  "actor_rollout_ref.rollout.n=1" \\
  "actor_rollout_ref.rollout.tensor_model_parallel_size={args.teacher_tp}" \\
  "actor_rollout_ref.rollout.temperature=0.0" \\
  "actor_rollout_ref.rollout.top_p=1.0" \\
  "actor_rollout_ref.rollout.response_length={MAX_RESPONSE_LENGTH}" \\
  "actor_rollout_ref.rollout.prompt_length={MAX_PROMPT_LENGTH}" \\
  "actor_rollout_ref.rollout.max_model_len={MAX_MODEL_LEN}" \\
  "actor_rollout_ref.rollout.max_num_batched_tokens={MAX_BATCHED_TOKENS}" \\
  "actor_rollout_ref.rollout.max_num_seqs={MAX_NUM_SEQS}" \\
  "actor_rollout_ref.rollout.gpu_memory_utilization={GPU_MEMORY_UTILIZATION}" \\
  "actor_rollout_ref.rollout.load_format=safetensors" \\
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"

ray stop --force || true

echo "Converting teacher rollouts to multiturn SFT parquet"
python "$PROJECT_ROOT/convert_gkd_rollouts.py" \\
  --train-src "$TEACHER_TRAIN" \\
  --val-src "$TEACHER_VAL" \\
  --train-dst "$SFT_TRAIN" \\
  --val-dst "$SFT_VAL"

SFT_TOTAL_STEPS=$(python - <<'PY'
import math
import os
import pandas as pd

train_path = os.environ["SFT_TRAIN"]
n_rows = len(pd.read_parquet(train_path))
steps_per_epoch = n_rows // {SFT_TRAIN_BATCH_SIZE}
total_steps = steps_per_epoch * {SFT_TOTAL_EPOCHS}
print(max(total_steps, 1))
PY
)
echo "SFT total training steps: $SFT_TOTAL_STEPS"

EVAL_DIR="$CKPT_DIR/gkd_eval"
mkdir -p "$EVAL_DIR"

for TARGET_STEP in $(seq {SFT_TEST_FREQ} {SFT_TEST_FREQ} "$SFT_TOTAL_STEPS"); do
  echo "Training student on teacher rollouts through step $TARGET_STEP"
  torchrun --standalone --nnodes=1 --nproc_per_node={GPUS_PER_NODE} \\
    -m verl.trainer.sft_trainer \\
    --config-name sft_trainer_engine \\
    "data.train_files=$SFT_TRAIN" \\
    "data.val_files=$SFT_VAL" \\
    "data.train_batch_size={SFT_TRAIN_BATCH_SIZE}" \\
    "data.micro_batch_size_per_gpu={SFT_MICRO_BATCH_SIZE_PER_GPU}" \\
    "data.max_length={SFT_MAX_LENGTH}" \\
    "data.truncation=right" \\
    "data.ignore_input_ids_mismatch=True" \\
    "model.path={args.student_model}" \\
    "model.trust_remote_code=True" \\
    "model.use_shm=True" \\
    "engine.model_dtype=bfloat16" \\
    "optim.lr={SFT_LR}" \\
    "trainer.project_name=gkd_base" \\
    "trainer.experiment_name={exp_name}" \\
    "trainer.default_local_dir=$CKPT_DIR" \\
    "trainer.logger=[console,wandb]" \\
    "trainer.total_epochs={SFT_TOTAL_EPOCHS}" \\
    "trainer.total_training_steps=$TARGET_STEP" \\
    "trainer.n_gpus_per_node={GPUS_PER_NODE}" \\
    "trainer.nnodes=1" \\
    "trainer.test_freq=-1" \\
    "trainer.save_freq={SFT_SAVE_FREQ}" \\
    "checkpoint.save_contents=[model,optimizer,extra,hf_model]" \\
    "checkpoint.load_contents=[model,optimizer,extra]"

  HF_CKPT="$CKPT_DIR/global_step_$TARGET_STEP/huggingface"
  if [ ! -d "$HF_CKPT" ]; then
    echo "Expected HF checkpoint not found: $HF_CKPT"
    exit 1
  fi

  ray stop --force || true

  echo "Running SDPO-style rollout eval for GKD checkpoint step $TARGET_STEP"
  python -m verl.trainer.main_ppo \\
    --config-name sdpo \\
    "data.train_files=[${{inputs.dataset}}/{VAL_FILE}]" \\
    "data.val_files=[${{inputs.dataset}}/{VAL_FILE}]" \\
    "vars.task={resolved_data_path}" \\
    "data.prompt_key=prompt" \\
    "data.train_batch_size={EVAL_BATCH_SIZE}" \\
    "data.max_prompt_length={MAX_PROMPT_LENGTH}" \\
    "data.max_response_length={MAX_RESPONSE_LENGTH}" \\
    "data.validation_shuffle=False" \\
    "trainer.group_name=GKD-azure-eval" \\
    "trainer.project_name=gkd_base" \\
    "trainer.experiment_name={exp_name}-eval-step$TARGET_STEP" \\
    "trainer.default_local_dir=$CKPT_DIR/eval_ckpts/step_$TARGET_STEP" \\
    "trainer.logger=[console,wandb]" \\
    "trainer.val_only=True" \\
    "trainer.val_before_train=True" \\
    "trainer.test_freq=-1" \\
    "trainer.n_gpus_per_node={GPUS_PER_NODE}" \\
    "trainer.nnodes=1" \\
    "trainer.validation_data_dir=$EVAL_DIR/step_$TARGET_STEP" \\
    "actor_rollout_ref.model.path=$HF_CKPT" \\
    "actor_rollout_ref.model.use_shm=False" \\
    "actor_rollout_ref.model.use_remove_padding=False" \\
    "actor_rollout_ref.rollout.name=vllm" \\
    "actor_rollout_ref.rollout.n=1" \\
    "actor_rollout_ref.rollout.tensor_model_parallel_size={EVAL_TP}" \\
    "actor_rollout_ref.rollout.load_format=safetensors" \\
    "actor_rollout_ref.rollout.gpu_memory_utilization={GPU_MEMORY_UTILIZATION}" \\
    "actor_rollout_ref.rollout.max_model_len={MAX_MODEL_LEN}" \\
    "actor_rollout_ref.rollout.max_num_batched_tokens={MAX_BATCHED_TOKENS}" \\
    "actor_rollout_ref.rollout.max_num_seqs={MAX_NUM_SEQS}" \\
    "actor_rollout_ref.rollout.enable_chunked_prefill=False" \\
    "actor_rollout_ref.rollout.temperature={EVAL_TEMPERATURE}" \\
    "actor_rollout_ref.rollout.top_p={EVAL_TOP_P}" \\
    "actor_rollout_ref.rollout.val_kwargs.do_sample={EVAL_DO_SAMPLE}" \\
    "actor_rollout_ref.rollout.val_kwargs.temperature={EVAL_TEMPERATURE}" \\
    "actor_rollout_ref.rollout.val_kwargs.top_p={EVAL_TOP_P}" \\
    "actor_rollout_ref.rollout.val_kwargs.n={EVAL_ROLLOUT_N}"

  ray stop --force || true
done

if [ $((SFT_TOTAL_STEPS % {SFT_TEST_FREQ})) -ne 0 ]; then
  TARGET_STEP="$SFT_TOTAL_STEPS"
  echo "Training final partial segment through step $TARGET_STEP"
  torchrun --standalone --nnodes=1 --nproc_per_node={GPUS_PER_NODE} \\
    -m verl.trainer.sft_trainer \\
    --config-name sft_trainer_engine \\
    "data.train_files=$SFT_TRAIN" \\
    "data.val_files=$SFT_VAL" \\
    "data.train_batch_size={SFT_TRAIN_BATCH_SIZE}" \\
    "data.micro_batch_size_per_gpu={SFT_MICRO_BATCH_SIZE_PER_GPU}" \\
    "data.max_length={SFT_MAX_LENGTH}" \\
    "data.truncation=right" \\
    "data.ignore_input_ids_mismatch=True" \\
    "model.path={args.student_model}" \\
    "model.trust_remote_code=True" \\
    "model.use_shm=True" \\
    "engine.model_dtype=bfloat16" \\
    "optim.lr={SFT_LR}" \\
    "trainer.project_name=gkd_base" \\
    "trainer.experiment_name={exp_name}" \\
    "trainer.default_local_dir=$CKPT_DIR" \\
    "trainer.logger=[console,wandb]" \\
    "trainer.total_epochs={SFT_TOTAL_EPOCHS}" \\
    "trainer.total_training_steps=$TARGET_STEP" \\
    "trainer.n_gpus_per_node={GPUS_PER_NODE}" \\
    "trainer.nnodes=1" \\
    "trainer.test_freq=-1" \\
    "trainer.save_freq=1" \\
    "checkpoint.save_contents=[model,optimizer,extra,hf_model]" \\
    "checkpoint.load_contents=[model,optimizer,extra]"
fi
"""

    display_name = f"gkd-{timestamp}"

    training_job = command(
        code=PROJECT_ROOT,
        command=command_str,
        inputs={
            "dataset": Input(type="uri_folder", path=data_input_uri, mode="ro_mount"),
        },
        outputs={
            "model_dir": Output(type="uri_folder", path=output_uri, mode="rw_mount"),
        },
        environment=args.environment,
        environment_variables={
            "HF_TOKEN": hf_token,
            "WANDB_API_KEY": wandb_token,
            "_AZUREML_SINGULARITY_JOB_UAI": DEFAULT_UAI,
        },
        compute=vc_cluster,
        resources={
            "instance_count": INSTANCE_COUNT,
            "instance_type": INSTANCE_TYPE,
            "properties": {
                "singularity": {
                    "interactive": False,
                    "imageVersion": "",
                    "slaTier": "Premium",
                    "priority": "high",
                    "tensorboardLogDirectory": "/scratch/tensorboard_logs",
                    "enableAzmlInt": False,
                }
            },
        },
        services={
            "jupyter": JupyterLabJobService(),
            "vscode": VsCodeJobService(),
            "tensorboard": TensorBoardJobService(log_dir="output/tblog"),
        },
        display_name=display_name,
        description="General KD: generate teacher rollouts, convert to SFT data, train student.",
    )

    returned_job = ml_client.jobs.create_or_update(training_job, experiment_name=EXPERIMENT_NAME)
    print("Studio URL:", returned_job.studio_url)
    print("Job name:", returned_job.name)
    print("Experiment:", EXPERIMENT_NAME)
    print("Display name:", display_name)
    print("Teacher model:", args.teacher_model)
    print("Student model:", args.student_model)
    print("Teacher rollout n:", args.teacher_rollout_n)
    print("General KD job submitted successfully.")
