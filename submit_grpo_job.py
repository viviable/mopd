from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input, command, Output
from azure.ai.ml.entities import JupyterLabJobService, VsCodeJobService, TensorBoardJobService
from datetime import datetime
from argparse import ArgumentParser
import os

# =========================== CONFIGURATION ===========================
CONFIG_NAME = "baseline_grpo"
DATA_ROOT_PATH = "UI/2026-04-28_173802_UTC/opd_hf"
DATA_SUBDIR = "lcb_v6"
DATASTORE_NAME = "workspaceblobstore"
TRAIN_FILE = "train.parquet"
VAL_FILE = "test.parquet"

TRAIN_BATCH_SIZE = 32
ROLLOUT_BATCH_SIZE = 8
MINI_BATCH_SIZE = 32
LR = 1e-5
MODEL_PATH = "Qwen/Qwen3-4B"
SEED = 42

MAX_PROMPT_LENGTH = 2048
MAX_RESPONSE_LENGTH = 1024
ROLLOUT_TP_SIZE = 1
ROLLOUT_GPU_MEMORY_UTILIZATION = 0.75
VAL_ROLLOUT_BATCH_SIZE = 8

INSTANCE_TYPE = "Singularity.ND96_H100_v5"
INSTANCE_COUNT = 1

ENVIRONMENT = "verl-grpo-xtab:7"
EXPERIMENT_NAME = "verl-grpo"
# ====================================================================


def _read_token_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


hf_token = _read_token_file("./tokens/.hf.txt")
if not hf_token:
    print("No HF token found, continuing without it")

wandb_token = os.environ.get("WANDB_API_KEY", "").strip() or _read_token_file("./tokens/.wandb.txt")
if not wandb_token:
    print("No W&B token found, continuing without it")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cluster", type=str, default="h100centralusvc")
    parser.add_argument("--resource_group", type=str, default="SingularityH100")
    parser.add_argument("--workspace", type=str, default="H100CentralUS")
    parser.add_argument("--subscription_id", type=str, default="d0c05057-7972-46ff-9bcf-3c932250155e")
    parser.add_argument("--suffix", type=str, default="azure_grpo")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
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

    vc_cluster = (
        f"/subscriptions/{args.subscription_id}"
        f"/resourceGroups/{args.resource_group}"
        f"/providers/Microsoft.MachineLearningServices/virtualclusters/{args.cluster}"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resolved_data_path = DATA_ROOT_PATH.rstrip("/")
    if DATA_SUBDIR:
        resolved_data_path = f"{resolved_data_path}/{DATA_SUBDIR.strip('/')}"
    model_name = args.model_path.replace("/", "-")
    exp_name = (
        f"GRPO-train{TRAIN_BATCH_SIZE}-mbs{MINI_BATCH_SIZE}-rollout{ROLLOUT_BATCH_SIZE}"
        f"-lr{LR}-seed{SEED}-{model_name}-{args.suffix}"
    )

    hydra_args = [
        "algorithm.adv_estimator=grpo",
        f"data.train_batch_size={TRAIN_BATCH_SIZE}",
        f"data.max_prompt_length={MAX_PROMPT_LENGTH}",
        f"data.max_response_length={MAX_RESPONSE_LENGTH}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.seed=42",
        f"data.train_files=[${{inputs.dataset}}/{TRAIN_FILE}]",
        f"data.val_files=[${{inputs.dataset}}/{VAL_FILE}]",
        "trainer.group_name=GRPO-azure",
        "trainer.project_name=grpo_base",
        "trainer.logger=[console,wandb]",
        "trainer.val_before_train=True",
        "trainer.test_freq=20",
        "trainer.n_gpus_per_node=8",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        f"actor_rollout_ref.actor.data_loader_seed={SEED}",
        f"actor_rollout_ref.actor.fsdp_config.seed={SEED}",
        f"actor_rollout_ref.rollout.n={ROLLOUT_BATCH_SIZE}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={ROLLOUT_TP_SIZE}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={ROLLOUT_GPU_MEMORY_UTILIZATION}",
        "actor_rollout_ref.rollout.load_format=safetensors",
        f"actor_rollout_ref.actor.optim.lr={LR}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={MINI_BATCH_SIZE}",
        f"critic.data_loader_seed={SEED}",
        f"actor_rollout_ref.model.path={args.model_path}",
        "actor_rollout_ref.model.use_shm=True",
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2",
        "+critic.model.override_config.attn_implementation=flash_attention_2",
        "algorithm.rollout_correction.rollout_is=token",
        f"actor_rollout_ref.rollout.val_kwargs.n={VAL_ROLLOUT_BATCH_SIZE}",
    ]

    hydra_args_str = " ".join([f'"{x}"' for x in hydra_args])
    data_input_uri = (
        f"azureml://subscriptions/{args.subscription_id}"
        f"/resourcegroups/{args.resource_group}"
        f"/workspaces/{args.workspace}"
        f"/datastores/{DATASTORE_NAME}/paths/{resolved_data_path}/"
    )

    command_str = f"""
export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:${{PYTHONPATH:-}}
export USER=${{USER:-$(whoami)}}
export WANDB_ENTITY=safety
export DISABLE_VERSION_CHECK=1
export USE_AZURE_VERL_TRAINING=1
export HYDRA_FULL_ERROR=1

echo "Mounted dataset directory: ${{inputs.dataset}}"
echo "Resolved datastore path: {resolved_data_path}"
echo "Model: {args.model_path}"
echo "GRPO settings: train_batch={TRAIN_BATCH_SIZE}, mini_batch={MINI_BATCH_SIZE}, rollout_n={ROLLOUT_BATCH_SIZE}, prompt={MAX_PROMPT_LENGTH}, response={MAX_RESPONSE_LENGTH}, rollout_tp={ROLLOUT_TP_SIZE}, gpu_mem_util={ROLLOUT_GPU_MEMORY_UTILIZATION}"
ls -la "${{inputs.dataset}}" || true
find "${{inputs.dataset}}" -maxdepth 2 -type f | sort || true

bash "$PROJECT_ROOT/training/verl_training.sh" "{exp_name}" "{CONFIG_NAME}" "{resolved_data_path}" {hydra_args_str}
"""

    display_name = f"grpo-{timestamp}"

    training_job = command(
        code=".",
        command=command_str,
        inputs={
            "dataset": Input(
                type="uri_folder",
                path=data_input_uri,
                mode="ro_mount",
            ),
        },
        outputs={
            "model_dir": Output(
                type="uri_folder",
                path=(
                    f"azureml://subscriptions/{args.subscription_id}"
                    f"/resourcegroups/{args.resource_group}"
                    f"/workspaces/{args.workspace}"
                    f"/datastores/workspaceblobstore/paths/cheng/grpo_models/{timestamp}/"
                ),
                mode="rw_mount",
            ),
        },
        environment=ENVIRONMENT,
        environment_variables={
            "HF_TOKEN": hf_token,
            "WANDB_API_KEY": wandb_token,
            "_AZUREML_SINGULARITY_JOB_UAI": (
                "/subscriptions/d0c05057-7972-46ff-9bcf-3c932250155e/"
                "resourceGroups/SingularityH100/providers/Microsoft.ManagedIdentity/"
                "userAssignedIdentities/singularityh100"
            ),
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
        description="Submit GRPO training job via verl_training.sh",
    )

    returned_job = ml_client.jobs.create_or_update(training_job, experiment_name=EXPERIMENT_NAME)
    print("Studio URL:", returned_job.studio_url)
    print("Job name:", returned_job.name)
    print("Experiment:", EXPERIMENT_NAME)
    print("Display name:", display_name)
    print("Model:", args.model_path)
    print("GRPO job submitted successfully.")
