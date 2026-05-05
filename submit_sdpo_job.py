from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input, command, Output
from azure.ai.ml.entities import JupyterLabJobService, VsCodeJobService, TensorBoardJobService, SshJobService
from datetime import datetime
from argparse import ArgumentParser
import os

# =========================== CONFIGURATION ===========================
# SDPO config (aligned with run_local_sdpo.sh)
CONFIG_NAME = "sdpo"
DATA_ROOT_PATH = "UI/2026-04-28_173802_UTC/opd_hf"
# Pick one dataset under the Azure folder, e.g. "tooluse", "lcb_v6",
# or "sciknoweval/chemistry". Leave empty only if the parquet files live
# directly under DATA_ROOT_PATH.
DATA_SUBDIR = "lcb_v6"
DATASTORE_NAME = "workspaceblobstore"
TRAIN_FILE = "train.parquet"
VAL_FILE = "test.parquet"

TRAIN_BATCH_SIZE = 32
ROLLOUT_BATCH_SIZE = 4
LR = 1e-5
LAMBDA = 0.0
CLIP_ADV_HIGH = "null"
DONTS_REPROMPT_ON_SELF_SUCCESS = "True"
ALPHA = 0.5
# MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
MODEL_PATH = "Qwen/Qwen3-8B"
# MODEL_PATH = "Qwen/Qwen3.6-27B"

MAX_PROMPT_LENGTH = 1024
MAX_RESPONSE_LENGTH = 4096
ROLLOUT_MAX_MODEL_LEN = 4096
ROLLOUT_MAX_BATCHED_TOKENS = 4096
ROLLOUT_MAX_NUM_SEQS = 64
ROLLOUT_GPU_MEMORY_UTILIZATION = 0.65

VAL_ROLLOUT_BATCH_SIZE = 8
INCLUDE_ANOTHER_SOLUTION = "False"
INCLUDE_FAILURE_SOLUTION = "False"
SUMMARIZE_SOLUTIONS = "True"
SUMMARY_FROM_ALL = "True"
SUMMARY_K = 4

# Compute config
INSTANCE_TYPE = "Singularity.ND96_H100_v5"
INSTANCE_COUNT = 1

# Environment config
ENVIRONMENT = "verl-grpo-xtab:7"

# Experiment config
EXPERIMENT_NAME = "verl-sdpo"
# ====================================================================

# Optional HF token
hf_token = ""
try:
    with open("./tokens/.hf.txt", "r", encoding="utf-8") as f:
        hf_token = f.read().strip()
except Exception:
    print("No HF token found, continuing without it")

# Optional W&B token
wandb_token = os.environ.get("WANDB_API_KEY", "").strip()
if not wandb_token:
    try:
        with open("./tokens/.wandb.txt", "r", encoding="utf-8") as f:
            wandb_token = f.read().strip()
    except Exception:
        print("No W&B token found, continuing without it")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cluster", type=str, default="h100centralusvc")
    parser.add_argument("--resource_group", type=str, default="SingularityH100")
    parser.add_argument("--workspace", type=str, default="H100CentralUS")
    parser.add_argument("--subscription_id", type=str, default="d0c05057-7972-46ff-9bcf-3c932250155e")
    parser.add_argument("--suffix", type=str, default="azure_sdpo")
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
    model_name = MODEL_PATH.replace("/", "-")
    exp_name = (
        f"{DATA_SUBDIR}-Success{INCLUDE_ANOTHER_SOLUTION}-Fail{INCLUDE_FAILURE_SOLUTION}-train{TRAIN_BATCH_SIZE}-alpha{ALPHA}-rollout{ROLLOUT_BATCH_SIZE}"
        f"-lr{LR}-lambda{LAMBDA}-clip_adv_high{CLIP_ADV_HIGH}"
        f"-dross{DONTS_REPROMPT_ON_SELF_SUCCESS}-{model_name}-{args.suffix}"
    )

    hydra_args = [
        f"data.train_batch_size={TRAIN_BATCH_SIZE}",
        f"data.max_prompt_length={MAX_PROMPT_LENGTH}",
        f"data.max_response_length={MAX_RESPONSE_LENGTH}",
        f"data.train_files=[${{inputs.dataset}}/{TRAIN_FILE}]",
        f"data.val_files=[${{inputs.dataset}}/{VAL_FILE}]",
        "trainer.group_name=SDPO-azure",
        "trainer.project_name=sdpo_base",
        "trainer.logger=[console,wandb]",
        "trainer.test_freq=5",
        "trainer.n_gpus_per_node=8",
        f"actor_rollout_ref.rollout.n={ROLLOUT_BATCH_SIZE}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.load_format=safetensors",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={ROLLOUT_GPU_MEMORY_UTILIZATION}",
        f"actor_rollout_ref.rollout.max_model_len={ROLLOUT_MAX_MODEL_LEN}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={ROLLOUT_MAX_BATCHED_TOKENS}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        f"actor_rollout_ref.model.path={MODEL_PATH}",
        "actor_rollout_ref.model.use_shm=True",
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2",
        "+critic.model.override_config.attn_implementation=flash_attention_2",
        f"actor_rollout_ref.actor.optim.lr={LR}",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.self_distillation.distillation_topk=100",
        "algorithm.rollout_correction.rollout_is=token",
        f"actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success={DONTS_REPROMPT_ON_SELF_SUCCESS}",
        f"actor_rollout_ref.actor.self_distillation.alpha={ALPHA}",
        f"actor_rollout_ref.actor.self_distillation.include_another_solution={INCLUDE_ANOTHER_SOLUTION}",
        f"actor_rollout_ref.actor.self_distillation.include_failure_solution={INCLUDE_FAILURE_SOLUTION}",
        f"actor_rollout_ref.actor.self_distillation.summarize_solutions={SUMMARIZE_SOLUTIONS}",
        f"actor_rollout_ref.actor.self_distillation.summary_k={SUMMARY_K}",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
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
echo "Rollout settings: n={ROLLOUT_BATCH_SIZE}, prompt={MAX_PROMPT_LENGTH}, response={MAX_RESPONSE_LENGTH}, max_model_len={ROLLOUT_MAX_MODEL_LEN}, max_batched_tokens={ROLLOUT_MAX_BATCHED_TOKENS}, gpu_mem_util={ROLLOUT_GPU_MEMORY_UTILIZATION}"
ls -la "${{inputs.dataset}}" || true
find "${{inputs.dataset}}" -maxdepth 2 -type f | sort || true

bash "$PROJECT_ROOT/training/verl_training.sh" "{exp_name}" "{CONFIG_NAME}" "{resolved_data_path}" {hydra_args_str}
"""

    display_name = f"sdpo-{timestamp}"

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
                    f"/datastores/workspaceblobstore/paths/cheng/sdpo_models/{timestamp}/"
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
            # 如果你有固定公钥可以加上；没有就先注释掉
            # "ssh": SshJobService(ssh_public_keys="ssh-rsa ..."),
        },
        display_name=display_name,
        description="Submit SDPO training job via verl_training.sh",
    )

    returned_job = ml_client.jobs.create_or_update(training_job, experiment_name=EXPERIMENT_NAME)
    print("Studio URL:", returned_job.studio_url)
    print("Job name:", returned_job.name)
    print("Experiment:", EXPERIMENT_NAME)
    print("Display name:", display_name)
    print("SDPO job submitted successfully.")
