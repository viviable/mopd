from argparse import ArgumentParser
from datetime import datetime
import os
import shlex

from azure.ai.ml import Input, MLClient, Output, PyTorchDistribution, command
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# =========================== CONFIGURATION ===========================
CONFIG_NAME = "sdpo"
DATA_ROOT_PATH = "UI/2026-04-28_173802_UTC/opd_hf"
DATA_SUBDIR = "tooluse"
DATASTORE_NAME = "workspaceblobstore"
TRAIN_FILE = "train.parquet"
VAL_FILE = "test.parquet"

TRAIN_BATCH_SIZE = 32
ROLLOUT_BATCH_SIZE = 4
VAL_ROLLOUT_BATCH_SIZE = 8
LR = 1e-5
LAMBDA = 0.0
CLIP_ADV_HIGH = "null"
DONTS_REPROMPT_ON_SELF_SUCCESS = "True"
ALPHA = 0.5
TEACHER_UPDATE_RATE = 0.01

DEFAULT_MODEL_PATH = "Qwen/Qwen3-14B"

MAX_PROMPT_LENGTH = 1024
MAX_RESPONSE_LENGTH = 1024
ROLLOUT_MAX_MODEL_LEN = 2048
ROLLOUT_MAX_BATCHED_TOKENS = 2048
ROLLOUT_MAX_NUM_SEQS = 32
ROLLOUT_GPU_MEMORY_UTILIZATION = 0.2
ROLLOUT_TENSOR_PARALLEL_SIZE = 8

INCLUDE_ANOTHER_SOLUTION = "False"
INCLUDE_FAILURE_SOLUTION = "False"
SUMMARIZE_SOLUTIONS = "False"
SUMMARY_K = 8

ACTOR_PARAM_OFFLOAD = "True"
ACTOR_OPTIMIZER_OFFLOAD = "True"
REF_PARAM_OFFLOAD = "True"

DEFAULT_INSTANCE_TYPE = "Singularity.ND96_H100_v5"
DEFAULT_INSTANCE_COUNT = 2
DEFAULT_GPUS_PER_NODE = 8

ENVIRONMENT = "verl-grpo-xtab:7"
EXPERIMENT_NAME = "verl-sdpo-multinode"

DEFAULT_SUBSCRIPTION_ID = "d0c05057-7972-46ff-9bcf-3c932250155e"
DEFAULT_RESOURCE_GROUP = "SingularityH100"
DEFAULT_WORKSPACE = "H100CentralUS"
DEFAULT_CLUSTER = "h100centralusvc"
DEFAULT_UAI = (
    "/subscriptions/d0c05057-7972-46ff-9bcf-3c932250155e/"
    "resourceGroups/SingularityH100/providers/Microsoft.ManagedIdentity/"
    "userAssignedIdentities/singularityh100"
)
# ===================================================================


def _read_token_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _build_command_str(
    exp_name: str,
    resolved_data_path: str,
    hydra_args_str: str,
    model_path: str,
    instance_count: int,
    gpus_per_node: int,
    clear_hf_cache: bool,
    use_shm: bool,
) -> str:
    return f"""
set -euo pipefail

export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:${{PYTHONPATH:-}}
export USER=${{USER:-$(whoami)}}
export WANDB_ENTITY=safety
export DISABLE_VERSION_CHECK=1
export USE_AZURE_VERL_TRAINING=1
export HYDRA_FULL_ERROR=1
export RAY_ADDRESS=auto
export NUM_NODES={instance_count}
export GPUS_PER_NODE={gpus_per_node}
export RAY_HEAD_PORT=6379
export RAY_DASHBOARD_PORT=8265
export CLEAN_SHM_CACHE=1
export CLEAR_HF_MODEL_CACHE={"1" if clear_hf_cache else "0"}
export USE_SHM_MODEL_CACHE={"1" if use_shm else "0"}
export HF_CLEAN_MODEL_1={shlex.quote(model_path)}

NODE_RANK_VALUE="${{NODE_RANK:-}}"
if [ -z "$NODE_RANK_VALUE" ]; then
  NODE_RANK_VALUE="${{OMPI_COMM_WORLD_RANK:-0}}"
fi
export NODE_RANK_VALUE

MASTER_HOST="${{MASTER_ADDR:-}}"
if [ -z "$MASTER_HOST" ]; then
  MASTER_HOST="${{AZ_BATCH_MASTER_NODE:-}}"
fi
export MASTER_HOST

MASTER_PORT_VALUE="${{MASTER_PORT:-6105}}"
export MASTER_PORT_VALUE

echo "Mounted dataset directory: ${{inputs.dataset}}"
echo "Resolved datastore path: {resolved_data_path}"
echo "Model: {model_path}"
echo "Rollout settings: n={ROLLOUT_BATCH_SIZE}, tp={ROLLOUT_TENSOR_PARALLEL_SIZE}, prompt={MAX_PROMPT_LENGTH}, response={MAX_RESPONSE_LENGTH}, max_model_len={ROLLOUT_MAX_MODEL_LEN}, max_batched_tokens={ROLLOUT_MAX_BATCHED_TOKENS}, gpu_mem_util={ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "Shared-memory model cache: $USE_SHM_MODEL_CACHE"
echo "AML distributed env: NODE_RANK=$NODE_RANK_VALUE MASTER_ADDR=$MASTER_HOST MASTER_PORT=$MASTER_PORT_VALUE NUM_NODES=$NUM_NODES GPUS_PER_NODE=$GPUS_PER_NODE"
hostname
df -h /dev/shm || true
ls -la "${{inputs.dataset}}" || true
find "${{inputs.dataset}}" -maxdepth 2 -type f | sort || true

if [ "$CLEAN_SHM_CACHE" = "1" ]; then
  echo "Clearing /dev/shm/verl-cache/*"
  rm -rf /dev/shm/verl-cache/* || true
fi

if [ "$CLEAR_HF_MODEL_CACHE" = "1" ]; then
  echo "Clearing Hugging Face cache for selected model(s)"
  python - <<'PY'
import os
import shutil

models = [os.environ.get("HF_CLEAN_MODEL_1", "").strip()]
models = [m for m in models if m]

roots = []
hf_home = os.environ.get("HF_HOME", "").strip()
home = os.path.expanduser("~")
if hf_home:
    roots.append(os.path.join(hf_home, "hub"))
roots.extend(
    [
        os.path.join(home, ".cache", "huggingface", "hub"),
        os.path.join(home, ".cache", "hf", "hub"),
    ]
)

seen = set()
for model in models:
    repo_dir = "models--" + model.replace("/", "--")
    for root in roots:
        path = os.path.join(root, repo_dir)
        if path in seen:
            continue
        seen.add(path)
        if os.path.isdir(path):
            print("Removing HF cache:", path)
            shutil.rmtree(path, ignore_errors=True)
PY
fi

echo "Preparing Hugging Face model snapshot on this node"
python - <<'PY'
import os
import sys

model = os.environ.get("HF_CLEAN_MODEL_1", "").strip()
if not model:
    print("No model configured for HF snapshot preparation")
    sys.exit(0)

if os.path.exists(model):
    print(f"Model path is already local: {{model}}")
    sys.exit(0)

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    print(f"WARNING: could not import huggingface_hub for snapshot preflight: {{exc}}")
    sys.exit(0)

try:
    local_path = snapshot_download(
        repo_id=model,
        allow_patterns=[
            "*.json",
            "*.model",
            "*.txt",
            "*.safetensors",
            "*.py",
        ],
    )
except Exception as exc:
    print(f"ERROR: failed to download HF model snapshot for {{model}}: {{exc}}")
    raise

required = ["config.json", "model.safetensors.index.json"]
missing = [name for name in required if not os.path.exists(os.path.join(local_path, name))]
shards = [name for name in os.listdir(local_path) if name.endswith(".safetensors")]
if missing or not shards:
    raise RuntimeError(
        f"Incomplete HF snapshot for {{model}} at {{local_path}}; "
        f"missing={{missing}}, safetensors_shards={{len(shards)}}"
    )

print(f"HF model snapshot ready: {{local_path}}")
print(f"Safetensors shard count: {{len(shards)}}")
PY

if [ -z "$MASTER_HOST" ]; then
  echo "MASTER_ADDR is empty; falling back to local node IP"
  MASTER_HOST=$(hostname -I | awk '{{print $1}}')
fi

ray stop --force || true

if [ "$NODE_RANK_VALUE" = "0" ]; then
  echo "Starting Ray head on $MASTER_HOST:$RAY_HEAD_PORT"
  ray start --head \
    --node-ip-address="$MASTER_HOST" \
    --port="$RAY_HEAD_PORT" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="$RAY_DASHBOARD_PORT" \
    --disable-usage-stats

  connected_nodes=0
  for attempt in $(seq 1 60); do
    connected_nodes=$(python -c "import ray; ray.init(address='auto', ignore_reinit_error=True, logging_level='ERROR'); print(sum(1 for node in ray.nodes() if node.get('Alive')))" 2>/dev/null || echo 0)
    echo "Ray nodes connected: $connected_nodes/$NUM_NODES (attempt $attempt/60)"
    if [ "$connected_nodes" -ge "$NUM_NODES" ]; then
      break
    fi
    sleep 10
  done

  if [ "$connected_nodes" -lt "$NUM_NODES" ]; then
    echo "Timed out waiting for all Ray workers to join."
    ray status || true
    exit 1
  fi

  bash "$PROJECT_ROOT/training/verl_training.sh" "{exp_name}" "{CONFIG_NAME}" "{resolved_data_path}" {hydra_args_str}
  train_exit=$?
  ray status || true
  ray stop --force || true
  exit $train_exit
else
  echo "Starting Ray worker on rank $NODE_RANK_VALUE -> $MASTER_HOST:$RAY_HEAD_PORT"
  sleep 15
  ray start --address "${{MASTER_HOST}}:${{RAY_HEAD_PORT}}" --disable-usage-stats --block
fi
"""


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cluster", type=str, default=DEFAULT_CLUSTER)
    parser.add_argument("--resource_group", type=str, default=DEFAULT_RESOURCE_GROUP)
    parser.add_argument("--workspace", type=str, default=DEFAULT_WORKSPACE)
    parser.add_argument("--subscription_id", type=str, default=DEFAULT_SUBSCRIPTION_ID)
    parser.add_argument("--suffix", type=str, default="azure_sdpo_multinode")
    parser.add_argument("--instance_count", type=int, default=DEFAULT_INSTANCE_COUNT)
    parser.add_argument("--gpus_per_node", type=int, default=DEFAULT_GPUS_PER_NODE)
    parser.add_argument("--instance_type", type=str, default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--environment", type=str, default=ENVIRONMENT)
    parser.add_argument("--rollout_tp", type=int, default=ROLLOUT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--teacher_update_rate", type=float, default=TEACHER_UPDATE_RATE)
    parser.add_argument("--clear_hf_cache", action="store_true")
    parser.add_argument(
        "--use_shm",
        action="store_true",
        help="Copy the HF model snapshot into /dev/shm before loading. Disabled by default for large multinode models.",
    )
    args = parser.parse_args()

    if args.instance_count < 2:
        raise ValueError("--instance_count must be at least 2 for this multinode SDPO script.")

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
    if DATA_SUBDIR:
        resolved_data_path = f"{resolved_data_path}/{DATA_SUBDIR.strip('/')}"

    model_name = args.model_path.replace("/", "-")
    exp_name = (
        f"SDPO-mn{args.instance_count}-train{TRAIN_BATCH_SIZE}-alpha{ALPHA}-rollout{ROLLOUT_BATCH_SIZE}"
        f"-lr{LR}-lambda{LAMBDA}-clip_adv_high{CLIP_ADV_HIGH}"
        f"-dross{DONTS_REPROMPT_ON_SELF_SUCCESS}-tur{args.teacher_update_rate}-{model_name}-{args.suffix}"
    )

    hydra_args = [
        f"vars.task={resolved_data_path}",
        f"data.train_batch_size={TRAIN_BATCH_SIZE}",
        f"data.max_prompt_length={MAX_PROMPT_LENGTH}",
        f"data.max_response_length={MAX_RESPONSE_LENGTH}",
        f"data.train_files=[${{inputs.dataset}}/{TRAIN_FILE}]",
        f"data.val_files=[${{inputs.dataset}}/{VAL_FILE}]",
        "trainer.group_name=SDPO-azure-multinode",
        f"trainer.experiment_name={exp_name}",
        "trainer.project_name=sdpo_base",
        f"trainer.default_local_dir=/project/flame/wyu3/SDPO/ttrl_runs/{resolved_data_path}/{exp_name}",
        "trainer.logger=[console,wandb]",
        "trainer.test_freq=5",
        f"trainer.n_gpus_per_node={args.gpus_per_node}",
        f"trainer.nnodes={args.instance_count}",
        f"actor_rollout_ref.rollout.n={ROLLOUT_BATCH_SIZE}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tp}",
        "actor_rollout_ref.rollout.load_format=safetensors",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={ROLLOUT_GPU_MEMORY_UTILIZATION}",
        f"actor_rollout_ref.rollout.max_model_len={ROLLOUT_MAX_MODEL_LEN}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={ROLLOUT_MAX_BATCHED_TOKENS}",
        f"actor_rollout_ref.rollout.max_num_seqs={ROLLOUT_MAX_NUM_SEQS}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        f"actor_rollout_ref.model.path={args.model_path}",
        f"actor_rollout_ref.model.use_shm={args.use_shm}",
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2",
        "+critic.model.override_config.attn_implementation=flash_attention_2",
        f"actor_rollout_ref.actor.optim.lr={LR}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={TRAIN_BATCH_SIZE}",
        "actor_rollout_ref.actor.self_distillation.distillation_topk=100",
        "algorithm.rollout_correction.rollout_is=token",
        f"actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success={DONTS_REPROMPT_ON_SELF_SUCCESS}",
        f"actor_rollout_ref.actor.self_distillation.alpha={ALPHA}",
        f"actor_rollout_ref.actor.self_distillation.teacher_update_rate={args.teacher_update_rate}",
        f"actor_rollout_ref.actor.self_distillation.include_another_solution={INCLUDE_ANOTHER_SOLUTION}",
        f"actor_rollout_ref.actor.self_distillation.include_failure_solution={INCLUDE_FAILURE_SOLUTION}",
        f"actor_rollout_ref.actor.self_distillation.summarize_solutions={SUMMARIZE_SOLUTIONS}",
        f"actor_rollout_ref.actor.self_distillation.summary_k={SUMMARY_K}",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        f"actor_rollout_ref.rollout.val_kwargs.n={VAL_ROLLOUT_BATCH_SIZE}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={ACTOR_PARAM_OFFLOAD}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={ACTOR_OPTIMIZER_OFFLOAD}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={REF_PARAM_OFFLOAD}",
    ]

    hydra_args_str = " ".join([f'"{x}"' for x in hydra_args])
    data_input_uri = (
        f"azureml://subscriptions/{args.subscription_id}"
        f"/resourcegroups/{args.resource_group}"
        f"/workspaces/{args.workspace}"
        f"/datastores/{DATASTORE_NAME}/paths/{resolved_data_path}/"
    )

    command_str = _build_command_str(
        exp_name=exp_name,
        resolved_data_path=resolved_data_path,
        hydra_args_str=hydra_args_str,
        model_path=args.model_path,
        instance_count=args.instance_count,
        gpus_per_node=args.gpus_per_node,
        clear_hf_cache=args.clear_hf_cache,
        use_shm=args.use_shm,
    )

    display_name = f"sdpo-multinode-{timestamp}"

    training_job = command(
        code=PROJECT_ROOT,
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
                    f"/datastores/workspaceblobstore/paths/cheng/sdpo_multinode_models/{timestamp}/"
                ),
                mode="rw_mount",
            ),
        },
        environment=args.environment,
        environment_variables={
            "HF_TOKEN": hf_token,
            "WANDB_API_KEY": wandb_token,
            "_AZUREML_SINGULARITY_JOB_UAI": DEFAULT_UAI,
        },
        compute=vc_cluster,
        resources={
            "instance_count": args.instance_count,
            "instance_type": args.instance_type,
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
        distribution=PyTorchDistribution(process_count_per_instance=1),
        display_name=display_name,
        description="Submit multinode self-distillation SDPO training via Ray + verl_training.sh",
    )

    returned_job = ml_client.jobs.create_or_update(training_job, experiment_name=EXPERIMENT_NAME)
    print("Studio URL:", returned_job.studio_url)
    print("Job name:", returned_job.name)
    print("Experiment:", EXPERIMENT_NAME)
    print("Display name:", display_name)
    print("Model:", args.model_path)
    print("Nodes:", args.instance_count)
    print("GPUs per node:", args.gpus_per_node)
    print("Rollout TP:", args.rollout_tp)
    print("Teacher update rate:", args.teacher_update_rate)
    print("Clear HF cache:", args.clear_hf_cache)
    print("Use /dev/shm model cache:", args.use_shm)
    print("Multinode SDPO job submitted successfully.")
