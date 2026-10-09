"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20

To train from a local recursive Parquet dataset with 32-token document alignment:
python -m scripts.base_train --data-dir /path/to/fwedu-100B --seq-align 32
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
import random
import numpy as np
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import Linear
from nanochat.models import (
    ModelSpec,
    PretrainOptimizerConfig,
    build_model,
    get_scaling_reference_spec,
    migrate_model_state_dict,
    model_spec_from_metadata,
    optimizer_schema,
    optimizer_schema_fingerprint,
)
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import (
    HuggingFaceTokenizer,
    describe_tokenizer,
    get_tokenizer,
    get_token_bytes,
)
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, load_rank_training_state
from nanochat.loss_eval import evaluate_bpb, evaluate_hard_soft
from nanochat.metrics import (
    register_residual_rms_probe,
    remove_probe,
    snapshot_parameters,
    update_rms_max_per_group,
    probe_qk_vocab_usage,
)
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3, HAS_FA2
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
parser.add_argument("--dual-path-loss", action="store_true", help="Separate all-soft/all-hard losses, paired direction, sequential backward")
# Logging
parser.add_argument("--wandb-project", type=str, default="nanochat")
parser.add_argument("--dual-eval-valid-tokens", type=int, default=0)
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--model-spec", type=str, default=None, help="JSON ModelSpec for a registered architecture")
parser.add_argument("--hf-tokenizer", type=str, default=None, help="Hugging Face tokenizer name/path instead of nanochat tokenizer")
parser.add_argument("--data-dir", type=str, default=None, help="recursive directory of parquet shards (default: nanochat base data)")
parser.add_argument("--seq-align", type=int, default=32, help="align every packed document start to this token multiple")
parser.add_argument("--loader-workers", type=int, default=0, help="ordered tokenizer workers with checkpoint-safe CPU prefetch")
parser.add_argument("--data-shuffle", action=argparse.BooleanOptionalAction, default=None,
                    help="shuffle training row groups/documents (default: on for new runs; checkpoint-defined on resume)")
parser.add_argument("--shuffle-seed", type=int, default=None,
                    help="deterministic data shuffle seed (default: 1337 for new runs; checkpoint-defined on resume)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-device-batch-size", type=int, default=None, help="Validation microbatch; defaults to training microbatch")
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Diagnostics
parser.add_argument("--log-every", type=int, default=1, help="log training metrics to wandb every N steps")
parser.add_argument("--log-update-rms", action=argparse.BooleanOptionalAction, default=True, help="log per-layer max optimizer update RMS")
parser.add_argument("--log-hidden-rms", action=argparse.BooleanOptionalAction, default=True, help="log per-layer max residual-stream RMS")
parser.add_argument("--qk-hist-every", type=int, default=500, help="log DISM q/k codebook selection stats every N steps (-1 = disable)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
if args.eval_device_batch_size is None:
    args.eval_device_batch_size = args.device_batch_size
if args.eval_device_batch_size <= 0:
    raise ValueError("eval-device-batch-size must be positive")
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=args.wandb_project, name=args.run, config=user_config)
if not use_dummy_wandb:
    wandb_run.define_metric("train/*", step_metric="step")
    wandb_run.define_metric("val/*", step_metric="step")

# Flash Attention status
from nanochat.flash_attention import USE_FA3, USE_FA2
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
elif USE_FA2:
    print0("✓ Using Flash Attention 2 (local package).")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0(f"WARNING: Flash Attention unavailable (FA3={HAS_FA3}, FA2={HAS_FA2}), using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
if args.hf_tokenizer:
    tokenizer = HuggingFaceTokenizer.from_pretrained(args.hf_tokenizer)
    token_bytes = tokenizer.get_token_bytes(device=device)
else:
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
tokenizer_spec = describe_tokenizer(tokenizer)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_spec(depth):
    """Build the current nanochat architecture spec for a requested depth."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    return ModelSpec(
        architecture="nanochat_v1",
        architecture_version=1,
        config={
            "sequence_len": args.max_seq_len,
            "vocab_size": vocab_size,
            "n_layer": depth,
            "n_head": num_heads,
            "n_kv_head": num_heads,
            "n_embd": model_dim,
            "window_pattern": args.window_pattern,
        },
    )


def load_model_spec_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return ModelSpec.from_dict(json.load(f))


# Resolve checkpoint identity before allocating the model. Resume always rebuilds
# from the saved ModelSpec rather than silently applying the current CLI architecture.
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
model_data = optimizer_data = meta_data = None
requested_model_spec = load_model_spec_file(args.model_spec) if args.model_spec else None
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, args.resume_from_step, device,
        load_optimizer=True, rank=ddp_rank,
    )
    model_spec = model_spec_from_metadata(meta_data)
    saved_tokenizer_spec = meta_data.get("tokenizer_spec")
    if saved_tokenizer_spec is not None and saved_tokenizer_spec != tokenizer_spec:
        raise ValueError("Selected tokenizer does not match the checkpoint tokenizer")
    if requested_model_spec is not None and requested_model_spec.fingerprint() != model_spec.fingerprint():
        raise ValueError("--model-spec does not match the checkpoint; architecture changes are not allowed during resume")
else:
    model_spec = requested_model_spec or build_model_spec(args.depth)

if model_spec.config.get("vocab_size") != vocab_size:
    raise ValueError(
        f"ModelSpec vocab_size {model_spec.config.get('vocab_size')} does not match tokenizer {vocab_size}"
    )
if "sequence_len" not in model_spec.config:
    raise ValueError("ModelSpec config must define sequence_len for pretraining")
if resuming or args.model_spec:
    args.max_seq_len = model_spec.config["sequence_len"]
    user_config["max_seq_len"] = args.max_seq_len

# Build the model, move to device, init the weights
with torch.device("meta"):
    model = build_model(model_spec) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
if resuming and getattr(model, 'exact_optimizer_hyperparameters', False):
    for key in ('num_iterations', 'warmup_steps', 'warmdown_ratio', 'final_lr_frac',
                'total_batch_size', 'device_batch_size', 'eval_device_batch_size', 'matrix_lr', 'weight_decay', 'seq_align', 'data_dir', 'dual_path_loss'):
        if meta_data['user_config'].get(key, False if key == 'dual_path_loss' else None) != user_config.get(key):
            raise ValueError(f'DISM exact resume requires unchanged {key}')
model_config_kwargs = dict(model_spec.config)
if not model.capabilities.varlen:
    raise ValueError(
        f"Architecture {model_spec.architecture!r} does not support packed varlen pretraining"
    )
print0(
    f"Model architecture: {model_spec.architecture}@{model_spec.architecture_version}\n"
    f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}"
)
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
rank_training_state = None
if resuming:
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_data = migrate_model_state_dict(model_spec, model_data, model.config)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy
    rank_training_state = load_rank_training_state(
        checkpoint_dir, args.resume_from_step, rank=ddp_rank
    )
    if rank_training_state is None:
        if meta_data.get("exact_resume_version") is not None:
            raise FileNotFoundError(
                f"Checkpoint step {args.resume_from_step} declares exact resume state "
                f"but rank {ddp_rank} state is missing"
            )
        print0("WARNING: checkpoint has no exact rank training state; using legacy approximate dataloader resume")
    else:
        assert rank_training_state.get("version") == meta_data.get("exact_resume_version"), \
            "Rank training state version does not match checkpoint metadata"
        saved_model_fingerprint = rank_training_state.get("model_fingerprint")
        if meta_data.get("model_fingerprint") is not None:
            assert saved_model_fingerprint == model_spec.fingerprint(), \
                "Rank training state was captured for a different model architecture/config"
        if meta_data.get("optimizer_schema_fingerprint") is not None:
            assert rank_training_state.get("optimizer_schema_fingerprint") == meta_data["optimizer_schema_fingerprint"], \
                "Rank training state optimizer schema does not match checkpoint metadata"
        print0("Loaded exact dataloader, prefetched batch, RNG, and scaler state")

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
if args.log_hidden_rms:
    hidden_rms_handles, hidden_rms_buffer = register_residual_rms_probe(orig_model)
else:
    hidden_rms_handles, hidden_rms_buffer = [], None
if getattr(orig_model, 'dynamic_compile', False):
    from torch.fx.experimental import _config as shape_config
    shape_config.use_duck_shape = False
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._inductor.config.compile_threads = 1
model = torch.compile(model, dynamic=getattr(orig_model, 'dynamic_compile', False))

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    return m.scaling_parameter_count()
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Each architecture defines the reference model used to transfer tuned scaling
# laws. nanochat_v1 preserves the historical d12 construction.
reference_spec = get_scaling_reference_spec(model_spec)
with torch.device("meta"):
    d12_ref = build_model(reference_spec)
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0 and not getattr(orig_model, 'exact_optimizer_hyperparameters', False):
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if getattr(orig_model, 'exact_optimizer_hyperparameters', False):
    batch_lr_scale = 1.0
    weight_decay_scaled = args.weight_decay
if weight_decay_scaled != args.weight_decay:
    print0(
        f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} "
        f"for {model_spec.architecture} depth {model_config.n_layer}"
    )

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_pretraining_optimizer(
    PretrainOptimizerConfig(
        unembedding_lr=args.unembedding_lr * batch_lr_scale,
        embedding_lr=args.embedding_lr * batch_lr_scale,
        scalar_lr=args.scalar_lr * batch_lr_scale,
        matrix_lr=args.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
    )
)
current_optimizer_schema = optimizer_schema(orig_model, optimizer)
current_optimizer_fingerprint = optimizer_schema_fingerprint(current_optimizer_schema)

if resuming:
    saved_optimizer_fingerprint = meta_data.get("optimizer_schema_fingerprint")
    if saved_optimizer_fingerprint is not None:
        assert saved_optimizer_fingerprint == current_optimizer_fingerprint, \
            "Optimizer parameter grouping does not match the checkpoint"
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")
    if rank_training_state is not None and rank_training_state.get("grad_scaler") is not None:
        scaler.load_state_dict(rank_training_state["grad_scaler"])

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None
if rank_training_state is not None:
    assert rank_training_state.get("version") == 1, \
        f"Unsupported rank training state version: {rank_training_state.get('version')}"
    dataloader_resume_state_dict = rank_training_state["dataloader"]
elif resuming:
    dataloader_resume_state_dict = meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
    resume_state_dict=dataloader_resume_state_dict, seq_align=args.seq_align, data_dir=args.data_dir,
    num_workers=args.loader_workers,
    shuffle=args.data_shuffle, shuffle_seed=args.shuffle_seed,
    aligned_segment_ends=getattr(orig_model, 'aligned_segment_ends', False),
)
base_train_loader = getattr(train_loader, "loader", train_loader)
data_order_state = base_train_loader.batches.state_dict()
user_config["data_shuffle"] = data_order_state["shuffle"]
user_config["shuffle_seed"] = data_order_state["shuffle_seed"]
print0(
    f"Training data order: shuffle={data_order_state['shuffle']} "
    f"seed={data_order_state['shuffle_seed']} "
    f"algorithm={data_order_state['shuffle_algorithm']}"
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.eval_device_batch_size, args.max_seq_len, split="val", device=device,
    seq_align=args.seq_align, data_dir=args.data_dir,
    shuffle=False, shuffle_seed=args.shuffle_seed,
    aligned_segment_ends=getattr(orig_model, 'aligned_segment_ends', False),
)
if rank_training_state is not None:
    saved_batch = rank_training_state["prefetched_batch"]
    x, y, cu_seqlens, segment_ids = (
        saved_batch[name].to(device=device)
        for name in ("inputs", "targets", "cu_seqlens", "segment_ids")
    )
    dataloader_state_dict = train_loader.progress_state()
    random.setstate(rank_training_state["python_rng"])
    np.random.set_state(rank_training_state["numpy_rng"])
    torch.set_rng_state(rank_training_state["torch_rng"])
    if device_type == "cuda" and rank_training_state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(rank_training_state["cuda_rng"], device=device)
else:
    x, y, cu_seqlens, segment_ids, dataloader_state_dict = next(train_loader) # prefetch first batch

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
if not use_dummy_wandb:
    wandb_run.config.update({"resolved_model_spec": model_spec.to_dict(),
                            "parameter_count": num_params,
                            "padded_training_tokens": total_tokens})
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")


def capture_rank_training_state():
    """Capture everything rank-local needed to consume the exact next batch."""
    return {
        "version": 1,
        "model_fingerprint": model_spec.fingerprint(),
        "optimizer_schema_fingerprint": current_optimizer_fingerprint,
        # The loader state is after the prefetched batch below was generated.
        "dataloader": train_loader.state_dict(),
        "prefetched_batch": {
            "inputs": x.detach().to(device="cpu", copy=True),
            "targets": y.detach().to(device="cpu", copy=True),
            "cu_seqlens": cu_seqlens.detach().to(device="cpu", copy=True),
            "segment_ids": segment_ids.detach().to(device="cpu", copy=True),
        },
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device_type == "cuda" else None,
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
    }


# Go!
def log_cuda_memory(phase, step):
    if device_type != "cuda" or not master_process:
        return
    stats = torch.cuda.memory_stats(device)
    free, total = torch.cuda.mem_get_info(device)
    record = dict(step=step, phase=phase,
        allocated_mib=torch.cuda.memory_allocated(device)/2**20,
        reserved_mib=torch.cuda.memory_reserved(device)/2**20,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
        active_mib=stats.get("active_bytes.all.current", 0)/2**20,
        inactive_split_mib=stats.get("inactive_split_bytes.all.current", 0)/2**20,
        allocation_count=stats.get("allocation.all.current", 0),
        alloc_retries=stats.get("num_alloc_retries", 0),
        oom_count=stats.get("num_ooms", 0),
        device_used_mib=(total-free)/2**20)
    if phase == "after_train_step" and (step in (100, 500) or
            (record["reserved_mib"] > 80*1024 and not getattr(log_cuda_memory, "high_snapshot", False))):
        import pickle
        with open(os.path.join(base_dir, f"cuda_memory_snapshot_{step:06d}.pickle"), "wb") as handle:
            pickle.dump(torch.cuda.memory_snapshot(), handle)
        if record["reserved_mib"] > 80*1024:
            log_cuda_memory.high_snapshot = True
    print0("CUDA_MEMORY " + json.dumps(record))
    with open(os.path.join(base_dir, "cuda_memory.jsonl"), "a") as handle:
        handle.write(json.dumps(record) + "\n")


while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step
    # A checkpoint is captured after this step's eval/sample hooks. Re-running
    # those hooks on the resume entry could consume RNG and break exactness.
    resume_entry = resuming and step == args.resume_from_step

    # once in a while: evaluate the val bpb (all ranks participate)
    if not resume_entry and args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        log_cuda_memory("before_validation", step)
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.eval_device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            # FlashAttention's no-grad multi-layer Dynamo trace currently hits
            # a compiler assertion. Keep training compiled; validation is eager.
            eval_model = orig_model if getattr(orig_model, 'dynamic_compile', False) else model
            dual_metrics = None
            if args.dual_eval_valid_tokens:
                dual_metrics = evaluate_hard_soft(eval_model, val_loader, args.dual_eval_valid_tokens, token_bytes)
                val_bpb = dual_metrics['hard_bpb']
                print0("DUAL_VALIDATION " + json.dumps(dict(step=step, **dual_metrics)))
                with open(os.path.join(base_dir, "dual_validation.jsonl"), "a") as handle:
                    handle.write(json.dumps(dict(step=step, **dual_metrics)) + "\n")
            else:
                val_bpb = evaluate_bpb(eval_model, val_loader, eval_steps, token_bytes)
        log_cuda_memory("after_validation", step)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
            **({"val/"+key: value for key, value in dual_metrics.items()} if dual_metrics else {}),
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if not resume_entry and args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if not resume_entry and args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        from nanochat.checkpoint_retention import check_space, prune
        check_space(checkpoint_dir)
        exact_rank_state = capture_rank_training_state()
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "model_spec": model_spec.to_dict(),
                "model_fingerprint": model_spec.fingerprint(),
                "tokenizer_spec": tokenizer_spec,
                "optimizer_schema": current_optimizer_schema,
                "optimizer_schema_fingerprint": current_optimizer_fingerprint,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                # JSON-safe cursor for degraded legacy fallback. Exact resume uses
                # the per-rank binary state, which also contains the token buffer.
                "dataloader_state_dict": train_loader.fallback_state_dict(),
                "exact_resume_version": exact_rank_state["version"],
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
            rank_training_state=exact_rank_state,
        )

        if master_process: prune(checkpoint_dir, keep=2)

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    if hasattr(orig_model, 'set_training_step'):
        orig_model.set_training_step(step)
    for micro_step in range(grad_accum_steps):
        if args.dual_path_loss:
            from nanochat.dual_loss import dual_loss_backward
            alpha = float(orig_model.hard_probability.item())
            train_loss, path_losses = dual_loss_backward(model, orig_model, x, y, cu_seqlens,
                segment_ids, alpha, grad_accum_steps, scaler)
        else:
            loss = model(x, y, cu_seqlens=cu_seqlens, segment_ids=segment_ids)
            train_loss = loss.detach() # for logging
            loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        x, y, cu_seqlens, segment_ids, dataloader_state_dict = next(train_loader) # prefetch while GPU is busy
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
    grad_norm = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), max_norm=float("inf"))
    if args.dual_path_loss and step % 100 == 0:
        group_sq = {}
        for name,p in orig_model.named_parameters():
            if p.grad is None: continue
            role = ('codebook' if name.endswith(('q_vocab','k_vocab')) else
                    'shared_qk' if name.endswith(('q_conv.weight','k_conv.weight')) else
                    'shortconv' if name.endswith('conv_weight') else 'other')
            group_sq[role] = group_sq.get(role, 0) + p.grad.detach().float().square().sum()
        wandb_run.log({'step':step, **{'train/grad_norm_'+k:v.sqrt().item() for k,v in group_sq.items()}})
    grad_norm_f = grad_norm.item()
    if not math.isfinite(grad_norm_f):
        raise FloatingPointError(f"Non-finite gradient norm at step {step}: {grad_norm_f}")
    log_this_step = step % args.log_every == 0
    update_snapshot = snapshot_parameters(orig_model) if (args.log_update_rms and log_this_step) else None
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    update_rms = update_rms_max_per_group(orig_model, update_snapshot) if update_snapshot is not None else {}
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    if step < 5 or step % 100 == 0:
        log_cuda_memory("after_train_step", step)
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | grad norm: {grad_norm_f:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if log_this_step:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/loss_raw": train_loss_f,
            "train/grad_norm": grad_norm_f,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        if hasattr(orig_model, "hard_probability"):
            log_data["train/hard_probability"] = orig_model.hard_probability.item()
        if args.dual_path_loss:
            log_data['train/hard_loss_weight'] = alpha
            for key,value in path_losses.items():
                log_data['train/'+key+'_path_loss'] = value.item()
        for key, value in update_rms.items():
            log_data[f"train/update_rms_max/{key}"] = value
        if hidden_rms_buffer is not None:
            for layer_index, value in enumerate(hidden_rms_buffer.tolist()):
                log_data[f"train/hidden_rms_max/layer{layer_index:02d}"] = value
        if args.qk_hist_every > 0 and step % args.qk_hist_every == 0:
            qk_stats = probe_qk_vocab_usage(orig_model, x, cu_seqlens, args.max_seq_len)
            for layer_index, tags in qk_stats.items():
                for tag, stats in tags.items():
                    prefix = f"qk/{tag}"
                    log_data[f"{prefix}_entropy/layer{layer_index:02d}"] = stats["entropy"]
                    log_data[f"{prefix}_max_share/layer{layer_index:02d}"] = stats["max_share"]
                    log_data[f"{prefix}_coverage/layer{layer_index:02d}"] = stats["coverage"]
                    log_data[f"{prefix}_coverage_pooled/layer{layer_index:02d}"] = stats.get("coverage_pooled", stats["coverage"])
                    for head_index, head_coverage in enumerate(stats.get("coverage_per_head", ())):
                        log_data[f"{prefix}_coverage_head{head_index}/layer{layer_index:02d}"] = head_coverage
                    counts = np.asarray(stats["counts"], dtype=np.float64)
                    log_data[f"{prefix}_hist/layer{layer_index:02d}"] = wandb.Histogram(
                        np_histogram=(counts, np.arange(counts.size + 1)))
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# cleanup
if hidden_rms_handles:
    remove_probe(hidden_rms_handles)
if hasattr(train_loader, 'close'):
    train_loader.close()
wandb_run.finish() # wandb run finish
compute_cleanup()
