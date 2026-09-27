# Shared DISM cloud5090 deployment (2026-09-10)

## Launch update

The arch120a retry completed: all5 tests in test_lm_shared.py passed in279.75s,
including the three real soft/mixed/hard CUDA forward/backward/RNG checks.
Full15-layer,FFN1408 preflight also passed:50,227,996 parameters,finite
gradients for all parameters,loss10.88517,peak9.521GiB for one microbatch.
All four extension SASS dumps contain zero CALL instructions. USETMAXREG
counts: embedding forward10,core240,backward360,embedding backward54.

Training has been launched from scratch,seed777,PID4432,output
`/root/autodl-tmp/dism-shared-20260910/run`,console.log in that directory.
Launch arguments: `--architecture hybrid_shared --ffn-hidden 1408`,30k updates,
batch64/micro8,context2048,LR1e-3/WD1e-2/softcap30,hard_prob0→1,offline W&B.
It is detached with its own session and stdin=/dev/null; no systemd user
manager was available in this container. The SSH token tunnel must still
remain alive; detaching training alone does not provide tunnel reconnection.
Preflight weights/optimizer were discarded; this is not a resume of a control.
The preparation/failure notes below are historical.

User authorized training `hybrid_shared`, FFN1408,50,227,996 parameters on the
5090 reached through `root@connect.weste.seetacloud.com:47485`. Task root:
`/root/autodl-tmp/dism-shared-20260910`. Existing local/A100 jobs are untouched.
Deployment uses a task-local venv inheriting the preinstalled Torch2.8.0+cu128,
CUDA12.8,Python3.12; original base environment is not modified.

Installed FLA0.4.2,transformers5.15.0,wandb0.25.0,pyarrow23.0.1 plus build/test
dependencies. Official FlashAttention2.8.3 wheel for torch2.8,cu12,ABI=True,
cp312 was downloaded through the local proxy and copied to the task directory.
Both copies have SHA256
`f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb`.
The task's duplicate remote HTTP download was stopped. Package and CUDA
extension imports passed. This is not yet a CUDA numerical validation result.

Local copy:
`/tmp/dism-flash-wheel.H8GVjB/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`.
No SSH/proxy passwords are written into source,run config or logs.

Token service18475 runs under local user unit`dism-shared-token-20260910`;
state uses the data disk's `dism-lm-runs/shared-token-service-20260910`.
SSH reverse forward18475 uses master`/tmp/dism-cloud-ssh.OV0WxO/control`.
Authentication file is task-local0600. Source identity and first batch match
both controls exactly. Initial throughput8.77MiB/s cached,3.59 fresh effective
batches/s (local tokenization+network), sufficient for initial testing but not
yet a training throughput result. Corpus remains local.

During preparation a local write failure temporarily prevented the service
from starting and truncated `check_remote_lm.py`; the service recovered and
the script was restored from the already-deployed copy before further edits.
The exact transient filesystem cause was not established; no unrelated files
were deleted to make space.

DISM extensions use copied TK/GLX headers,task-local extension/Triton caches,
MAX_JOBS=8,OPT13,tanh_finite. Initial remote command executes
`pytest tests/test_lm_shared.py -q -x`; build/test output is `cuda-smoke.log`,
session18269. Training must only start after CUDA smoke and the full-sized
FFN1408 preflight pass. No successful training launch is claimed here yet.

Initial build failed because Torch2.8 appended automatic generic sm120 targets
alongside our explicit sm120a; ptxas rejected setmaxnreg on the generic target.
The new environment explicitly sets TORCH_CUDA_ARCH_LIST=12.0a (supported by
this Torch version). No kernel math or shared environment source was changed.
Retry output: `cuda-smoke-arch120a.log`; original failed log remains intact.
