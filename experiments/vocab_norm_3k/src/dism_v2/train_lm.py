"""Single-GPU streaming FineWeb-Edu training; run with conda blkw."""
import argparse
from dataclasses import asdict
import json
import hashlib
import math
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import time

# Set before importing either Hugging Face or the CUDA extension configuration.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("DISM_TILE_LSE", "tanh_finite")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch
from transformers import GPT2TokenizerFast
import wandb

from .lm_data import PackedStream, split_files
from .lm_model import DecoderLM, LMConfig, parameter_groups, hard_probability, learning_rate, matched_swa_config

DATA_ROOT = "/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/finewebedu"
RUN_ROOT = "/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs"


def load_tokenizer(source):
    local = Path(source) / "tokenizer.json"
    if local.is_file():
        tok = GPT2TokenizerFast(tokenizer_file=str(local), bos_token="<|endoftext|>",
                                eos_token="<|endoftext|>", unk_token="<|endoftext|>")
    else:
        tok = GPT2TokenizerFast.from_pretrained(source)
    if len(tok) != 50257 or tok.eos_token_id != 50256:
        raise ValueError("Expected unmodified GPT-2 vocabulary (50257, EOS50256)")
    tok.model_max_length = 10**12  # Documents are packed, not silently truncated.
    return tok


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DATA_ROOT)
    p.add_argument("--tokenizer", default=RUN_ROOT + "/tokenizer-gpt2")
    p.add_argument("--stream-url", help="Authenticated loopback service through SSH")
    p.add_argument("--stream-secret-file", type=Path)
    p.add_argument("--output", default=RUN_ROOT + "/dism-swa-50m-" + time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--batch", type=int, default=64, help="Effective sequences per optimizer update")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--softcap", type=float, default=30.)
    p.add_argument("--architecture", choices=("hybrid", "hybrid_shared", "swa_only", "full_attention"), default="hybrid")
    p.add_argument("--ffn-hidden", type=int, help="SWA-only default matches the hybrid parameter budget")
    p.add_argument('--dism-tie-qk-vocab', action='store_true', help='One shared Q/K codebook per group')
    p.add_argument('--dism-vocab-groups', type=int, help='Divisor of DISM heads; default one group per head')
    p.add_argument('--dism-activation', choices=('baseline', 'vocab_silu', 'no_qk_silu'), default='baseline')
    p.add_argument('--dism-output-gla', action='store_true')
    p.add_argument('--dism-vocab-norm', action='store_true')
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--grad-clip", type=float, default=1.)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=100, help="Same effective batch as training")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after", type=int, help="Smoke/interruption test: stop at this update; does not alter annealing")
    p.add_argument("--wandb-project", default="dism-finewebedu")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    if min(a.steps, a.micro_batch, a.batch, a.eval_every, a.eval_batches, a.save_every, a.log_every, a.threads) <= 0:
        p.error("Step, batch, interval and thread counts must be positive")
    if a.batch % a.micro_batch:
        p.error("--batch must be divisible by --micro-batch")
    if not 0 <= a.warmup < a.steps or a.lr <= 0 or a.grad_clip <= 0:
        p.error("Invalid warmup, learning rate or gradient clipping")
    if not math.isfinite(a.softcap) or a.softcap <= 0:
        p.error("--softcap must be positive and finite")
    if a.ffn_hidden is not None and a.ffn_hidden <= 0:
        p.error("--ffn-hidden must be positive")
    return a


def main():
    a = arguments()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if a.architecture in ('hybrid', 'hybrid_shared') and torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("CUDA DISM currently requires sm120; SWA-only also supports A100")
    output = Path(a.output)
    if output.exists() and (output / "config.json").exists() and not a.resume:
        raise FileExistsError("Run exists: choose a new output directory or explicitly --resume")
    output.mkdir(parents=True, exist_ok=True)
    stream_meta = None
    if a.stream_url:
        from .lm_token_stream import RemotePackedStream, fetch
        if a.stream_secret_file is None:
            raise ValueError("--stream-secret-file required")
        secret = a.stream_secret_file.read_text().strip()
        stream_meta = json.loads(fetch(a.stream_url, secret, '/meta')[0])
        for key, expected in dict(batch=a.batch, context=2048, seed=a.seed,
                                  steps=a.steps, eval_batches=a.eval_batches).items():
            if stream_meta[key] != expected:
                raise ValueError(f"Token service configuration mismatch: {key}")
        a.data, a.tokenizer = stream_meta['data'], stream_meta['tokenizer']
        stream = RemotePackedStream(a.stream_url, secret, 'train', stream_meta, a.micro_batch)
    else:
        tok = load_tokenizer(a.tokenizer)
        train_files, val_files = split_files(a.data)
        stream = PackedStream(train_files, tok, batch_size=a.micro_batch, seed=a.seed, repeat=True)
    cfg = LMConfig(softcap=a.softcap)
    if a.architecture == 'hybrid_shared':
        cfg.architecture = a.architecture
    if a.architecture in ("swa_only", "full_attention"):
        cfg = matched_swa_config(cfg)
        if a.architecture == 'full_attention':
            cfg.architecture, cfg.window = 'full_attention', -1
    if a.ffn_hidden is not None:
        cfg.ffn_hidden = a.ffn_hidden
    cfg.dism_tie_qk_vocab = a.dism_tie_qk_vocab
    cfg.dism_vocab_groups = a.dism_vocab_groups
    cfg.dism_activation = a.dism_activation
    cfg.dism_output_gla = a.dism_output_gla
    cfg.dism_vocab_norm = a.dism_vocab_norm
    model = DecoderLM(cfg).cuda()
    assert model.embedding.weight is not model.lm_head.weight
    optimizer = torch.optim.AdamW(parameter_groups(model, a.weight_decay), lr=a.lr,
                                  betas=(.9, .95), eps=1e-8, fused=True)
    generator = torch.Generator(device="cuda").manual_seed(a.seed + 1)
    val_generator = torch.Generator(device="cuda")
    accum = a.batch // a.micro_batch
    if cfg.architecture in ('hybrid', 'hybrid_shared'):
        from .backward import DEFAULT_OPTIMIZATION
        backward_opt = os.environ.get('DISM_BWD_OPT', DEFAULT_OPTIMIZATION)
    else:
        backward_opt = 'not applicable'
    git_commit = os.environ.get('DISM_SOURCE_COMMIT')
    if not git_commit:
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], text=True, capture_output=True)
        git_commit = git.stdout.strip() if git.returncode == 0 else 'source snapshot'
    config = {**vars(a), "resume": str(a.resume) if a.resume else None, "model": asdict(cfg),
              "stream_secret_file": str(a.stream_secret_file) if a.stream_secret_file else None,
              "stream_metadata": stream_meta,
              "parameters": sum(p.numel() for p in model.parameters()), "gradient_accumulation": accum,
              "tokens_per_update": a.batch * cfg.context, "tile_lse": os.environ["DISM_TILE_LSE"],
              "backward_opt": backward_opt,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "process_id": os.getpid(),
              "git_commit": git_commit,
              "untied": True, "hard_schedule": (f"linear: update0=0, update{a.steps-1}=1"
                  if cfg.architecture in ('hybrid', 'hybrid_shared') else "not applicable: no DISM"),
              "validation": f"fixed held-out packed prefix,{a.eval_batches} effective batches,current hard_prob,reset eval RNG",
              "decay_parameters": sum(p.numel() for p in optimizer.param_groups[0]["params"]),
              "no_decay_parameters": sum(p.numel() for p in optimizer.param_groups[1]["params"])}
    snapshot = output / "sources"
    snapshot.mkdir(exist_ok=True)
    source_hashes = {}
    source_names = ["train_lm.py", "lm_model.py", "lm_data.py", "softcap_cross_entropy.py", "lm_token_stream.py"]
    if cfg.architecture in ('hybrid', 'hybrid_shared'):
        source_names += ['autograd.py', 'backward.py']
    for name in source_names:
        source = Path(__file__).parent / name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        source_hashes[name] = digest
        dest = snapshot / (digest[:12] + "_" + name)
        if not dest.exists():
            shutil.copyfile(source, dest)
    config["source_sha256"] = source_hashes
    start, best, run_id = 0, float("inf"), None
    if a.resume:
        saved = torch.load(a.resume, map_location="cpu", weights_only=False)  # Only own trusted checkpoints.
        old = saved["config"]
        # Existing running hybrid checkpoints predate the architecture field.
        old["model"] = asdict(LMConfig(**old["model"]))
        for key in ("model", "batch", "micro_batch", "steps", "warmup", "lr", "weight_decay", "seed", "tile_lse", "backward_opt"):
            if old[key] != config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        stream.load_state_dict(saved["stream"])
        generator.set_state(saved["dism_rng"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        start, best, run_id = saved["step"], saved["best_val"], saved["wandb_id"]
        del saved
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=output.name,
                     id=run_id, resume="must" if run_id and a.wandb_mode == "online" else None,
                     config=config, dir=str(output), mode=a.wandb_mode)
    run.define_metric("optimizer_step")
    run.define_metric("*", step_metric="optimizer_step")
    if run.url:
        (output / "wandb_url.txt").write_text(run.url + "\n")
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
        print(f"Signal {signum}: finish current update and checkpoint", flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def log(values, step):
        row = {"step": step, **values}
        print(json.dumps(row), flush=True)
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        # Train and validation can share an optimizer step without W&B dropping
        # the second record as an already-committed internal history step.
        run.log({"optimizer_step": step, **values})

    def save(step, name="latest.pt"):
        state = dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     stream=stream.state_dict(), dism_rng=generator.get_state(),
                     cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                     python_rng=random.getstate(), step=step, best_val=best,
                     wandb_id=run.id if a.wandb_mode != "disabled" else None)
        temporary = output / (name + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, output / name)

    @torch.no_grad()
    def validate(hard_prob):
        model.eval()
        valid = (RemotePackedStream(a.stream_url, secret, 'validation', stream_meta, a.micro_batch)
                 if a.stream_url else PackedStream(val_files, tok, batch_size=a.micro_batch, seed=a.seed, repeat=False))
        val_generator.manual_seed(a.seed + 2)
        total = torch.zeros((), device="cuda")
        try:
            for _ in range(a.eval_batches * accum):
                x, y = valid.next_batch()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x.cuda(), y.cuda(), hard_prob, val_generator)
                total += loss.float()
            value = (total / (a.eval_batches * accum)).item()
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite validation loss")
            return value
        finally:
            if a.stream_url:
                valid.close()
            model.train()

    print(json.dumps({"event": "initialized", **config, "wandb_url": run.url}), flush=True)
    model.train()
    completed = start
    limit = min(a.steps, a.stop_after or a.steps)
    try:
        for step in range(start, limit):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            lr = learning_rate(step, a.steps, a.warmup, a.lr)
            probability = hard_probability(step, a.steps) if cfg.architecture in ('hybrid', 'hybrid_shared') else 0.
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device="cuda")
            for _ in range(accum):
                x, y = stream.next_batch()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x.cuda(), y.cuda(), probability, generator)
                total += loss.detach().float() / accum
                (loss / accum).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip, error_if_nonfinite=True)
            if not torch.isfinite(total).item():
                raise FloatingPointError("Nonfinite training loss")
            optimizer.step()
            completed = step + 1
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            if completed % a.log_every == 0 or completed == start + 1:
                taus = [torch.nn.functional.softplus(b.dism.log_sel_tau.detach())
                        for b in model.blocks if b.dism is not None]
                tau_metrics = {}
                if taus:
                    tau = torch.cat(taus)
                    tau_metrics = {"train/tau_min": tau.min().item(), "train/tau_max": tau.max().item()}
                log({"train/loss": total.item(), "train/perplexity": math.exp(min(total.item(), 80)),
                     "train/grad_norm": norm.item(), "train/lr": lr, "train/hard_prob": probability,
                     "train/step_seconds": seconds, "train/tokens_per_second": a.batch * cfg.context / seconds,
                     "train/tokens_seen": completed * a.batch * cfg.context, "train/data_epoch": stream.epoch,
                     **tau_metrics,
                     "memory/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}, completed)
            if completed % a.eval_every == 0:
                val_begin = time.perf_counter()
                val_loss = validate(probability)
                log({"val/loss": val_loss, "val/perplexity": math.exp(min(val_loss, 80)),
                     "val/hard_prob": probability, "val/effective_batches": a.eval_batches,
                     "val/seconds": time.perf_counter() - val_begin}, completed)
                if val_loss < best:
                    best = val_loss
                    save(completed, "best.pt")
            if completed % a.save_every == 0:
                save(completed)
            if stopping:
                break
        save(completed)
        log({"status/completed_steps": completed, "status/finished": completed == a.steps}, completed)
    except BaseException:
        # Never label partially accumulated or nonfinite state as resumable.
        # The previous atomic latest.pt remains the recovery point.
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
    finally:
        if a.stream_url:
            stream.close()


if __name__ == "__main__":
    main()
