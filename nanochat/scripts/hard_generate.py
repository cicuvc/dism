"""Hard-inference generation CLI: prefill then sampled decoding.

Wraps `nanochat.hard_inference` (parallel SAM prefill + native cache + FLA GDN
recurrence). This is an experimental harness, not a supported production path:
the GDN hybrid is marked generation=False and cached hybrid decoding is not
implemented upstream. Run it with flash_dism importable, e.g.

    PYTHONPATH=dism_v3/python:nanochat python -m scripts.hard_generate \
        --checkpoint /path/to/run/base_checkpoints/<tag>

The prefill text comes from a built-in preset or --prompt/--prompt-file; all
sampling knobs are flags. No checkpoint path is hardcoded.
"""
import argparse
import time
from pathlib import Path

import torch

from nanochat.hard_inference import (HardInferenceEngine, decode, default_tokenizer_dir,
                                     load_hard_model)
from nanochat.tokenizer import HuggingFaceTokenizer
from flash_dism.inference import HardDismPrefill

PRESETS = {
    "solar": ("The solar system consists of the Sun and the objects that orbit it. "
              "The largest planet is"),
    "science": ("The history of science is the study of the development of science, "
                "including both the natural and social sciences. Science is a body of "
                "empirical, theoretical, and practical knowledge about the natural world. "
                "The scientific method"),
    "story": ("Once upon a time, a young girl found a mysterious book in the library. "
              "When she opened it,"),
    "code": ("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    return"),
    "chat": ("User: What is the capital of France?\nAssistant:"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="checkpoint directory containing meta_*.json and model_*.pt")
    parser.add_argument("--step", type=int, default=None, help="default: latest step in the directory")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="default: nearest tokenizer/ next to the checkpoint")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", type=str, default=None)
    prompt.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="science",
                        help="built-in prefill text, used when --prompt/--prompt-file are absent")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=200)
    parser.add_argument("-t", "--temperature", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true", help="equivalent to --temperature 0")
    parser.add_argument("--top-k", type=int, default=0, help="0 disables")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--planner", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mma-precision", choices=["bf16", "tf32x3"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stream", action="store_true", help="print tokens as they are produced")
    parser.add_argument("--raw", action="store_true", help="print only the continuation")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text()
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = PRESETS[args.preset]

    tokenizer_dir = args.tokenizer or default_tokenizer_dir(args.checkpoint)
    tokenizer = HuggingFaceTokenizer.from_pretrained(str(tokenizer_dir))
    model, meta, step = load_hard_model(args.checkpoint, step=args.step, device=args.device)
    eos = tokenizer.tokenizer.eos_token_id
    ids = [tokenizer.get_bos_token_id()] + tokenizer.encode(prompt)
    ids = ids[: model.config.sequence_len - args.max_new_tokens]
    prompt_tokens = torch.tensor([ids], device=args.device, dtype=torch.long)
    temperature = 0.0 if args.greedy else args.temperature
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    with HardDismPrefill(workers=args.workers, mma_precision=args.mma_precision) as prefill:
        engine = HardInferenceEngine(model, prefill, capacity=len(ids) + args.max_new_tokens + 8,
                                     planner_backend=args.planner)
        started = time.perf_counter()
        logits = engine(prompt_tokens)[:, -1]
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - started) * 1000

        if not args.raw:
            print("=" * 100)
            print(f"checkpoint {args.checkpoint} step {step}  arch {meta['model_spec']['architecture']}")
            print("hard-inference harness (cached hybrid decoding is not a supported upstream path)")
            print(f"sampling: temperature={temperature} top_k={args.top_k} top_p={args.top_p} "
                  f"repetition_penalty={args.repetition_penalty} seed={args.seed}")
            print(f"prefill {len(ids)} tokens in {prefill_ms:.1f} ms")
            print("-" * 100)

        streamed = {"ids": [], "text": ""}

        def on_token(token_id):
            streamed["ids"].append(token_id)
            now = tokenizer.decode(streamed["ids"])
            if len(now) > len(streamed["text"]):
                print(now[len(streamed["text"]):], end="", flush=True)
                streamed["text"] = now

        started = time.perf_counter()
        generated = decode(engine, logits, max_new_tokens=args.max_new_tokens,
                           temperature=temperature, top_k=args.top_k, top_p=args.top_p,
                           repetition_penalty=args.repetition_penalty, eos_token_id=eos,
                           generator=generator, seen=ids, on_token=on_token if args.stream else None)
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - started

    text = tokenizer.decode(generated)
    if args.raw:
        print(text)
        return
    if not args.stream:
        print(text)
    print("-" * 100)
    print(f"decoded {len(generated)} tokens in {decode_s:.2f} s "
          f"({len(generated)/max(decode_s, 1e-9):.1f} tok/s)")
    print("=" * 100)


if __name__ == "__main__":
    main()
