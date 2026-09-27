"""Scoped remote preflight: token throughput, windowed attention gradients, LM."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-only', action='store_true')
    p.add_argument('--full-attention', action='store_true')
    p.add_argument('--shared-dism', action='store_true', help='Shared DISM with FFN1408')
    p.add_argument('--stream-url', default='http://127.0.0.1:18473')
    p.add_argument('--secret-file', default='auth_token')
    a = p.parse_args()
    if a.full_attention and a.shared_dism:
        p.error('Select one architecture')
    torch.set_num_threads(4)
    from .lm_token_stream import fetch, RemotePackedStream
    secret = Path(a.secret_file).read_text().strip()
    meta = json.loads(fetch(a.stream_url, secret, '/meta')[0])
    begin = time.perf_counter()
    first, _ = fetch(a.stream_url, secret, '/batch/train/0')
    startup = time.perf_counter() - begin
    begin = time.perf_counter()
    for _ in range(32):
        data, _ = fetch(a.stream_url, secret, '/batch/train/0')
        assert data == first
    seconds = time.perf_counter() - begin
    print(json.dumps(dict(event='stream_check', identity=meta['identity'],
                         first_batch_sha256=hashlib.sha256(first).hexdigest(),
                         bytes_per_batch=len(first), first_request_seconds=startup,
                         cached_mib_per_second=32 * len(first) / 2**20 / seconds)), flush=True)
    begin = time.perf_counter()
    for index in range(1, 17):
        fetch(a.stream_url, secret, f'/batch/train/{index}')
    seconds = time.perf_counter() - begin
    print(json.dumps(dict(event='fresh_stream_check', effective_batches=16,
                         seconds=seconds, batches_per_second=16 / seconds)), flush=True)
    stream = RemotePackedStream(a.stream_url, secret, 'train', meta, batch_size=8)
    x, y = stream.next_batch()
    stream.close()
    if a.data_only:
        return
    from flash_attn import flash_attn_func, __version__ as fa_version
    from .lm_model import DecoderLM, LMConfig, matched_swa_config
    torch.manual_seed(41)
    # Covers window edge, nonaligned tail, and all q/k/v gradients.
    for n in (127, 128, 129, 257):
        tensors = [torch.randn(1, n, 4, 64, device='cuda', dtype=torch.bfloat16,
                               requires_grad=True) for _ in range(3)]
        refs = [t.detach().float().requires_grad_() for t in tensors]
        q, k, v = [t.transpose(1, 2) for t in refs]
        i = torch.arange(n, device='cuda')
        mask = i[:, None] >= i[None, :]
        if not a.full_attention:
            mask = mask & (i[:, None] - i[None, :] < 128)
        expected = ((q @ k.transpose(-1, -2) / 8).masked_fill(~mask, -torch.inf).softmax(-1) @ v).transpose(1, 2)
        actual = flash_attn_func(*tensors, causal=True,
                                window_size=(-1, -1) if a.full_attention else (127, 0))
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream.float())
        torch.testing.assert_close(actual.float(), expected, atol=.008, rtol=.03)
        for t, ref in zip(tensors, refs):
            torch.testing.assert_close(t.grad.float(), ref.grad, atol=.015, rtol=.04)
    # All production-sized vocabulary/cap mean CE checks run separately in pytest.
    config = matched_swa_config(LMConfig(softcap=30.))
    if a.full_attention:
        config.architecture, config.window = 'full_attention', -1
    if a.shared_dism:
        config = LMConfig(architecture='hybrid_shared', ffn_hidden=1408, softcap=30.)
    model = DecoderLM(config).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, fused=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(x.cuda(), y.cuda(), 0., None)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    opt.step()
    print(json.dumps(dict(event='gpu_preflight_pass', gpu=torch.cuda.get_device_name(),
                         torch=torch.__version__, flash_attn=fa_version,
                         parameters=sum(p.numel() for p in model.parameters()),
                         loss=loss.item(), peak_gib=torch.cuda.max_memory_allocated() / 2**30)), flush=True)


if __name__ == '__main__':
    main()
