"""Paired gradient direction/scale audit; never changes production precision.

IID core: independent FP64 explicit oracle, identical BF16 inputs and dO.
Trajectory: real V512 embedding, FP32 Torch oracle, same parameters and dO on
every step; only CUDA gradients update parameters. This is a synthetic local
fit, not an LM training-quality experiment. Correlated steps are not IID seeds.
JSONL raw observations go to stdout; progress goes to stderr.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import cu_flash_dism as cu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests'))
from test_voc_backward import torch_voc
from flash_dism.forward import forward_core
from flash_dism.backward import backward_core
from flash_dism.voc import voc_dism
from flash_dism.reference.dism_v3_ref import dism_ref_backward


def metrics(actual, oracle):
    x, y = actual.detach().double().flatten(), oracle.detach().double().flatten()
    xx, yy, xy = torch.stack((x.dot(x), y.dot(y), x.dot(y))).tolist()
    assert all(math.isfinite(t) for t in (xx, yy, xy)), 'nonfinite gradient'
    result = dict(xx=xx, yy=yy, xy=xy, oracle_norm=math.sqrt(yy), actual_norm=math.sqrt(xx))
    # Zero cases stay visible, and do not contribute fabricated cosine=1.
    if yy > 1e-24:
        result.update(norm_ratio=math.sqrt(xx / yy),
                      gain_bias=xy / yy - 1.,
                      relative_l2=math.sqrt(max(0., xx + yy - 2*xy) / yy),
                      cosine=xy / math.sqrt(xx*yy) if xx else 0.)
    else:
        result.update(norm_ratio=None, gain_bias=None, relative_l2=None, cosine=None)
    return result


def emit(actual, oracle, **identity):
    for name in actual:
        print(json.dumps(dict(**identity, gradient=name, **metrics(actual[name], oracle[name]))),
              flush=True)


def parameters(n, seed, tau_value):
    torch.manual_seed(6701 + seed)
    b, h = 1, 4
    q, k = [torch.nn.Parameter(torch.randn(b, n, h, 64, device='cuda') * .2)
            for _ in range(2)]
    sq, sk = [torch.nn.Parameter(torch.randn(b, n, h, 32, device='cuda') * .5)
              for _ in range(2)]
    v = torch.nn.Parameter(torch.randn_like(q))
    eq, ek = [torch.nn.Parameter(torch.randn(h, 512, 64, device='cuda') * .2)
              for _ in range(2)]
    tau = torch.nn.Parameter(torch.full((h,), tau_value, device='cuda'))
    return [q, k, sq, sk, v, eq, ek, tau]


@torch.no_grad()
def core_case(n, seed, mode, tau_value):
    q, k, sq, sk, v, eq, ek, tau = parameters(n, seed, tau_value)
    q, k, sq, sk, v = q.bfloat16(), k.bfloat16(), F.silu(sq).bfloat16(), F.silu(sk).bfloat16(), v.bfloat16()
    # Actual V512 interpolation produces realistic normalizers and matches.
    from embedding_bhnd_reference import EmbInterpFunction
    qfk, kfq, lk, lq, _, _, ik, iq = EmbInterpFunction.apply(
        q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
        eq.bfloat16(), ek.bfloat16(), 1.)
    direction = torch.tensor([[False, True, False, True]], device='cuda')
    qv = torch.where(direction[:, None, :, None], q, kfq.transpose(1, 2))
    kv = torch.where(direction[:, None, :, None], qfk.transpose(1, 2), k)
    hard = torch.rand(1, 4, n, device='cuda') < dict(soft=0., mixed=.5, hard=1.)[mode]
    lq, lk = lq.transpose(1, 2), lk.transpose(1, 2)
    out, _, state = forward_core(qv, kv, sq, sk, v, lq, lk, iq, ik, direction, hard, tau,
                                save_state=True)
    dout = torch.randn_like(out).bfloat16()
    actual = backward_core(state, dout)  # Default BF16 private / FP32 atomic.
    oracle = dism_ref_backward(qv.double(), kv.double(), sq.double(), sk.double(),
                              lq.double(), lk.double(), iq, ik, direction, hard,
                              v.double(), tau.double(), dout.double())
    emit(actual, oracle, source='iid_core', seed=seed, mode=mode, n=n, tau=tau_value, step=0)


def trajectory(n, seed, mode, steps, tau_value=2.):
    owners = parameters(n, seed, tau_value)
    names = ('q', 'k', 'sq', 'sk', 'v', 'eq', 'ek', 'tau')
    optimizer = torch.optim.AdamW(owners, lr=.003, weight_decay=0.)
    direction = torch.tensor([[False, True, False, True]], device='cuda')
    hard = torch.rand(1, 4, n, device='cuda') < dict(soft=0., mixed=.5, hard=1.)[mode]

    def operands():
        q, k, sq, sk, v, eq, ek, tau = owners
        return q.bfloat16(), k.bfloat16(), F.silu(sq).bfloat16(), F.silu(sk).bfloat16(), v.bfloat16(), eq, ek, tau

    with torch.no_grad():
        target = .5 * voc_dism(*operands(), direction=direction, hard=hard).float()
    cumulative_actual = [torch.zeros_like(x, dtype=torch.float64) for x in owners]
    cumulative_oracle = [torch.zeros_like(x, dtype=torch.float64) for x in owners]
    for step in range(steps):
        actual_o = voc_dism(*operands(), direction=direction, hard=hard)
        # Same upstream derivative: avoid confusing forward loss differences
        # with backward errors. Explicit conversion matches the core contract.
        dout = (2. * (actual_o.float().detach() - target) / target.numel()).bfloat16()
        ga = torch.autograd.grad(actual_o, owners, dout)
        oracle_o = torch_voc(*operands(), direction, hard)
        ge = torch.autograd.grad(oracle_o, owners, dout.float())
        emit(dict(zip(names, ga)), dict(zip(names, ge)), source='trajectory',
             seed=seed, mode=mode, n=n, tau=float(owners[-1].detach().mean()), step=step)
        with torch.no_grad():
            for index, (x, y) in enumerate(zip(ga, ge)):
                cumulative_actual[index].add_(x)
                cumulative_oracle[index].add_(y)
                owners[index].grad = x
        optimizer.step()
        if step % 32 == 0:
            print(f'trajectory seed={seed} mode={mode} step={step}/{steps}', file=sys.stderr, flush=True)
    emit(dict(zip(names, cumulative_actual)), dict(zip(names, cumulative_oracle)),
         source='trajectory_sum', seed=seed, mode=mode, n=n, tau=tau_value, step=steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iid-seeds', type=int, default=8)
    parser.add_argument('--lengths', nargs='+', type=int, default=[65, 257, 513])
    parser.add_argument('--trajectory-seeds', type=int, default=3)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--trajectory-n', type=int, default=129)
    parser.add_argument('--trajectory-tau', type=float, default=2.)
    args = parser.parse_args()
    if cu.summary_key_dim() != 64 or cu.forward_head_dim() != 64 or not cu.forward_output_bf16():
        raise ValueError('This audit requires the default D64/DV64/BF16-output build')
    # FP32 reference GEMMs must not silently become TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    for seed, n, mode, tau in itertools.product(range(args.iid_seeds), args.lengths,
                                                ['soft', 'mixed', 'hard'], [2., math.log(64)]):
        core_case(n, seed, mode, tau)
        print(f'iid seed={seed} N={n} mode={mode} tau={tau:.3f}', file=sys.stderr, flush=True)
    for seed, mode in itertools.product(range(args.trajectory_seeds), ['soft', 'mixed', 'hard']):
        trajectory(args.trajectory_n, seed, mode, args.steps, args.trajectory_tau)


if __name__ == '__main__':
    main()
