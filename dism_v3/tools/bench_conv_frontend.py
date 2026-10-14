"""Benchmark-only fused -> linear/FLA checkpoint conversion; never writes training state."""
import copy

import torch
from nanochat.models.spec import ModelSpec
from nanochat.models import optimizer_schema_fingerprint


def mapped_name(name):
    return name.removesuffix('conv_weight') + 'conv.weight' if name.endswith('.conv_weight') else name


def to_fla(value):
    # Fused stores [lag, channel], current token first. Conv1d stores
    # [channel, 1, tap], oldest token first. Moments need the same permutation.
    return value.flip(0).T.unsqueeze(1).contiguous()


def convert(spec, model_state, optim_state, schema):
    if spec.config.get('conv_impl', 'fused') != 'fused':
        raise ValueError('conversion requires a fused source checkpoint')
    original_fingerprint = optimizer_schema_fingerprint(schema)
    target_spec = ModelSpec(spec.architecture, spec.architecture_version,
                           dict(spec.config, conv_impl='linear_fla', conv_backend='cuda'))
    converted = {mapped_name(n): to_fla(v) if n.endswith('.conv_weight') else v
                 for n, v in model_state.items()}
    if len(converted) != len(model_state):
        raise ValueError('state name collision')
    target_optim = copy.deepcopy(optim_state)
    target_schema = copy.deepcopy(schema)
    if len(schema['groups']) != len(target_optim['param_groups']):
        raise ValueError('optimizer group mismatch')
    changed = []
    for source, dest, group in zip(schema['groups'], target_schema['groups'], target_optim['param_groups']):
        if len(source['parameters']) != len(group['params']):
            raise ValueError('optimizer parameter count mismatch')
        dest['parameters'] = [mapped_name(n) for n in source['parameters']]
        for name, pid in zip(source['parameters'], group['params']):
            if not name.endswith('.conv_weight'):
                continue
            slots = target_optim['state'][pid]
            for slot, tensor in slots.items():
                if isinstance(tensor, torch.Tensor) and tensor.shape == model_state[name].shape:
                    slots[slot] = to_fla(tensor)
                elif slot != 'step':
                    raise ValueError(f'unexpected optimizer slot {name}/{slot}')
            changed.append(name)
    info = dict(source_config=spec.to_dict(), target_config=target_spec.to_dict(),
                source_optimizer_fingerprint=original_fingerprint,
                target_optimizer_fingerprint=optimizer_schema_fingerprint(target_schema),
                converted_parameters=changed,
                transform='[lag,channel] -> flip(lag).T[:,None,:]; same permutation for Adam moments; '
                          'all group positions/hyperparameters/step counters and other tensors retained')
    return target_spec, converted, target_optim, target_schema, info


def validate_convolution(model_state):
    """Real checkpoint filter, ragged documents, BF16 forward/backward vs FP32 oracle."""
    from flash_dism.kernels.conv1d import short_conv_silu
    from nanochat.models.dism.conv import LinearShortConvSiLU
    import torch.nn.functional as F
    name = 'layers.0.attn.gdn_q_conv'
    w = model_state[name+'.weight'].cuda()
    c = model_state[name+'.conv_weight'].cuda()
    layer = LinearShortConvSiLU(w.shape[1], w.shape[0], c.shape[0], 'cuda').cuda()
    if layer.conv.backend != 'cuda':
        raise RuntimeError('CUDA convolution backend unavailable; refusing silent fallback')
    with torch.no_grad():
        layer.weight.copy_(w)
        layer.conv.weight.copy_(to_fla(c))
    generator = torch.Generator(device='cuda').manual_seed(1803)
    x = torch.randn((1, 1024, w.shape[1]), generator=generator, device='cuda')
    do = torch.randn((1, 1024, w.shape[0]), generator=generator, device='cuda')
    boundaries = torch.tensor([0,256,768,1024], device='cuda', dtype=torch.int32)
    result = {}
    outputs = {}
    for mode in ('reference', 'fused', 'linear_fla'):
        a = x.clone().requires_grad_()
        if mode == 'linear_fla':
            params = [a, layer.weight, layer.conv.weight]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = layer(a, cu_seqlens=boundaries, max_seqlen=512)
        else:
            ww, cc = w.clone().requires_grad_(), c.clone().requires_grad_()
            params = [a, ww, cc]
            if mode == 'fused':
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    y = short_conv_silu(a, ww, cc, boundaries, 512)
            else:
                z = F.linear(a, ww)
                pieces = []
                for start, end in ((0,256), (256,768), (768,1024)):
                    p = z[:,start:end].transpose(1,2)
                    pieces.append(F.silu(F.conv1d(F.pad(p,(c.shape[0]-1,0)), to_fla(cc),
                                                  groups=c.shape[1])).transpose(1,2))
                y = torch.cat(pieces,dim=1)
        grads = list(torch.autograd.grad((y.float()*do).sum(), params))
        if mode == 'linear_fla':
            grads[-1] = grads[-1].squeeze(1).T.flip(0).contiguous()
        outputs[mode] = [y.float().detach(), *[g.detach().float() for g in grads]]
    for mode in ('fused', 'linear_fla'):
        result[mode] = {}
        for name, actual, reference in zip(('output','dx','dweight','dconv'), outputs[mode], outputs['reference']):
            a, r = actual.flatten(), reference.flatten()
            if not torch.isfinite(a).all():
                raise FloatingPointError(f'{mode}/{name} nonfinite')
            result[mode][name] = dict(relative_l2=float((a-r).norm()/r.norm()),
                                      cosine=float(F.cosine_similarity(a,r,dim=0)),
                                      norm_ratio=float(a.norm()/r.norm()),
                                      max_abs=float((a-r).abs().max()))
    return dict(backend=layer.conv.backend, parameters='first-layer real checkpoint Q convolution',
                input='seeded random FP32 input, ragged lengths256/512/256; BF16 autocast vs FP32 oracle',
                comparisons=result)
