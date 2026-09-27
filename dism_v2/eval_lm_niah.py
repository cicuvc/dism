"""Small synthetic key/value needle smoke test, not an official NIAH benchmark."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch


def build_prompt(background, length, needle, suffix, depth):
    budget = length - len(needle) - len(suffix)
    if budget < 0 or len(background) < budget:
        raise ValueError('Insufficient background or context budget')
    at = int(depth * budget)
    ids = background[:at] + needle + background[at:budget] + suffix
    assert len(ids) == length
    return ids, at


def case_plan(samples, seed):
    """Paired lengths; each depth has exactly balanced main answer colors."""
    rng = random.Random(seed)
    if not samples:
        return [(key, *rng.sample(range(8), 2), (.1, .5, .9))
                for key in ('silver telescope', 'wooden compass')]
    if samples % 64 or not 0 < samples <= 256:
        raise ValueError('Expanded sample count must be64/128/192/256')
    adjectives = 'silver wooden ancient tiny giant hidden broken golden copper bronze crystal old new small large mysterious'.split()
    nouns = 'telescope compass lantern mirror statue camera suitcase notebook'.split()
    keys = [f'{a} {b}' for a in adjectives for b in nouns]
    rng.shuffle(keys)
    plan = []
    for group in range(samples // 16):
        targets = list(range(8))
        rng.shuffle(targets)
        for target in targets:
            alternative = (target + rng.randrange(1, 8)) % 8
            plan.append((keys[len(plan)], target, alternative, ((.1, .35, .65, .9)[group % 4],)))
    return plan


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=0, help='0: original smoke; expanded:64/128/192/256 main cases')
    p.add_argument('--seed', type=int, default=9321)
    a = p.parse_args()
    plan = case_plan(a.samples, a.seed)
    if a.output.exists():
        p.error('Choose a new report path')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg, step = saved['config'], saved['step']
    os.environ['DISM_TILE_LSE'] = cfg['tile_lse']
    os.environ['DISM_BWD_OPT'] = str(cfg['backward_opt'])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from .lm_model import DecoderLM, LMConfig
    from .lm_data import PackedStream, split_files
    from .train_lm import load_tokenizer
    c = LMConfig(**cfg['model'])
    model = DecoderLM(replace(c, context=8192)).cuda().eval()
    model.load_state_dict(saved['model'], strict=True)
    del saved
    tok = load_tokenizer(cfg['tokenizer'])
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    colors = [' red', ' blue', ' green', ' yellow', ' white', ' black', ' orange', ' purple']
    candidates = [encode(x) for x in colors]
    assert all(len(x) == 1 for x in candidates)
    candidate_ids = torch.tensor([x[0] for x in candidates], device='cuda')
    _, files = split_files(cfg['data'])
    stream = PackedStream(files, tok, context=8192, batch_size=1, repeat=False)
    rows = []
    start = time.perf_counter()
    def evaluate(ids, target):
        x = torch.tensor([ids], device='cuda')
        gen = torch.Generator(device='cuda').manual_seed(779)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = model.forward_features(x, 1., gen)
            logits = model.lm_head(h[:, -1]).float()[0]
        if c.softcap is not None:
            logits = c.softcap * torch.tanh(logits / c.softcap)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite logits')
        lp = logits.log_softmax(-1)
        scores = logits[candidate_ids]
        pred = logits.argmax().item()
        return dict(greedy_token=tok.decode([pred]), exact_match=pred==candidate_ids[target].item(),
                    candidate_top1=colors[scores.argmax().item()],
                    candidate_correct=scores.argmax().item()==target,
                    target_rank_in_candidates=1+int((scores>scores[target]).sum()),
                    target_nll=-lp[candidate_ids[target]].item(),
                    target_probability=lp[candidate_ids[target]].exp().item(),
                    candidate_logits=scores.tolist())
    for trial, (key, target, alternative, depths) in enumerate(plan):
        background = stream.next_batch()[0][0].tolist()
        prefix = f'The secret color of the {key} is'
        needle = encode('\n\n' + prefix + colors[target] + '.\n\n')
        changed = encode('\n\n' + prefix + colors[alternative] + '.\n\n')
        suffix = encode('\n\n' + prefix)
        assert len(needle) == len(changed)
        # Recent/easy positive control and no-information prior control.
        for condition, ids in [('near', needle+suffix), ('no_information', suffix)]:
            rows.append(dict(trial=trial, condition=condition, length=len(ids), depth=None,
                             target=colors[target], key=key, **evaluate(ids, target)))
        for length in (2048, 8192):
            for depth in depths:
                ids, at = build_prompt(background, length, needle, suffix, depth)
                counter = ids.copy()
                counter[at:at+len(needle)] = changed
                # Same length, exactly same suffix/background; remove all key/value
                # tokens by replacing only the inserted span with newline tokens.
                absent = ids.copy()
                absent[at:at+len(needle)] = [encode('\n')[0]] * len(needle)
                entry = dict(trial=trial, condition='needle', length=length, depth=depth,
                    key=key, target=colors[target], alternative=colors[alternative],
                    needle_text=tok.decode(needle), suffix_text=tok.decode(suffix),
                    needle_start=at, tokens_after_needle=length-at-len(needle),
                    prompt_sha256=hashlib.sha256(torch.tensor(ids).numpy().tobytes()).hexdigest(),
                    **evaluate(ids, target))
                entry['absent'] = evaluate(absent, target)
                entry['counterfactual'] = evaluate(counter, alternative)
                entry['nll_gain_over_absent'] = entry['absent']['target_nll']-entry['target_nll']
                rows.append(entry)
                if not a.samples:
                    print(json.dumps(entry), flush=True)
        if a.samples and (trial+1) % 8 == 0:
            print(json.dumps(dict(main_cases=(trial+1)*2, total=a.samples,
                                  seconds=time.perf_counter()-start)), flush=True)
    report = dict(checkpoint=str(a.checkpoint), step=step, hard_prob=1., seed=a.seed,
        model=asdict(c), evaluation_context=8192,
        main_samples=a.samples or 12, independent_key_background_pairs=len(plan),
        candidate_colors=colors, rows=rows, seconds=time.perf_counter()-start,
        caveats=['Two lengths reuse each key/background; main cases are not all independent.',
                 'Expanded mode balances eight target colors at each length/depth; original smoke does not.',
                 'Exact repeated-prefix completion, intentionally easier than instruction-following QA.',
                 'Eight single-token colors; forced-choice chance reference12.5%, actual model priors nonuniform.',
                 'Near/absent/counterfactual controls distinguish task formatting and prior guessing.',
                 'Natural held-out packed background, needle may be inserted inside a sentence.',
                 'Only configured context capacity enlarged; no positional-frequency changes.',
                 'GDN/DISM and pure GDN configurations have no SWA or RoPE.'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)
    for condition in ('near','no_information','needle'):
        selected = [r for r in rows if r['condition']==condition]
        print(json.dumps(dict(condition=condition, samples=len(selected),
            exact=sum(r['exact_match'] for r in selected),
            candidate_correct=sum(r['candidate_correct'] for r in selected),
            mean_nll=sum(r['target_nll'] for r in selected)/len(selected))), flush=True)


if __name__ == '__main__':
    main()
