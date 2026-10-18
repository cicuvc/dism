"""Small local-only paired NIAH evaluation, isolated from training data/RNG.

Retains the circuit-probe protocol: bare number-tail insertion, six-token
answer budget, teacher-forced answer scoring. No downloads or loader inputs.
"""
from contextlib import contextmanager
import hashlib
import json
import logging
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)
DEFAULT_DATA = str(Path.home() / 'downstream-eval/data/pg19/test.parquet')


@contextmanager
def _preserve_rng(device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def _evaluation_state(model, context):
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(name, buffer, buffer.detach().clone()) for name, buffer in model.named_buffers()]
    old_hard, old_context = model.eval_hard, model.config.sequence_len
    changed = []
    try:
        model.eval()
        model.eval_hard = True
        model.config.sequence_len = max(old_context, context)
        yield
    finally:
        model.eval_hard = old_hard
        model.config.sequence_len = old_context
        for module, mode in modes:
            module.training = mode
        # Normal eval must not write counters/QB state. Do not even copy into
        # unchanged buffers: their storage and tensor version remain untouched.
        for name, buffer, saved in buffers:
            if not torch.equal(buffer, saved):
                buffer.copy_(saved)
                changed.append(name)
        if changed:
            raise RuntimeError('NIAH eval mutated model buffers (restored): '+', '.join(changed))


def build_suite(tokenizer, data_path, contexts, depths, trials, seed):
    """Read only the first six PG19 texts; construct deterministic CPU inputs."""
    import pyarrow.parquet as pq
    hay = []
    parquet = pq.ParquetFile(data_path)
    try:
        for index, batch in enumerate(parquet.iter_batches(batch_size=1, columns=['text'])):
            text = batch.column('text')[0].as_py()
            if not isinstance(text, str):
                raise ValueError('NIAH background text must be a string')
            hay.extend(tokenizer.encode(text))
            if index == 5:
                break
    finally:
        parquet.close()
    query = tokenizer.encode('\nThe special magic number is')
    if not query:
        raise ValueError('NIAH query has no tokens')
    eos = getattr(getattr(tokenizer, 'tokenizer', None), 'eos_token_id', None)
    pad = tokenizer.get_bos_token_id() if eos is None else eos
    cells = []
    for length in contexts:
        for depth in depths:
            rng = random.Random(f'{seed}:{length}:{depth}')
            samples = []
            for _ in range(trials):
                number = rng.randint(10000, 99999)
                needle = tokenizer.encode(f'\nThe special magic number is {number}.')
                if needle[:len(query)] != query:
                    raise ValueError('NIAH tokenizer does not preserve the query prefix')
                tail = needle[len(query):]
                if not tail:
                    raise ValueError('NIAH answer has no tokens')
                budget = length-6
                haylen = budget-len(query)-len(tail)
                if haylen < 1 or len(hay) < haylen+len(tail):
                    raise ValueError(f'NIAH background too short or context too small for {length} tokens')
                offset = rng.randrange(len(hay)-haylen-len(tail)+1)
                background = hay[offset:offset+haylen]
                position = int(depth*max(1, haylen-1))
                sample = dict(number=number, target_start=budget, target_tokens=tail[:6])
                # Counterfactual: insert a DIFFERENT number but still score the
                # true one. margin = counterfactual_nll - present_nll is a
                # log-likelihood ratio in which the answer's prior cancels, so
                # unlike present_nll and gain it is comparable across models
                # with different number priors.
                other = number
                while other == number:
                    other = rng.randint(10000, 99999)
                other_tail = tokenizer.encode(f'\nThe special magic number is {other}.')[len(query):]
                for condition in ('present', 'absent', 'counterfactual'):
                    if condition == 'present':
                        insert = tail
                    elif condition == 'absent':
                        insert = hay[offset+haylen:offset+haylen+len(tail)]
                    else:
                        insert = other_tail[:len(tail)]
                        if len(insert) < len(tail):
                            insert = (insert + tail)[:len(tail)]
                    head = background[:position]+insert+background[position:]+query
                    sequence = (head+tail)[:length]
                    sequence += [pad]*(length-len(sequence))
                    sample[condition] = sequence
                samples.append(sample)
            cells.append(dict(context=length, depth=depth, samples=samples))
    # Fingerprint includes actual tokenized inputs, not just the seed/path.
    fingerprint = hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest()
    return dict(cells=cells, sha256=fingerprint)


class _FeaturesReady(Exception):
    pass


def _score(model, tokens, start, target, device):
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    cu = torch.tensor([0, len(tokens)], dtype=torch.int32, device=device)
    selected = []

    def capture(_module, _inputs, output):
        selected.append(output.reshape(-1, output.shape[-1])[start-1:start-1+len(target)].detach())
        # Avoid allocating logits for the complete long context. Like the
        # standalone probe, score only answer positions using an FP32 head.
        raise _FeaturesReady()

    handle = model.norm.register_forward_hook(capture)
    try:
        try:
            model(x, cu_seqlens=cu)
        except _FeaturesReady:
            pass
    finally:
        handle.remove()
    if len(selected) != 1:
        raise RuntimeError('NIAH final-norm hook did not capture exactly one output')
    with torch.autocast(device.type, enabled=False):
        logits = F.linear(selected[0].float(), model.lm_head.weight.float(),
                          None if model.lm_head.bias is None else model.lm_head.bias.float())
        softcap = float(model.config.softcap)
        if softcap > 0:
            logits = softcap*torch.tanh(logits/softcap)
        targets = torch.tensor(target, dtype=torch.long, device=device)
        nll = F.cross_entropy(logits, targets, reduction='mean')
        hits = logits.argmax(-1).eq(targets)
    return float(nll), int(hits.sum()), len(target), int(hits.all())


class ShortNIAH:
    """Cache local CPU trials across eval steps; never accept a training loader."""
    def __init__(self, data_path=DEFAULT_DATA, *, contexts=(8192,), depths=(.05,.5,.95),
                 trials=2, seed=1234):
        self.data_path = Path(data_path).expanduser()
        self.contexts, self.depths = tuple(contexts), tuple(depths)
        if not self.contexts or any(type(n) is not int or n <= 0 or n % 256 for n in self.contexts):
            raise ValueError('NIAH contexts must be positive multiples of 256')
        if not self.depths or any(not math.isfinite(d) or not 0 <= d <= 1 for d in self.depths):
            raise ValueError('NIAH depths must be finite and in [0,1]')
        if type(trials) is not int or trials <= 0:
            raise ValueError('NIAH trials must be a positive integer')
        self.trials, self.seed = trials, seed
        self._key, self._suite, self._tokenizer = None, None, None
        self._warned = set()

    def _skip(self, reason):
        if reason not in self._warned:
            logger.warning('Skipping short NIAH evaluation: %s', reason)
            self._warned.add(reason)
        return dict(status='skipped', reason=reason, metrics={'niah/skipped': 1})

    @torch.no_grad()
    def evaluate(self, model, tokenizer):
        if not (getattr(model, 'aligned_segment_ends', False) and hasattr(model, 'eval_hard')
                and hasattr(model, 'norm') and hasattr(model, 'lm_head')):
            return self._skip('model does not support the DISM packed NIAH probe')
        device = next(model.parameters()).device
        with _preserve_rng(device):
            # Only data/tokenizer preparation errors are skippable. Model/runtime
            # failures propagate after state restoration, never become fake zeros.
            try:
                stat = self.data_path.stat()
                key = (id(tokenizer), stat.st_size, stat.st_mtime_ns)
                if key != self._key:
                    suite = build_suite(tokenizer, self.data_path, self.contexts,
                                        self.depths, self.trials, self.seed)
                    self._suite, self._key, self._tokenizer = suite, key, tokenizer
            except (OSError, ValueError, KeyError, ImportError) as error:
                return self._skip(f'{self.data_path}: {type(error).__name__}: {error}')
            cells, aggregate = [], {c: [0.,0,0,0,0] for c in ('present','absent','counterfactual')}
            with _evaluation_state(model, max(self.contexts)):
                for cell in self._suite['cells']:
                    totals = {c: [0.,0,0,0,0] for c in ('present','absent','counterfactual')}
                    for sample in cell['samples']:
                        for condition in totals:
                            nll, hits, count, all_hit = _score(model, sample[condition],
                                sample['target_start'], sample['target_tokens'], device)
                            for dest in (totals[condition], aggregate[condition]):
                                for i, value in enumerate((nll,hits,count,all_hit,1)):
                                    dest[i] += value
                    metrics = self._summarize(totals)
                    cells.append(dict(context=cell['context'], depth=cell['depth'], **metrics))
            metrics = {'niah/'+k: v for k,v in self._summarize(aggregate).items()}
            metrics['niah/skipped'] = 0
            for cell in cells:
                prefix = f"niah/ctx_{cell['context']}/depth_{cell['depth']:g}/"
                metrics.update({prefix+k: v for k,v in cell.items() if k not in ('context','depth')})
            return dict(status='ok', seed=self.seed, trials=self.trials, data_path=str(self.data_path),
                        suite_sha256=self._suite['sha256'], task='bare-number; six-token budget; teacher-forced',
                        cells=cells, metrics=metrics)

    @staticmethod
    def _summarize(totals):
        result = {}
        for condition, (nll,hits,tokens,all_hit,trials) in totals.items():
            result.update({condition+'_nll': nll/trials,
                           condition+'_token_acc': hits/tokens,
                           condition+'_trial_all': all_hit/trials})
        result['gain'] = result['absent_nll']-result['present_nll']
        # Prior-invariant retrieval strength: the answer's prior cancels in both
        # of these, so they compare across models whose number priors differ.
        result['retrieval_ratio'] = (result['gain']/result['absent_nll']
                                     if result['absent_nll'] > 0 else 0.0)
        result['margin'] = result['counterfactual_nll']-result['present_nll']
        return result

    def evaluate_distributed(self, model, tokenizer):
        """Only rank 0 evaluates the unwrapped model; broadcast success/skip/error.

        Other ranks wait here, so they cannot advance their training iterator
        while rank 0 probes. No collectives run inside the eval-mode model.
        """
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        payload, failure = [None], None
        if rank == 0:
            try:
                payload[0] = dict(result=self.evaluate(model, tokenizer))
            except Exception as error:
                failure = error
                payload[0] = dict(error=f'{type(error).__name__}: {error}')
        if distributed:
            dist.broadcast_object_list(payload, src=0)
        if 'error' in payload[0]:
            raise RuntimeError('Short NIAH evaluation failed: '+payload[0]['error']) from failure
        return payload[0]['result']
