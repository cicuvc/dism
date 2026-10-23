"""Opt-in, fixed-batch SAM generation session for the nanochat GDN/DISM model.

State belongs to a request, never to the model. No padding enters recurrent or
SAM history. Parameters must remain fixed and calls must use the owning stream.
"""
import math
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from flash_dism.inference import HardDismDecoder
from flash_dism.module import value_residual_mix


class SamDecodeSession:
    def __init__(self, model, prefill, capacity, *, rebuild_interval=512, profile_scope=None):
        if model.training or not model.eval_hard:
            raise ValueError('SAM decoding requires eval and all-hard routing')
        if not 0 < capacity <= model.config.sequence_len:
            raise ValueError('invalid session capacity')
        if model.config.mxfp8_projections:
            raise NotImplementedError('SAM model decode currently supports BF16 projections only')
        if any(not layer.post_norm for layer in model.layers):
            raise NotImplementedError('SAM model decode requires postnorm')
        if not getattr(model.layers[0].attn, 'is_pure_gdn', False):
            raise ValueError('value residual requires a pure GDN first layer')
        if prefill is None or not callable(prefill):
            raise ValueError('a SAM prefill engine is required')
        if rebuild_interval <= 0:
            raise ValueError('rebuild interval must be positive')
        if model.get_device().type != 'cuda':
            raise ValueError('SAM session requires a CUDA model')
        self.model, self.prefill, self.capacity = model, prefill, capacity
        self.rebuild_interval = rebuild_interval
        self._scope = profile_scope or (lambda name: nullcontext())
        self.position = 0
        self.batch = None
        self.states = [dict(conv={}, gdn=None, dism=None) for _ in model.layers]
        self.failed = False
        self.stream = torch.cuda.current_stream(model.get_device())

    def close(self):
        self.states.clear()
        self.failed = True

    @staticmethod
    def _conv(conv, x, states, name):
        projected = conv.project_linear(x)
        previous = states.get(name)
        joined = projected if previous is None else torch.cat((previous, projected), 1)
        output = conv.depthwise_silu(joined)[:, -x.shape[1]:].contiguous()
        keep = conv.kernel_size - 1
        tail = joined[:, -keep:] if keep else joined[:, :0]
        if tail.shape[1] < keep:
            tail = F.pad(tail, (0, 0, keep-tail.shape[1], 0))
        if previous is None:
            states[name] = tail.clone()
        else:
            previous.copy_(tail)
        return output

    def _mixer(self, attn, x, state, v_first, prime):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        def conv(name, dim):
            return self._conv(getattr(attn, name), x, state['conv'], name).unflatten(-1, (attn.heads, dim))
        with self._scope('decode/gdn_frontend'):
            q = conv('gdn_q_conv', attn.head_dim)
            k = conv('gdn_k_conv', attn.head_dim)
            v = conv('gdn_v_conv', attn.value_dim)
            if v_first is None:
                v_first = v
            if attn.value_residual:
                v = value_residual_mix(v, v_first, attn.gdn_v_residual_gate(x))
            beta = attn.gdn_b_proj(x).sigmoid()
            g = -attn.gdn_A_log.float().exp() * F.softplus(attn.gdn_a_proj(x).float() + attn.gdn_dt_bias.float())
        fn = chunk_gated_delta_rule if prime else fused_recurrent_gated_delta_rule
        with self._scope('decode/gdn_recurrent'):
            output, next_gdn = fn(q=q, k=k, v=v, g=g, beta=beta,
                initial_state=state['gdn'], output_final_state=True, use_qk_l2norm_in_kernel=True)
            if state['gdn'] is None:
                state['gdn'] = next_gdn
            else:
                state['gdn'].copy_(next_gdn)
        if not getattr(attn, 'is_pure_gdn', False):
            from flash_dism.kernels.voc_glue import prepare_embedding
            from flash_dism.kernels.interp_true import emb_fwd_true
            with self._scope('decode/dism_frontend'):
                q = conv('q_conv', attn.head_dim)
                k = conv('k_conv', attn.head_dim)
                v = conv('v_conv', attn.value_dim)
                if attn.value_residual:
                    v = value_residual_mix(v, v_first, attn.v_residual_gate(x))
                sq = attn._activate_readout(attn.sq_proj(x))
                sk = attn._activate_readout(attn.sk_proj(x), is_key=True)
                tau = F.softplus(attn.log_sel_tau.float()).contiguous() * attn.tau_scale
            with self._scope('decode/vocabulary'):
                n = x.shape[1]
                # Only vocabulary selection needs aligned physical rows. Slice labels
                # before planning/priming so synthetic rows never enter any history.
                padded = math.ceil(n / 256) * 256
                pad = lambda t: F.pad(t, (0, 0, 0, 0, 0, padded-n)).contiguous()
                q, k, eq, ek = prepare_embedding(pad(q), pad(k), attn.q_vocab.float(), attn.k_vocab.float())
                *_, ik, iq = emb_fwd_true(q, k, eq, ek, self.model.vocab_temperature, tau)
                iq, ik = iq[:, :, :n].contiguous(), ik[:, :, :n].contiguous()
            with self._scope('decode/sam_core'):
                if prime:
                    hard = self.prefill(iq, ik, sq, sk, v, tau).to(v.dtype)
                    decoder = HardDismDecoder(x.shape[0], attn.heads, attn.readout_dim,
                        attn.value_dim, self.capacity, tau, device=x.device,
                        planner_backend='gpu', rebuild_interval=self.rebuild_interval)
                    decoder.prime(iq, ik, sk, v)
                    state['dism'] = decoder
                else:
                    hard = state['dism'].append(iq, ik, sq, sk, v).to(v.dtype)
            output = hard + output
        with self._scope('decode/gate_norm_output'):
            gate = attn.g_proj_up(attn.g_proj_down(x)).unflatten(-1, (attn.heads, attn.value_dim))
            return attn.o_proj(attn.norm(output, gate).flatten(2)), v_first

    @torch.inference_mode()
    def features(self, tokens):
        if torch.cuda.is_current_stream_capturing() and not getattr(self, '_capturing', False):
            raise ValueError('model SAM session requires eager execution')
        if self.failed or self.model.training or not self.model.eval_hard:
            raise ValueError('session is closed, failed, or model is not in all-hard eval mode')
        if torch.cuda.current_stream(self.model.get_device()) != self.stream:
            raise ValueError('session CUDA stream changed')
        if (tokens.ndim != 2 or min(tokens.shape) <= 0
                or tokens.dtype not in (torch.int32, torch.int64)
                or tokens.device != self.model.get_device()):
            raise ValueError('tokens must be nonempty CUDA [B,N] on model device')
        b, n = tokens.shape
        if self.batch is not None and b != self.batch:
            raise ValueError('session batch cannot change or reorder')
        if self.position + n > self.capacity:
            raise ValueError('session capacity exceeded')
        if self.position and n != 1:
            raise ValueError('after prefill this session accepts one token per call')
        self.batch = b
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with self._scope('decode/embedding'):
                    x = self.model.embedding(tokens)
                v_first = None
                for layer, state in zip(self.model.layers, self.states):
                    branch, v_first = self._mixer(layer.attn, x, state, v_first, self.position == 0)
                    if layer.mxfp8:
                        # Keep the existing fused postnorm/quantization/FFN path.
                        # Padding is local to tokenwise operations, never cached.
                        with self._scope('decode/postnorm_quantize'):
                            padded = math.ceil(n / 256) * 256
                            pad = lambda t: F.pad(t, (0, 0, 0, padded-n)).contiguous()
                            h, row, rs, col, cs = layer.fused_attn_norm(pad(x), pad(branch), layer.attn_norm.weight)
                        with self._scope('decode/mxfp8_ffn'):
                            mlp = layer.mlp.forward_quantized(h, row, rs, col, cs)
                        with self._scope('decode/ffn_postnorm'):
                            x = layer.mlp_norm(h + mlp).to(torch.bfloat16)[:, :n].contiguous()
                    else:
                        x = layer.attn_norm(x + branch)
                        x = layer.mlp_norm(x + layer.mlp(x))
                with self._scope('decode/final_norm'):
                    result = self.model.norm(x).to(torch.bfloat16)
            self.position += n
            return result
        except Exception:
            self.failed = True  # A partly advanced cache cannot safely be retried.
            raise


class SamGraphDecodeSession:
    """Own a non-default stream and recapture after each SAM rebuild.

    Prefill and the first append execute eagerly. Capture itself consumes no
    tokens; only replay advances device state and the mirrored host positions.
    Calls are serialized with the caller stream. Returned hidden is owned.
    """
    def __init__(self, model, prefill, capacity, *, rebuild_interval=512):
        self.stream = torch.cuda.Stream(device=model.get_device())
        self.stream.wait_stream(torch.cuda.current_stream(model.get_device()))
        with torch.cuda.stream(self.stream):
            self.session = SamDecodeSession(model, prefill, capacity, rebuild_interval=rebuild_interval)
        self.graph = None
        self.warmed = False
        self.captures = 0
        self.static_tokens = None
        self.static_output = None

    @property
    def position(self):
        return self.session.position

    def close(self):
        self.stream.synchronize()
        self.graph = self.static_tokens = self.static_output = None
        self.session.close()

    @torch.inference_mode()
    def features(self, tokens):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('graph session manages its own capture and replay')
        inner = self.session
        if inner.failed or inner.model.training or not inner.model.eval_hard:
            raise ValueError('session is closed, failed, or model is not in all-hard eval mode')
        if (tokens.device != inner.model.get_device() or tokens.ndim != 2
                or min(tokens.shape) <= 0 or tokens.dtype not in (torch.int32, torch.int64)):
            raise ValueError('tokens must be nonempty CUDA [B,N] on model device')
        if self.position and tokens.shape != (inner.batch, 1):
            raise ValueError('graph session requires a fixed batch and one new token')
        if self.position + tokens.shape[1] > inner.capacity:
            raise ValueError('session capacity exceeded')
        caller = torch.cuda.current_stream(tokens.device)
        self.stream.wait_stream(caller)
        try:
            with torch.cuda.stream(self.stream):
                tokens.record_stream(self.stream)
                if not self.position or not self.warmed:
                    was_primed = bool(self.position)
                    result = inner.features(tokens)
                    self.warmed = was_primed
                else:
                    decoders = [s['dism'] for s in inner.states if s['dism'] is not None]
                    if any(d.position-d.snapshot_position >= d.rebuild_interval for d in decoders):
                        self.graph = self.static_output = None
                        for d in decoders:
                            d.core.rebuild()
                            d.snapshot_position = d.position
                    if self.graph is None:
                        self.static_tokens = tokens.clone()
                        self.stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        position = inner.position
                        inner._capturing = True
                        try:
                            with torch.cuda.graph(graph, stream=self.stream):
                                self.static_output = inner.features(self.static_tokens)
                        finally:
                            inner._capturing = False
                            inner.position = position
                            for d in decoders:
                                d.position = position
                        self.graph = graph
                        self.captures += 1
                    self.static_tokens.copy_(tokens)
                    self.graph.replay()
                    inner.position += 1
                    for d in decoders:
                        d.position += 1
                    result = self.static_output.clone()
                result.record_stream(caller)
            caller.wait_stream(self.stream)
            return result
        except Exception:
            inner.failed = True
            caller.wait_stream(self.stream)
            raise


@torch.inference_mode()
def generate_until_sam(adapter, requests):
    """Evaluation adapter entrypoint: greedy, independent unpadded requests.

    Deliberately serial across requests in this prototype. This avoids changing
    batch membership or ingesting pad/EOS tokens into an active request's cache.
    """
    results = []
    for request in requests:
        prompt, options = request.args
        if options.get('do_sample', False) or options.get('temperature', 0) not in (0, None):
            raise ValueError('Greedy evaluation only')
        limit = int(options.get('max_gen_toks', 48))
        if not 0 < limit < adapter.max_length:
            raise ValueError(limit)
        ids = adapter.tok_encode(prompt)
        budget = adapter.max_length - limit
        adapter.truncated += int(len(ids) > budget)
        ids = ids[-budget:]
        until = options.get('until', [])
        stops = [until] if isinstance(until, str) else until
        session_type = SamGraphDecodeSession if getattr(adapter, 'decode_cuda_graph', False) else SamDecodeSession
        session = session_type(adapter.model, adapter.prefill, adapter.max_length,
            rebuild_interval=getattr(adapter, 'decode_rebuild_interval', 512))
        output = []
        try:
            tokens = torch.tensor([ids], device=adapter.model.get_device(), dtype=torch.long)
            for step in range(limit):
                hidden = session.features(tokens)
                token = int(adapter._greedy(hidden[:, -1])[0])
                if token == adapter.eot_token_id:
                    break
                output.append(token)
                text = adapter.tokenizer.decode(output, skip_special_tokens=True)
                if any(s and s in text for s in stops):
                    break
                tokens = torch.tensor([[token]], device=tokens.device, dtype=torch.long)
            text = adapter.tokenizer.decode(output, skip_special_tokens=True)
            for stop in stops:
                if stop:
                    text = text.split(stop)[0]
            results.append(text)
        finally:
            session.close()
    return results
