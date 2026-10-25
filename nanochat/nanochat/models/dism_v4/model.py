"""Reuse nanochat state, sampling, optimizer and CE contracts with a v4 mixer."""
from flash_dism.modeling_dism import DismBlock
from flash_dism_v4.module import V4DismGdnAttention
from nanochat.models.dism.model import DismLM


class DismV4LM(DismLM):
    dynamic_compile = False
    supports_cuda_graph = False

    def _make_block(self, config, index):
        factory = V4DismGdnAttention if config.attention_type == 'hybrid_gdn' else None
        block = DismBlock(config, index, attention_factory=factory)
        if factory is not None:
            block.attn.block_size = self.config.block_size
            block.attn.readout_precision = self.config.readout_precision
            block.attn.soft_readout = self.config.soft_readout
            block.attn.hybrid_fusion = self.config.hybrid_fusion
        return block

    def forward(self, *args, **kwargs):
        if any(kwargs.get(key) is not None for key in
               ('graph_layout', 'graph_planner_bindings', 'gdn_chunk_indices')):
            raise ValueError('v4 prototype does not support planner/CUDA graph')
        return super().forward(*args, **kwargs)

    def estimate_flops(self):
        # Soft-mode dense GEMMs including backward recomputation. This is a
        # scheduling/MFU estimate, not a measured cost for sampled hard heads.
        c = self.config
        hybrid = sum(isinstance(layer.attn, V4DismGdnAttention) for layer in self.layers)
        pair_gemms = hybrid * c.n_head * c.sequence_len * (16*c.qk_vocab_size + 6*c.value_dim)
        if c.soft_readout == 'rmsnorm_jensen':
            pair_gemms = hybrid * c.n_head * c.sequence_len * (8*c.head_dim + 8*c.value_dim)
            # Extra expected-codebook projection, forward and backward.
            pair_gemms += hybrid * c.n_head * 6*c.qk_vocab_size*c.head_dim
        return 6*self.scaling_parameter_count() + pair_gemms
