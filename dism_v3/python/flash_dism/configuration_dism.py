"""Hugging Face configuration for the DISM v3 decoder-only model."""
from transformers import PretrainedConfig


class DismConfig(PretrainedConfig):
    model_type = 'dism_v3'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=50257, hidden_size=256, num_hidden_layers=12,
                 num_heads=4, head_dim=64, value_dim=64, readout_dim=32,
                 qk_vocab_size=512, attention_type='hybrid', window_size=128,
                 conv_size=4, rope_theta=10000., qknorm_eps=1e-6, readout_l2_norm=False, soft_k_l2_norm=False,
                 value_residual=False,
                 vocab_transvq=False,
                 vocab_transvq_lite=False,
                 intermediate_size=None, hidden_ratio=4., norm_eps=1e-6,
                 initializer_range=.02, use_cache=True, fuse_cross_entropy=True,
                 ce_softcap=30., ce_chunk_size=None, ignore_index=-100,
                 pad_token_id=None, bos_token_id=50256, eos_token_id=50256,
                 tie_word_embeddings=False, post_norm=False, **kwargs):
        if attention_type not in ('hybrid', 'dism', 'hybrid_gdn', 'pure_gdn'):
            raise ValueError("unsupported v3 attention_type")
        if min(vocab_size, hidden_size, num_hidden_layers, num_heads) <= 0:
            raise ValueError('model dimensions must be positive')
        if intermediate_size is None:
            # Same SwiGLU parameter-budget convention as FLA GatedMLP / SBA.
            intermediate_size = 256 * ((int(hidden_size * hidden_ratio * 2 / 3) + 255) // 256)
        if intermediate_size <= 0:
            raise ValueError('intermediate_size must be positive')
        if tie_word_embeddings:
            raise ValueError('DISM LM currently uses untied input/output embeddings')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.readout_dim = readout_dim
        self.qk_vocab_size = qk_vocab_size
        self.attention_type = attention_type
        self.post_norm = bool(post_norm)
        self.window_size = window_size
        self.conv_size = conv_size
        self.rope_theta = rope_theta
        self.qknorm_eps = qknorm_eps
        self.readout_l2_norm = bool(readout_l2_norm)
        self.soft_k_l2_norm = bool(soft_k_l2_norm)
        self.value_residual = bool(value_residual)
        self.vocab_transvq = bool(vocab_transvq)
        self.vocab_transvq_lite = bool(vocab_transvq_lite)
        self.intermediate_size = intermediate_size
        self.hidden_ratio = hidden_ratio
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.fuse_cross_entropy = fuse_cross_entropy
        self.ce_softcap = ce_softcap
        self.ce_chunk_size = ce_chunk_size
        self.ignore_index = ignore_index
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id, tie_word_embeddings=False, **kwargs)
