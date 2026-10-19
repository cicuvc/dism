"""DISM hybrid pretraining registration; optional dependencies load lazily."""
from dataclasses import dataclass
from nanochat.models import register_model, ModelCapabilities


@dataclass
class DismConfig:
    direction_policy: str = "random"
    direction_freeze_step: int = 3000
    direction_antithetic: bool = False
    hard_granularity: str = "sequence"
    sequence_len: int = 1024
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    intermediate_size: int = 2152
    head_dim: int = 64
    value_dim: int = 64
    readout_dim: int = 32
    qk_vocab_size: int = 512
    post_norm: bool = False
    alternating_gdn: bool = False
    rear_half_dism: bool = False
    window_size: int = 128
    anneal_steps: int = 3000
    hard_start_fraction: float = 0.0
    softcap: float = 30.0
    loss_balance: str = "document"
    rng_seed: int = 1729
    hard_end_fraction: float = 1.0
    readout_mode: str = "default"
    vocab_share_qk: bool = False
    vocab_share_heads: bool = False
    vocab_share_layers: bool = False
    soft_qk_rope: bool = False
    shared_readout: bool = False
    soft_k_l2_norm: bool = False
    vocab_temperature: float = 1.0
    vocab_temperature_start: float = 1.0
    vocab_temperature_end: float = 1.0
    split_branch_output: bool = False
    hard_prob_max: float = 1.0  # Preserve old checkpoint semantics; new runs explicitly set .95.
    hard_prob_start: float = 0.0
    selector_beta1: float = 0.0  # >0 splits gated DISM selector params into their own AdamW groups
    selector_beta2: float = 0.0
    value_residual: bool = False
    value_residual_gate_bias: float = 2.0
    conv_impl: str = "fused"
    conv_backend: str = "cuda"
    gqa_query_channels: int = 1
    # Opt out of the global count-normalized objective that arrived with
    # document-balanced training, restoring the older per-microbatch mean
    # (loss / grad_accum_steps on the model's own token mean). Default off
    # keeps every existing spec bit-identical.
    legacy_loss_normalization: bool = False
    # Gradient scale for rtau on hard rows only. 1.0 = unchanged; 0.0 detaches
    # rtau on the hard path so the soft relaxation alone sets it. On layers
    # 8/10/11 both paths push rtau down (dL/dtau > 0) while raising it improves
    # NIAH, so this knob trades "tau is trained by both paths" against "tau is
    # trained by the soft path only".
    hard_tau_grad_scale: float = 1.0
    # rtau release schedule. While the retrieval circuit is still forming, a
    # large rtau can winner-take-all onto sparse accidental matches and starve
    # its own gradient; once it has formed, a larger rtau raises the signal by
    # suppressing dilution from the many short matches. tau_scale ramps from
    # tau_schedule_floor to 1.0 over the first tau_schedule_fraction of the run
    # (0 disables the schedule). Applied to training and eval alike, so the
    # final steps are the deployed value.
    tau_schedule_fraction: float = 0.0
    tau_schedule_floor: float = 0.0
    # Selector normalisation: RMSNorm at the q/k conv+silu exit and at the
    # codebook exit, each with a per-channel scale of selector_norm_scale_init.
    # Frozen for the first selector_norm_freeze_fraction of the run, then
    # released. Calibrated on a trained checkpoint: scale 1.0 collapses the
    # selector to ~2 effective symbols, 0.5 gives 3.6-5.2, and ~0.35 reproduces
    # the un-normalised level, so 0.2 starts slightly flatter than today.
    selector_norm: bool = False
    selector_norm_scale_init: float = 0.2
    selector_norm_freeze_fraction: float = 0.5
    # QB is opt-in; each side uses zero strength to disable its price.
    qb_q_strength: float = 0.0
    qb_k_strength: float = 0.0
    qb_iterations: int = 4
    qb_update_interval: int = 16
    qb_samples: int = 2048
    qb_update_rate: float = 0.1
    # >1 groups the query heads so every kv_group_size of them share one
    # key/value head (classic GQA). Mutually exclusive with gqa_query_channels.
    kv_group_size: int = 1
    vocab_transvq: bool = False
    vocab_transvq_lite: bool = False
    vocab_transvq_map: str = "linear_attn"
    vocab_transvq_rank: int = 4
    vocab_transvq_separate_qk: bool = False
    # Optional product-quantised q/k codebook. Empty keeps the dense V-way
    # vocabulary; (16,32) preserves V=512 with two 32-dimensional groups.
    pq_vocab_sizes: tuple[int, ...] = ()
    # Compact head-paired q/k tables. "swap_halves" keeps H/2 Q tables and H/2
    # K tables; second-half heads use the paired tables in swapped roles.
    vocab_head_pairing: str = ""
    # Recurrent half of the hybrid: the original gated delta rule or the
    # gate-free sliding-window DeltaNet. The two use the same layout symbols.
    recurrent_mixer: str = "gdn"
    sliding_delta_chunk_size: int = 32
    # Per-layer mixer layout, e.g. "ggggggddgddg". 'g'=pure recurrent mixer,
    # 'd'=DISM only, 'h'=DISM+recurrent hybrid. Empty string keeps the
    # alternating_gdn/rear_half_dism
    # behaviour. Used to test dedicated (non-mixed) layer layouts.
    layer_layout: str = ""

    def __post_init__(self):
        # JSON permits integer softcaps. Normalize before Dynamo can turn an
        # integer attribute into a SymInt (Triton would infer an integer ABI
        # even though CE validates/converts it to float at runtime).
        self.softcap = float(self.softcap)
        if isinstance(self.pq_vocab_sizes, list):
            self.pq_vocab_sizes = tuple(self.pq_vocab_sizes)
        if self.qb_q_strength or self.qb_k_strength:
            if self.direction_policy != 'true' or self.direction_antithetic:
                raise ValueError('QB requires direction_policy=true and direction_antithetic=false')


def _build(config):
    from .model import DismLM
    return DismLM(config)


register_model('dism_hybrid', 1, DismConfig, _build,
               ModelCapabilities(varlen=True, generation=False, sliding_window=True, compile=True),
               scaling_reference_spec=lambda spec: spec)


def _build_swa(config):
    from .swa import SwaLM
    return SwaLM(config)


register_model('swa_control', 1, DismConfig, _build_swa,
               ModelCapabilities(varlen=True, generation=False, sliding_window=True, compile=True),
               scaling_reference_spec=lambda spec: spec)


def _build_gdn(config):
    from .model import DismLM
    if config.split_branch_output:
        raise ValueError("Initial GDN hybrid uses one shared post-sum output path")
    return DismLM(config, attention_type='hybrid_gdn')

register_model('dism_gdn_hybrid', 1, DismConfig, _build_gdn,
               ModelCapabilities(varlen=True, generation=False, sliding_window=False, compile=True),
               scaling_reference_spec=lambda spec: spec)
