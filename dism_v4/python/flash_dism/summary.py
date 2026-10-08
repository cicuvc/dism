"""Standalone aligned summary diagnostics; host preparation lives in C++."""
import cu_flash_dism as cu


def chunk_scan(summary_a,summary_b,seqlen,*,backend=None):
    """Compose32-row summaries. Length must be a positive multiple of256."""
    return cu.summary_chunk(summary_a,summary_b,seqlen)


def summarize(q,k,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,*,ctas=0,backend=None):
    """SoA log2 summaries [B,H,(N-1)//32,N], N divisible by256.

    LSE must already contain raw_lse-rtau in natural-log units.
    No padding, RNG generation or soft readout is performed.
    """
    return cu.summary_forward(q,k,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,ctas)
