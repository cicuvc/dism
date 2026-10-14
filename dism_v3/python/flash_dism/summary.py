"""Standalone aligned summary diagnostics; host preparation lives in C++."""
import cu_flash_dism as cu


def summarize_hard(idx_q, idx_k, hard, *, layout=None, ctas=0):
    """Standalone integer hard summary, not a complete attention forward.

    Labels: contiguous CUDA int32 [B,H,N]; hard: bool [training_rows,H].
    Fixed output: int32 [B,H,(N-1)//32,N]. Packed output follows layout's
    document-major [H,(length-1)//32,length] checkpoint regions.
    R[c,j] is the largest mismatching key coordinate on the diagonal segment
    ending at (32*c+31,j) within this 32-row strip, or -1 if none. Coordinates
    reset at each document. Only causal hard entries are computed; others -1.
    Compose with the preceding strip as max(R[c,j], prefix[c-1,j-32]) when
    j>=32, then L=j-prefix. This is NOT a soft affine/chunk_scan input.
    """
    return cu.hard_summary(idx_q, idx_k, hard, None if layout is None else layout.table, ctas)


def hard_chunk_scan(summary, *, layout=None, heads=0):
    """Compose hard R summaries into document-wide inclusive R boundaries.

    Returns a new int32 tensor with the same shape; input is not modified.
    Packed input requires its VarlenLayout and head count. Fixed input infers
    heads from [B,H,C,N]. Noncausal and skipped soft entries remain -1.
    On causal hard entries the match length is key_coordinate - boundary.
    """
    return cu.hard_chunk(summary, None if layout is None else layout.table, heads)


def chunk_scan(summary_a,summary_b,seqlen,*,backend=None):
    """Compose32-row summaries. Length must be a positive multiple of256."""
    return cu.summary_chunk(summary_a,summary_b,seqlen)


def summarize(q,k,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,*,ctas=0,backend=None):
    """SoA log2 summaries [B,H,(N-1)//32,N], N divisible by256.

    LSE must already contain raw_lse-rtau in natural-log units.
    No padding, RNG generation or soft readout is performed.
    """
    return cu.summary_forward(q,k,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,ctas)
