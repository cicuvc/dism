import sys
sys.path.append('build/linux/x86_64/release')

import torch
import dism_C

@torch.compile
def prepare_offsets(seqlens: torch.Tensor, Q_BLOCK: int, VH_BLOCK: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_blks = (seqlens + Q_BLOCK - 1) // Q_BLOCK
    cu_seq_blks = torch.cumsum(seq_blks, 0)
    q_offsets = torch.zeros((cu_seq_blks[-1], ), dtype = seqlens.dtype)
    q_offsets[cu_seq_blks[:-1]] = seqlens[:-1] - seq_blks[:-1] * Q_BLOCK


    vh_seq_blks = (seqlens + VH_BLOCK - 1) // VH_BLOCK
    vh_cu_seq_blks = torch.cumsum(vh_seq_blks, 0)
    vh_offsets = torch.ones(vh_cu_seq_blks[-1], dtype = seqlens.dtype)
    vh_offsets[vh_cu_seq_blks[:-1]] = 1 - seq_blks[:-1]
    cu_vh_offsets = (torch.cumsum(torch.cumsum(vh_offsets, 0) - 1, 0)) * VH_BLOCK
    vh_size = cu_vh_offsets[-1].clone()
    cu_vh_offsets[vh_cu_seq_blks - 1] = -1

    return torch.cumsum(q_offsets, 0), cu_vh_offsets, vh_size


if __name__ == "__main__":
    torch.set_default_device('cuda:0')
    torch.set_printoptions(threshold=100000, linewidth=100000, sci_mode = False)
    #seqlen = torch.tensor([4096, 4096, 4096, 4096], dtype = torch.int)
    #q_offsets, vh_offsets, size = prepare_offsets(seqlen, 256, 64)
    #print(torch.arange(0, q_offsets.shape[-1]) * 256 + q_offsets, vh_offsets, size)
    M = torch.zeros((1, 128, 2, 32), dtype = torch.bfloat16)
    dism_C.test_tma(M, 0, 1, 0, 0)

    print(M.view(128, -1))