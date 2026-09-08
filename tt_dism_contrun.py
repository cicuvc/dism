import torch
from tt_dism import ParallelSoftDiscreteAttention
from tqdm import tqdm

if __name__ == "__main__":
    torch.set_default_device('cuda:0')

    BATCH = 2
    HEADS = 16
    N = 4096
    N_HEADDIM = 64
    N_VOCAB = 64
    seps = 1e-4

    betas = torch.randn((BATCH, HEADS, N), dtype = torch.float32).requires_grad_(True)
    rcptaus = torch.nn.Parameter(torch.full((HEADS,), 3.0, dtype = torch.float32), requires_grad=True)
    values = torch.randn((BATCH, HEADS, N, N_HEADDIM), dtype = torch.bfloat16).requires_grad_(True)
    logq, logk = torch.randn((BATCH, HEADS, N, N_VOCAB), dtype = torch.bfloat16).requires_grad_(True), torch.randn((BATCH, HEADS, N, N_VOCAB), dtype = torch.bfloat16).requires_grad_(True)

    for i in tqdm(range(10000)):
        o = ParallelSoftDiscreteAttention.apply(logq, logk, values, betas, rcptaus)
        do = torch.randn_like(o)
        #o.backward(do.to(o.dtype))

    exit(0)