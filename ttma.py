import sys
sys.path.append('build/linux/x86_64/release')

import torch
import dism_C

if __name__ == "__main__":
    torch.set_printoptions(threshold=100000, precision=4, linewidth=100000, sci_mode=False)

    B, N, H = 1, 128, 1
    C = 32
    q = torch.randn((B, N, H, C), dtype = torch.bfloat16, device = 'cuda:0')

    b, s, h = 0,64,0

    print(q[b,:,h])
    dism_C.testTMA(q, b, s, h)
    