import torch
import math
import matplotlib.pyplot as plt

def sinkhorn(m: torch.Tensor, N: int = 10):
    for i in range(N):
        m = torch.nn.functional.normalize(m, 1, dim = 0)
        m = torch.nn.functional.normalize(m, 1, dim = 1)
    return m

def sample_gumbel(logits):
    return  -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()

torch.set_default_device('cuda:0')

BATCH = 32
CLS = 64
RANK = 16

X = torch.nn.Parameter(torch.randn((CLS, RANK), dtype = torch.float32), True)
Y = torch.nn.Parameter(torch.randn((RANK, CLS), dtype = torch.float32), True)
target_perm = torch.randperm(CLS)

optim = torch.optim.Muon([X, Y], lr=1e-3)

tau_min = 0.2
tau_max = 1.0
tau_T = 7000

running_avg = 0.0

torch.set_printoptions(threshold=10000, linewidth=10000)

for step in range(10000):
    
    tau = tau_min + (tau_max - tau_min) * (((1+math.cos((step / tau_T) * math.pi)) / 2) if step < tau_T else 0)

    q = torch.randint(0, CLS, (BATCH, ))
    k_target = target_perm[q]

    q_vecs = torch.nn.functional.one_hot(q, CLS).float()
    k_vecs = torch.nn.functional.one_hot(k_target, CLS).float()

    P = X @ Y
    noise = sample_gumbel(P)
    W = sinkhorn(torch.softmax((P + noise) / tau, dim = -1))

    L = - torch.log(torch.diag(q_vecs @ W @ k_vecs.T)).mean()

    optim.zero_grad(True)
    L.backward()
    torch.nn.utils.clip_grad_norm_([X, Y], 1)
    optim.step()

    running_avg += L.item()

    if step % 100 == 99:
        print(f"Loss = {(running_avg / 100):.5}, tau = {tau:.5}")
        running_avg = 0

    if step % 100 == 99:
        fig = plt.figure(1)
        ax = fig.subplots(1,1)
        im = ax.imshow(W.detach().cpu().numpy(), 'viridis')
        fig.colorbar(im, ax = ax)
        fig.savefig('p.png')
        fig.clear()

    if step == 2000:
        target_perm = torch.randperm(CLS)