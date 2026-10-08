import torch

def direction_for_policy(random, counter, step, hybrid_index, policy, freeze_step, flip=False):
    if policy == 'random': out=random
    elif policy == 'true': out=torch.ones_like(random)
    elif policy == 'false': out=torch.zeros_like(random)
    elif policy == 'cycle8':
        out=(((counter >> hybrid_index) & 1) != 0).expand_as(random).contiguous()
    elif policy == 'late_true':
        out=torch.where(step >= freeze_step, torch.ones_like(random), random)
    else: raise ValueError(policy)
    return torch.logical_not(out) if flip else out
