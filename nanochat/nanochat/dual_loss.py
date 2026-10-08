"""Full-batch soft/hard losses; optional antithetic direction pair."""
import torch

def dual_loss_backward(model, original, x, y, cu, segments, alpha, divisor=1, scaler=None):
    if not 0 <= alpha <= 1: raise ValueError(alpha)
    counter=original.rng_counter.detach().clone();original.rng_counter.add_(1)
    flips=(False,True) if original.config.direction_antithetic else (False,)
    values={}
    for hard,weight in ((False,1-alpha),(True,alpha)):
        if weight == 0: continue
        key='hard' if hard else 'soft'
        for flip in flips:
            loss=model(x,y,cu_seqlens=cu,segment_ids=segments,
                hard_mode=hard,paired_counter=counter,direction_flip=flip)
            values[key]=values.get(key,0)+loss.detach()/len(flips)
            scaled=loss*(weight/(divisor*len(flips)))
            if scaler is None: scaled.backward()
            else: scaler.scale(scaled).backward()
            del loss,scaled
    mixed=sum(values[k]*w for k,w in [('soft',1-alpha),('hard',alpha)] if k in values)
    return mixed,values
