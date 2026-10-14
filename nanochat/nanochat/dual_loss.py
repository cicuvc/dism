"""Full-batch soft/hard losses; optional antithetic direction pair."""
import torch

def dual_loss_backward(model, original, x, y, cu, segments, alpha, divisor=1, scaler=None,
                       *, document_step=None):
    if not 0 <= alpha <= 1: raise ValueError(alpha)
    counter=original.rng_counter.detach().clone();original.rng_counter.add_(1)
    flips=(False,True) if original.config.direction_antithetic else (False,)
    values={}
    numerator = None
    for hard,weight in ((False,1-alpha),(True,alpha)):
        if weight == 0: continue
        key='hard' if hard else 'soft'
        for flip in flips:
            options = {'loss_reduction': 'document_sum'} if document_step is not None else {}
            result=model(x,y,cu_seqlens=cu,segment_ids=segments,
                hard_mode=hard,paired_counter=counter,direction_flip=flip, **options)
            if document_step is not None:
                loss, count = result
                contribution = loss.detach()*(weight/len(flips))
                numerator = contribution if numerator is None else numerator+contribution
                report = loss.detach()/count.clamp_min(1)
            else:
                loss = result
                report = loss.detach()
            values[key]=values.get(key,0)+report/len(flips)
            scaled=loss*(weight/(divisor*len(flips)))
            if scaler is None: scaled.backward()
            else: scaler.scale(scaled).backward()
            del loss,scaled
    if document_step is not None:
        # Same labels/documents on both paths: count once, not once per branch.
        document_step.add(numerator, count)
    mixed=sum(values[k]*w for k,w in [('soft',1-alpha),('hard',alpha)] if k in values)
    return mixed,values
