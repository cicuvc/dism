"""Experimental single-key/multiple-query backward; appended to copied kernels."""
# The builder prepends this function to a private hard_backward_fused module.
import triton as tr
import triton.language as tl

@tr.jit
def _single_key_readout(SQ, SK, V, DO, DSQ, DSK, DV_OUT, TAU, ROWS, PREFIX,
                        WEIGHT, DLOG, pos, stream, ki, length,
                        R: tl.constexpr, DV: tl.constexpr, C: tl.constexpr,
                        BR: tl.constexpr, BD: tl.constexpr):
    t,r,d=tl.arange(0,C),tl.arange(0,BR),tl.arange(0,BD)
    row=tl.load(ROWS+pos+t)
    query=(t<length)&(t>0)
    qi=tl.where(query,-row-1,0)
    sk=tl.load(SK+ki*R+r,r<R,0).to(tl.float32)
    v=tl.load(V+ki*DV+d,d<DV,0).to(tl.float32)
    sq=tl.load(SQ+qi[:,None]*R+r[None,:],query[:,None]&(r[None,:]<R),0).to(tl.float32)
    do=tl.load(DO+qi[:,None]*DV+d[None,:],query[:,None]&(d[None,:]<DV),0).to(tl.float32)
    p=tl.load(PREFIX+pos+t)
    p0=tl.load(PREFIX+pos)
    w=tl.load(WEIGHT+pos+t)
    w0=tl.load(WEIGHT+pos)
    pair=tl.where(query,tl.exp(tl.minimum(p-p0,0.))*w*w0,0.)
    ds=tl.sum(do*v[None,:],1)*pair
    score=tl.sum(sq*sk[None,:],1)
    dw=tl.load(DLOG+pos+t)+tl.load(DLOG+pos)
    tau=tl.sum(ds*score*dw,0)
    # Match the local BF16 tensor-core path's vector rounding sites.
    ds=ds.to(tl.bfloat16).to(tl.float32)
    sp=(score*pair).to(tl.bfloat16).to(tl.float32)
    tl.atomic_add(DSQ+qi[:,None]*R+r[None,:],ds[:,None]*sk[None,:],
                  query[:,None]&(r[None,:]<R),sem="relaxed")
    dsk=tl.sum(ds[:,None]*sq,0)
    dv=tl.sum(sp[:,None]*do,0)
    tl.atomic_add(DSK+ki*R+r,dsk,r<R,sem="relaxed")
    tl.atomic_add(DV_OUT+ki*DV+d,dv,d<DV,sem="relaxed")
    tl.store(TAU+stream,tau)
