"""CPU batch/head ordering, reset/tau, numeric and ownership regression."""
import numpy as np
from prefill import HardPrefillPlan
from prefill_parallel import ParallelPrefillPlanner


def test():
    rng=np.random.default_rng(8164)
    q,k=rng.integers(0,4,(2,2,3,65),dtype=np.int32)
    reset=rng.random(q.shape)<.2
    taus=(np.array([0.,.7,4.2]),np.array([[0.,.7,4.2],[-.7,1e-9,2.]]))
    checks=0
    for workers in (1,2,4,8):
        with ParallelPrefillPlanner(workers) as pool:
            for tau in taus:
                batch=pool.plan(q,k,tau,reset=reset)
                for index,(p,chunks) in enumerate(batch):
                    b,h=divmod(index,3)
                    t=tau[h] if tau.ndim==1 else tau[b,h]
                    ref=HardPrefillPlan(q[b,h],k[b,h],t,reset=reset[b,h])
                    for name,array in p.arrays().items():
                        np.testing.assert_array_equal(array,ref.arrays()[name])
                    for name,array in chunks.items():
                        np.testing.assert_array_equal(array,ref.native.chunks(16)[name])
                    sq,sk=rng.normal(size=(2,65,3));v=rng.normal(size=(65,5))
                    np.testing.assert_array_equal(p.execute(sq,sk,v,dtype=np.float64),ref.execute(sq,sk,v,dtype=np.float64))
                    checks+=1
            try:pool.plan(q,k,np.zeros(2))
            except ValueError:pass
            else:raise AssertionError("invalid tau shape accepted")
            # Native worker exceptions propagate, and the pool stays usable.
            try:pool.plan(q,k,np.full(3,np.nan))
            except ValueError:pass
            else:raise AssertionError("invalid tau accepted")
            assert len(pool.plan(q,k,np.zeros(3)))==6
    print(f"PASS: {checks} head plans, ordering/tau/reset/FP64 output, exception recovery")


if __name__=="__main__":test()
