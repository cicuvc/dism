"""Validate BNHD integration against the actual v4 dense Torch oracle."""
import importlib.util
from pathlib import Path
import sys
import argparse
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dism_v4.decoding import HardDismDecoder

path=Path(__file__).resolve().parents[1]/'python/flash_dism/reference/dism_v4_ref.py'
spec=importlib.util.spec_from_file_location('independent_v4_oracle',path)
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--planner',choices=['cpu','gpu'],default='cpu')
    args=parser.parse_args()
    torch.manual_seed(91)
    for dtype in (torch.float32,torch.bfloat16):
        b,n,h,r,d=2,73,3,16,32
        sk=torch.randn(b,n,h,r,device='cuda').to(dtype);sq=torch.randn_like(sk)
        v=torch.randn(b,n,h,d,device='cuda').to(dtype)
        iq=torch.randint(0,4,(b,h,n),device='cuda',dtype=torch.int32);ik=torch.randint_like(iq,0,4)
        reset=torch.rand(b,h,n,device='cuda')<.07
        tau=torch.tensor([0.,1e-8,.7],device='cuda')
        dummy=torch.zeros(b,n,h,32,device='cuda',dtype=torch.float64)
        lse=torch.zeros(b,n,h,device='cuda',dtype=torch.float64)
        expected=oracle.dism_ref(dummy,dummy,sq.double(),sk.double(),lse,lse,iq,ik,
            torch.ones(b,h,device='cuda',dtype=torch.bool),torch.ones_like(iq,dtype=torch.bool),
            torch.where(reset,torch.inf,0.).double(),v.double(),tau.double())
        def create():return HardDismDecoder(b,h,r,d,n,tau,cache_dtype=dtype,
            planner_backend=args.planner,
            rebuild_interval=11,sample_interval=5,materialize_threshold=23,rebuild_chunk=32)
        decoder=create()
        actual=decoder.append(iq,ik,sq,sk,v,reset=reset)
        torch.testing.assert_close(actual.double(),expected,atol=3e-4,rtol=3e-4)
        decoder=create();cut=37
        decoder.prime(iq[:,:,:cut],ik[:,:,:cut],sk[:,:cut],v[:,:cut],reset=reset[:,:,:cut])
        actual=decoder.append(iq[:,:,cut:],ik[:,:,cut:],sq[:,cut:],sk[:,cut:],v[:,cut:],reset=reset[:,:,cut:])
        torch.testing.assert_close(actual.double(),expected[:,cut:],atol=3e-4,rtol=3e-4)
        before=decoder.memory_stats()['position']
        try:decoder.append(iq[:,:,:1],ik[:,:,:1],sq[:,:1],sk[:,:1],v[:,:1])
        except ValueError:pass
        else:raise AssertionError('capacity overflow not rejected')
        assert decoder.memory_stats()['position']==before
        print('API_PASS',args.planner,dtype,flush=True)


if __name__=='__main__':main()
