from pathlib import Path
import subprocess

import pytest
import torch

from flash_dism.forward import forward_core
from flash_dism.backward import backward_core
from flash_dism.varlen import forward_varlen,backward_varlen
from test_varlen_layout import make_layout
from test_varlen_summary import packed_inputs
from test_varlen_forward import slice_inputs


def test_no_operand_pack_or_unpack(monkeypatch):
    import cu_flash_dism as cu
    layout,inputs=packed_inputs([256,512])
    inputs=(*inputs[:7],inputs[7].int(),inputs[8].int(),*inputs[9:])
    def forbidden(*args,**kwargs):
        raise AssertionError('production must not launch pack/unpack')
    monkeypatch.setattr(cu,'varlen_pack',forbidden,raising=False)
    monkeypatch.setattr(cu,'varlen_unpack',forbidden,raising=False)
    output,_,state=forward_varlen(*inputs,layout=layout,save_state=True)
    assert layout.tokens==layout.padded_tokens
    for source,saved in zip(inputs[:5],state['operands'][:5]):
        assert source.data_ptr()==saved.data_ptr()
        assert source.numel()==saved.numel()
    for source,saved in zip(inputs[7:9],state['operands'][7:9]):
        assert source.data_ptr()==saved.data_ptr()
    assert state['lse2'].data_ptr()==state['normalizer'].data_ptr()
    dout=torch.randn_like(output).bfloat16()
    assert layout.pack(dout,vectors=True).data_ptr()==dout.data_ptr()
    backward_varlen(state,dout)


def test_head_major_metadata_with_nonuniform_lse():
    layout,inputs=packed_inputs([0,256,512,0,256])
    # Different rows/heads/documents must not accidentally share a scalar base.
    for lse in inputs[5:7]:
        lse.add_(torch.rand_like(lse))
    output,norm,state=forward_varlen(*inputs,layout=layout,save_state=True)
    dout=torch.randn_like(output).bfloat16()
    actual=backward_varlen(state,dout)
    tau=torch.zeros_like(inputs[-1])
    for start,n,*_ in layout.table.tolist():
        if not n:
            continue
        expected,expected_norm,saved=forward_core(*slice_inputs(inputs,start,n),save_state=True)
        torch.testing.assert_close(output[:,start:start+n],expected,atol=0,rtol=0)
        torch.testing.assert_close(norm[:,:,start:start+n],expected_norm,atol=0,rtol=0)
        gradients=backward_core(saved,dout[:,start:start+n].contiguous())
        tau+=gradients.pop('rtau')
        for name,value in gradients.items():
            torch.testing.assert_close(actual[name][:,start:start+n],value,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(actual['rtau'],tau,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('heads',[1,2,8])
def test_heads_and_stream(heads):
    layout,inputs=packed_inputs([256,512,768,256],heads=heads)
    baseline,_,saved=forward_varlen(*inputs,layout=layout,save_state=True)
    dout=torch.randn_like(baseline).bfloat16()
    expected=backward_varlen(saved,dout)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output,_,state=forward_varlen(*inputs,layout=layout,save_state=True)
        gradients=backward_varlen(state,dout)
    stream.synchronize()
    torch.testing.assert_close(output,baseline,atol=0,rtol=0)
    for name,value in expected.items():
        torch.testing.assert_close(gradients[name],value,atol=2e-6,rtol=2e-5)


def test_long_documents():
    layout,inputs=packed_inputs([256,1024,2048,4096],heads=1)
    output,norm,state=forward_varlen(*inputs,layout=layout,save_state=True)
    dout=torch.randn_like(output).bfloat16()
    gradient=backward_varlen(state,dout)
    tau=torch.zeros_like(inputs[-1])
    for start,n,*_ in layout.table.tolist():
        ref,ref_norm,saved=forward_core(*slice_inputs(inputs,start,n),save_state=True)
        torch.testing.assert_close(output[:,start:start+n],ref,atol=0,rtol=0)
        torch.testing.assert_close(norm[:,:,start:start+n],ref_norm,atol=2e-5,rtol=2e-6)
        expected=backward_core(saved,dout[:,start:start+n].contiguous())
        for name,value in expected.items():
            if name=='rtau':
                tau+=value
            else:
                torch.testing.assert_close(gradient[name][:,start:start+n],value,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(gradient['rtau'],tau,atol=2e-4,rtol=2e-5)


def test_document_permutation():
    layout,inputs=packed_inputs([256,512,768,256])
    output,_,state=forward_varlen(*inputs,layout=layout,save_state=True)
    dout=torch.randn_like(output).bfloat16()
    original=backward_varlen(state,dout)
    order=[2,0,3,1]
    indices=torch.cat([torch.arange(layout.table[s,0],layout.table[s,0]+layout.lengths[s],
                                    device='cuda') for s in order]).long()
    changed=[x[:,indices].contiguous() for x in inputs[:7]]
    changed.extend([inputs[7][:,:,indices].contiguous(),inputs[8][:,:,indices].contiguous(),
                    inputs[9],inputs[10][:,:,indices].contiguous(),inputs[11]])
    permuted=make_layout([layout.lengths[s] for s in order])
    out,_,saved=forward_varlen(*changed,layout=permuted,save_state=True)
    gradient=backward_varlen(saved,dout[:,indices].contiguous())
    torch.testing.assert_close(out,output[:,indices],atol=0,rtol=0)
    for name,value in original.items():
        expected=value if name=='rtau' else value[:,indices]
        torch.testing.assert_close(gradient[name],expected,atol=2e-6,rtol=2e-5)


def test_varlen_sass():
    root=Path(__file__).resolve().parents[1]
    names=['summary','forward','chunk','backward_summary','backward_qk','delta']
    if hasattr(__import__('cu_flash_dism'),'varlen_pack'):
        names+=['pack','tma_probe']
    for name in names:
        binary=root/f'build/object/r32_d64_v64/varlen_{name}.cu.dev.sm120a.o'
        sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','-sass',str(binary)],text=True)
        assert 'CALL' not in sass,name
        if name in ('summary','forward','chunk'):
            assert 'PackedArgs' in sass,name
            assert 'PK11ChunkRecord' not in sass,name
        if name in ('backward_summary','backward_qk'):
            assert 'UTMAREDG' in sass,name
