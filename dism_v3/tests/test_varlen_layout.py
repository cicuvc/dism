import pytest
import torch

from flash_dism.varlen import VarlenLayout


def make_layout(lengths, device='cpu'):
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1]+n)
    return VarlenLayout.from_cu_seqlens(
        torch.tensor(offsets, dtype=torch.int32, device=device), offsets[-1])


def test_checkpoint_capacity():
    layout = make_layout([2048]*32)
    assert layout.checkpoint_bytes(8) == 764*1024**2
    assert make_layout([65536]).checkpoint_bytes(8) == 24572*1024**2


@pytest.mark.parametrize('lengths', [[], [0,0], [256], [256,512,768],
                                   [0,256,1024,512,0]])
@pytest.mark.parametrize('vectors', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16,torch.float32,torch.int32,torch.uint8])
def test_packing(lengths, vectors, dtype):
    layout = make_layout(lengths, 'cuda')
    h, c = 3, 32
    shape = (1, layout.tokens, h, c) if vectors else (1, h, layout.tokens)
    x = torch.arange(torch.tensor(shape).prod().item(), device='cuda')
    x = (x.remainder(113)-55).to(dtype).reshape(shape)
    actual = layout.pack(x, vectors=vectors)
    assert layout.padded_tokens == layout.tokens
    if vectors:
        assert actual.data_ptr() == x.data_ptr()
    expected = []
    for begin, n, _, p, *_ in layout.table.tolist():
        if vectors:
            item = torch.zeros((p,h,c), device='cuda', dtype=dtype)
            item[:n] = x[0,begin:begin+n]
        else:
            item = torch.zeros((h,p), device='cuda', dtype=dtype)
            item[:,:n] = x[0,:,begin:begin+n]
            item = item.flatten()
        expected.append(item)
    ref = torch.cat(expected) if expected else torch.empty_like(actual)
    torch.testing.assert_close(actual, ref, atol=0, rtol=0)


@pytest.mark.parametrize('offsets,total', [([1,4],4), ([0,4,3,5],5), ([0,3],4), ([],0),
                                          ([0,128,512],512),([0,256,257],257)])
def test_bad_offsets(offsets,total):
    with pytest.raises(RuntimeError):
        VarlenLayout.from_cu_seqlens(torch.tensor(offsets,dtype=torch.int32),total)


def test_lowlevel_checks():
    layout = make_layout([256,512])
    x = torch.zeros((1,768,2,64),dtype=torch.bfloat16,device='cuda')
    table = layout.table.clone()
    table[1,2] += 1
    import cu_flash_dism as cu
    with pytest.raises(RuntimeError,match='inconsistent'):
        cu.varlen_pack(x,table,True)
    with pytest.raises(RuntimeError,match='token count'):
        cu.varlen_pack(x[:,:17].contiguous(),layout.table,True)
    assert cu.varlen_pack(x,layout.table,True).data_ptr() == x.data_ptr()
    table=layout.table.clone()
    table[0,1]=255
    with pytest.raises(RuntimeError,match='aligned'):
        cu.varlen_pack(x,table,True)


@pytest.mark.parametrize('rows',[32,64])
@pytest.mark.parametrize('heads',[1,3])
def test_device_descriptor_array(rows,heads):
    import cu_flash_dism as cu
    layout = make_layout([0,256,512,768,0,1024], 'cuda')
    x = torch.arange(layout.tokens*heads*64,device='cuda').reshape(1,layout.tokens,heads,64)
    x = x.remainder(127).to(torch.bfloat16)
    packed = layout.pack(x,vectors=True)
    # Direct aliases have no padding. A deliberate load beyond a document's
    # extent must zero-fill, not consume the neighboring document.
    expected = []
    for s,(_,n,start,p,*_) in enumerate(layout.table.tolist()):
        if not n:
            continue
        expected.extend([packed[start:start+p],torch.zeros((rows,heads,64),device='cuda',dtype=x.dtype)])
    actual = cu.varlen_tma_probe(packed,layout.table,rows)
    torch.testing.assert_close(actual,torch.cat(expected),atol=0,rtol=0)


@pytest.mark.parametrize('mode',[0,1,2])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_scalar_unpack(mode,dtype):
    import cu_flash_dism as cu
    layout=make_layout([0,256,512,256,768])
    h=3
    expected=torch.arange(h*layout.tokens,device='cuda').reshape(1,h,layout.tokens).to(dtype)
    source=torch.full(((layout.padded_tokens if mode else layout.tokens)*h,),
                      -19,device='cuda',dtype=dtype)
    for start,n,pstart,p,*_ in layout.table.tolist():
        base=(pstart if mode else start)*h
        pitch=p if mode==1 else n
        source[base:base+pitch*h].view(1,h,pitch)[...,:n]=expected[...,start:start+n]
    torch.testing.assert_close(cu.varlen_unpack(source,layout.table,h,mode),expected,atol=0,rtol=0)
