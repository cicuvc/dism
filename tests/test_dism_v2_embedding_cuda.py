"""CUDA embedding E1: ordinary shared views, online statistics and tails."""
import pytest
import torch
import re
import subprocess
from dism_v2.embedding import forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(params=[False, True], ids=["single", "ws"])
def ws(request):
    return request.param


def oracle(x, key, value, scale):
    s = torch.einsum("bhnd,hvd->bhnv", x.float(), key.float()) * scale
    p = s.softmax(-1)
    return (torch.einsum("bhnv,hvd->bhnd", p, value.float()),
            s.logsumexp(-1), p.max(-1).values, s.argmax(-1).int())


@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("v", [1, 31, 32, 63, 64, 65, 127, 128, 129, 257])
def test_forward(d, v, ws, record_property, block_v=64):
    torch.manual_seed(d * 1000 + v)
    q, k = [torch.randn(2, 2, 65, d, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    eq, ek = [torch.randn(2, v, d, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    scale = d ** -0.5
    out = forward(q, k, eq, ek, scale, warp_specialized=ws, block_v=block_v)
    expected = (oracle(k, ek, eq, scale), oracle(q, eq, ek, scale))
    for direction in range(2):
        ref = expected[direction]
        actual = [out[direction], out[2+direction], out[4+direction], out[6+direction]]
        error = actual[0].double() - ref[0].double()
        record_property(f"direction{direction}_relative_l2", (error.norm()/ref[0].double().norm()).item())
        record_property(f"direction{direction}_cosine", torch.nn.functional.cosine_similarity(
            actual[0].double().flatten(), ref[0].double().flatten(), dim=0).item())
        torch.testing.assert_close(actual[0].float(), ref[0], atol=0.012, rtol=0.012)
        torch.testing.assert_close(actual[1], ref[1], atol=3e-5, rtol=3e-6)
        torch.testing.assert_close(actual[2], ref[2], atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(actual[3], ref[3], atol=0, rtol=0)


@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("n", [1, 15, 16, 17, 63, 64, 129])
def test_ties(d, n, ws, block_v=64):
    # All logits equal: select vocabulary index zero across tile boundaries.
    q = torch.zeros(1, 2, n, d, device="cuda", dtype=torch.bfloat16)
    e = torch.ones(2, 129, d, device="cuda", dtype=torch.bfloat16)
    out = forward(q, q, e, e, warp_specialized=ws, block_v=block_v)
    for x in out[:2]:
        torch.testing.assert_close(x, torch.ones_like(x), atol=0, rtol=0)
    for x in out[2:4]:
        torch.testing.assert_close(x, torch.full_like(x, torch.tensor(129.).log()), atol=1e-6, rtol=0)
    for x in out[4:6]:
        torch.testing.assert_close(x, torch.full_like(x, 1/129), atol=1e-8, rtol=0)
    for x in out[6:]:
        assert torch.count_nonzero(x) == 0


@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("scale", [-1., 0., 1.])
def test_cross_tile_argmax(d, scale, ws, block_v=64):
    q = torch.ones(1, 1, 17, d, device="cuda", dtype=torch.bfloat16)
    e = torch.zeros(1, 257, d, device="cuda", dtype=torch.bfloat16)
    e[:, 31] = e[:, 128] = -1 if scale < 0 else 1
    out = forward(q, q, e, e, scale, warp_specialized=ws, block_v=block_v)
    for x in out[6:]:
        assert (x == (0 if scale == 0 else 31)).all()


def test_codegen():
    from dism_v2.embedding import _extension
    so = _extension().__file__
    sass = subprocess.check_output(["/usr/local/cuda/bin/cuobjdump", "--dump-sass", so], text=True)
    assert not re.search(r"\bCALL(?:\.|\s)", sass)
    functions = re.split(r"Function\s*:\s*", sass)[1:]
    assert len(functions) == 8
    for body in functions:
        if "5fused" in body.splitlines()[0]:
            assert "UTMALDG" in body and "USETMAXREG.DEALLOC" in body and "USETMAXREG.TRY_ALLOC" in body
        assert not re.search(r"\b(?:LDL|STL)(?:\.|\s)", body), "unexpected spill/local traffic"


@pytest.mark.parametrize("d", [32, 64, 128])
def test_triton_comparison(d):
    from dism_v2.emb_kernel import emb_fwd_wrapper
    torch.manual_seed(881 + d)
    q,k = [torch.randn(1,2,129,d,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    eq,ek = [torch.randn(2,257,d,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    actual=forward(q,k,eq,ek,d**-.5,warp_specialized=True)
    baseline=forward(q,k,eq,ek,d**-.5,warp_specialized=False)
    triton=emb_fwd_wrapper(q,k,eq,ek,d**-.5)
    for x,y,z in zip(actual,baseline,triton):
        torch.testing.assert_close(x,y,atol=0,rtol=0)
        if x.dtype==torch.bfloat16:
            torch.testing.assert_close(x,z,atol=.008,rtol=.008)
        elif x.is_floating_point():
            torch.testing.assert_close(x,z,atol=3e-5,rtol=2e-5)
        else:
            torch.testing.assert_close(x,z,atol=0,rtol=0)


@pytest.mark.parametrize("d", [32,64,128])
@pytest.mark.parametrize("dv", [32,64,128])
@pytest.mark.parametrize("direction", ["q_from_k","k_from_q"])
@pytest.mark.parametrize("probability", [0.,.37,1.])
@pytest.mark.parametrize("oracle_kind", ["torch","same_embedding"])
def test_cuda_autograd_reference(d,dv,direction,probability,oracle_kind,record_property):
    from test_dism_v2_autograd import check_reference
    check_reference(d,dv,direction,probability,oracle_kind,record_property,embedding_backend="cuda")


@pytest.mark.parametrize("d", [32,64])
@pytest.mark.parametrize("v", [1,31,32,63,64,65,127,128,129,255,256,257,513,1025])
def test_bv128_forward(d,v,record_property):
    test_forward(d,v,True,record_property,block_v=128)


@pytest.mark.parametrize("d", [32,64])
@pytest.mark.parametrize("n", [1,15,16,17,63,64,129])
def test_bv128_ties(d,n):
    test_ties(d,n,True,block_v=128)


@pytest.mark.parametrize("d", [32,64])
@pytest.mark.parametrize("scale", [-1.,0.,1.])
def test_bv128_argmax(d,scale):
    test_cross_tile_argmax(d,scale,True,block_v=128)
