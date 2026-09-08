"""Experimental WS path; retain strict dV thresholds and visible precision failures."""
import itertools
import pytest
from test_dism_v2_dv import run, exact_oracle_matmul, pytestmark


@pytest.mark.parametrize("d,dv",itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("probability",(0.,.37,1.))
def test_ws_dimensions(d,dv,direction,probability,record_property):
    run(d,dv,139,direction,probability,record_property,check_summary=True,warp_specialized=True)


@pytest.mark.parametrize("n",(1,17,31,32,63,64,65,128,129,257,513))
def test_ws_tails(n,record_property):
    run(64,128,n,"random",.63,record_property,check_summary=True,warp_specialized=True)


@pytest.mark.parametrize("mode",("chain","break","bounded_soft"))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("n",(1025,8193))
def test_ws_long(mode,direction,n,record_property):
    run(64,128,n,direction,0. if mode=="bounded_soft" else 1.,record_property,mode,warp_specialized=True)


@pytest.mark.parametrize("n",(2049,8193))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("probability",(0.,.37))
def test_ws_long_soft(n,direction,probability,record_property):
    run(64,128,n,direction,probability,record_property,"bounded_soft",warp_specialized=True)


@pytest.mark.parametrize("n",(1025,2049))
@pytest.mark.parametrize("mode",("chain","break","bounded_soft"))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
def test_ws_long_summary(n,mode,direction,record_property):
    # dV alone does not exercise accuracy of the approximate sigmoid/passing.
    run(64,128,n,direction,0. if mode=="bounded_soft" else 1.,record_property,
        mode,check_summary=True,warp_specialized=True)
