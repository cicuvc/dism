"""Exact register ownership and bank mapping for the OPT9 dA output gather."""

def test_float4_output_ownership_and_banks():
    def source(lane, r):
        row=lane//4+(r%2)*8
        col=2*(lane%4)+(r//2)*8
        return [(row,col),(row,col+1)]

    writes=[]
    for h in (0,1):
        wavefront_banks=[[] for _ in range(4)]
        for lane in range(32):
            g=lane%4
            own=source(lane,h+2*(g%2))
            peer_lane=lane^1
            peer=source(peer_lane,h+2*((peer_lane^1)%2))
            values=peer+own if g%2 else own+peer
            row=8*h+lane//4
            col=4*(g//2)+8*(g%2)
            expected=[(row,col+i) for i in range(4)]
            assert values==expected
            assert (row*16+col)%4==0 # STS.128 alignment.
            writes.extend(expected)
            wavefront_banks[lane//8].extend((r*16+c)%32 for r,c in expected)
        for banks in wavefront_banks:
            assert sorted(banks)==list(range(32))
    assert sorted(writes)==[(r,c) for r in range(16) for c in range(16)]
