"""Create a private fused readout module; production launch defaults unchanged."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();root=Path(__file__).resolve().parents[1]
    src=root/'python/flash_dism/kernels/hard_backward_fused.py'
    extra=Path(__file__).with_name('single_key_readout.py').read_text()
    extra=extra[extra.index('@tr.jit'):]
    source=src.read_text()
    source=source.replace('@tr.jit\ndef _single_pair',extra+'\n\n@tr.jit\ndef _single_pair')
    old='''    else:
        K: tl.constexpr = C if BC == 0 else BC'''
    new='''    else:
        if CHUNKS is None and COMPENSATED:
            count_t = tl.arange(0, C if BC == 0 else BC)
            count_rows = tl.load(ROWS + chunk*C + count_t)
            single_key = (row0 >= 0) & (length > 2) & (tl.sum(((count_t < length) & (count_rows >= 0)).to(tl.int32), 0) == 1)
        else:
            single_key = False
            length = 0
        if single_key:
            _single_key_readout(SQ, SK, V, DO, DSQ, DSK, DV_OUT, TAU, ROWS, PREFIX,
                                WEIGHT, DLOG, chunk*C, stream, row0, length,
                                R, DV, C if BC == 0 else BC, BR, BD)
        else:
            K: tl.constexpr = C if BC == 0 else BC'''
    assert source.count(old)==1
    start=source.index(old);end=source.index('\n\n@tr.jit\ndef _chunk_summaries',start)
    block=source[start+len(old):end]
    # Nest the existing general implementation unchanged in the fallback.
    source=source[:start]+new+'\n'.join(('    '+line if line else line) for line in block.split('\n'))+source[end:]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(source)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(dict(source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
