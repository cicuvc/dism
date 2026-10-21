"""Build the optional SM120 MXFP8 shared libraries."""
import argparse
from pathlib import Path

from . import tk_mxfp8_dgrad
from . import tk_mxfp8_gemm
from . import tk_mxfp8_swiglu
from . import tk_mxfp8_swiglu_dgrad
from . import tk_mxfp8_wgrad
from . import tk_silu_bwd
from . import tk_weight_quantize
from .core import ROOT


_TARGETS = (
    (tk_weight_quantize.OUTPUT if hasattr(tk_weight_quantize, "OUTPUT")
     else tk_weight_quantize.DEFAULT_OUTPUT, "libquant.so",
     tk_weight_quantize.build),
    (tk_mxfp8_gemm.OUTPUT, "libgemm.so", tk_mxfp8_gemm.build),
    (tk_mxfp8_dgrad.OUTPUT, "libdgrad.so", tk_mxfp8_dgrad.build),
    (tk_mxfp8_wgrad.OUTPUT, "libwgrad.so", tk_mxfp8_wgrad.build),
    (tk_mxfp8_swiglu.OUTPUT, "libswiglu.so", tk_mxfp8_swiglu.build),
    (tk_mxfp8_swiglu_dgrad.OUTPUT, "libswd.so", tk_mxfp8_swiglu_dgrad.build),
)


def build_all(*, force=False):
    sources = [
        *ROOT.glob("cuda/*"),
        *ROOT.glob("third_party/thunderkittens/include/**/*"),
    ]
    newest_source = max(
        (path.stat().st_mtime_ns for path in sources if path.is_file()),
        default=0,
    )
    for output, library, build in _TARGETS:
        path = Path(output) / library
        if force or not path.is_file() or path.stat().st_mtime_ns < newest_source:
            build()
    silu_libraries = (
        tk_silu_bwd.OUTPUT / "libsilu_bwd.so",
        tk_silu_bwd.OUTPUT / "libsilu_bwd_quantize.so",
    )
    if force or any(
        not path.is_file() or path.stat().st_mtime_ns < newest_source
        for path in silu_libraries
    ):
        tk_silu_bwd.build()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_all(force=args.force)


if __name__ == "__main__":
    main()
