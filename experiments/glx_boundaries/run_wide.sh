#!/usr/bin/env bash
set -euo pipefail
cuda_root=${CUDA_ROOT:-/usr/local/cuda}
glx_root=${GLX_ROOT:-/home/cicuvc/cs/projects/glx}
probe_build=$(mktemp -d /tmp/dism-glx-wide.XXXXXX)
echo "Results: $probe_build"
for kind in scan reduce; do
    "$cuda_root/bin/nvcc" -std=c++20 -O3 -arch=sm_120a \
        -I"$glx_root/include" -DGLX_TEST_WIDE_ROWS=16 -DGLX_TEST_WIDE_COLS=128 \
        "$glx_root/tests/diagonal_${kind}_test.cu" -o "$probe_build/$kind" \
        > "$probe_build/$kind.build.log" 2>&1
    "$probe_build/$kind" | tee "$probe_build/$kind.run.log"
    "$cuda_root/bin/compute-sanitizer" --tool memcheck --error-exitcode 1 \
        "$probe_build/$kind" > "$probe_build/$kind.memcheck.log" 2>&1
    tail -1 "$probe_build/$kind.memcheck.log"
done
