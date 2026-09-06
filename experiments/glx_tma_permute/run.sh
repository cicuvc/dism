#!/usr/bin/env bash
set -euo pipefail
cuda_root=${CUDA_ROOT:-/usr/local/cuda}
glx_root=${GLX_ROOT:-/home/cicuvc/cs/projects/glx}
probe_build=$(mktemp -d /tmp/dism-glx-tma.XXXXXX)
echo "Results: $probe_build"
"$cuda_root/bin/nvcc" -std=c++20 -O3 -arch=sm_120 -lineinfo \
    --extended-lambda --expt-relaxed-constexpr \
    -Iinclude -I"$glx_root/include" --ptxas-options=-v \
    experiments/glx_tma_permute/tma_mma_permute.cu \
    -lcuda -o "$probe_build/tma_mma_permute" \
    > "$probe_build/build.log" 2>&1
"$probe_build/tma_mma_permute" | tee "$probe_build/run.log"
"$cuda_root/bin/compute-sanitizer" --tool memcheck --error-exitcode 1 \
    "$probe_build/tma_mma_permute" > "$probe_build/memcheck.log" 2>&1
tail -1 "$probe_build/memcheck.log"
"$cuda_root/bin/cuobjdump" --dump-resource-usage \
    "$probe_build/tma_mma_permute" > "$probe_build/resource_usage.log"
