#!/usr/bin/env bash
set -euo pipefail
cuda_root=${CUDA_ROOT:-/usr/local/cuda}
glx_root=${GLX_ROOT:-/home/cicuvc/cs/projects/glx}
probe_build=$(mktemp -d /tmp/dism-glx-boundaries.XXXXXX)
echo "Results: $probe_build"
"$cuda_root/bin/nvcc" -std=c++20 -O3 -arch=sm_120a -lineinfo \
    -I"$glx_root/include" --ptxas-options=-v \
    experiments/glx_boundaries/boundaries.cu -o "$probe_build/boundaries" \
    > "$probe_build/build.log" 2>&1
"$probe_build/boundaries" | tee "$probe_build/run.log"
"$cuda_root/bin/cuobjdump" --dump-sass "$probe_build/boundaries" > "$probe_build/sass.txt"
"$cuda_root/bin/cuobjdump" --dump-resource-usage "$probe_build/boundaries" > "$probe_build/resources.txt"
for checker in memcheck racecheck synccheck; do
    "$cuda_root/bin/compute-sanitizer" --tool "$checker" --error-exitcode 1 \
        "$probe_build/boundaries" > "$probe_build/$checker.log" 2>&1
    tail -2 "$probe_build/$checker.log"
done
"${DISM_PYTHON:-/home/cicuvc/miniconda3/envs/blkw/bin/python}" \
    experiments/glx_boundaries/check_codegen.py "$probe_build"
