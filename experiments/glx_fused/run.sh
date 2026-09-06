#!/usr/bin/env bash
set -euo pipefail
cuda_root=${CUDA_ROOT:-/usr/local/cuda}
glx_root=${GLX_ROOT:-/home/cicuvc/cs/projects/glx}
probe_build=$(mktemp -d /tmp/dism-glx-fused.XXXXXX)
echo "Results: $probe_build"
"$cuda_root/bin/nvcc" -std=c++20 -O3 -arch=sm_120 -lineinfo \
    --extended-lambda --expt-relaxed-constexpr \
    -Iinclude -I"$glx_root/include" --ptxas-options=-v \
    experiments/glx_fused/fused.cu -lcuda -o "$probe_build/fused" \
    > "$probe_build/build.log" 2>&1
"$probe_build/fused" | tee "$probe_build/run.log"
for tool in memcheck racecheck synccheck; do
    "$cuda_root/bin/compute-sanitizer" --tool "$tool" --error-exitcode 1 \
        "$probe_build/fused" > "$probe_build/$tool.log" 2>&1
    tail -3 "$probe_build/$tool.log"
done
"$cuda_root/bin/cuobjdump" --dump-resource-usage "$probe_build/fused" > "$probe_build/resources.log"
"$cuda_root/bin/nvcc" -std=c++20 -O3 -arch=sm_120 -lineinfo \
    -I"$glx_root/include" --ptxas-options=-v \
    experiments/glx_fused/checkpoint.cu -o "$probe_build/checkpoint" \
    > "$probe_build/checkpoint_build.log" 2>&1
"$probe_build/checkpoint" | tee "$probe_build/checkpoint_run.log"
for tool in memcheck racecheck synccheck; do
    "$cuda_root/bin/compute-sanitizer" --tool "$tool" --error-exitcode 1 \
        "$probe_build/checkpoint" > "$probe_build/checkpoint_$tool.log" 2>&1
    tail -3 "$probe_build/checkpoint_$tool.log"
done
