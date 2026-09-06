#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. External GLX remains unchanged.
glx_root=${GLX_ROOT:-/home/cicuvc/cs/projects/glx}
cuda_root=${CUDA_ROOT:-/usr/local/cuda}
probe_build=$(mktemp -d /tmp/dism-glx-scan.XXXXXX)
echo "Results: $probe_build"
for probe in scan_compat reduce_compat log_compat resources; do
    "$cuda_root/bin/nvcc" -std=c++20 -O3 -arch=sm_120 -lineinfo \
        -I"$glx_root/include" -I"$glx_root/tests" --ptxas-options=-v \
        "experiments/glx_scan/$probe.cu" -o "$probe_build/$probe" \
        > "$probe_build/$probe.build.log" 2>&1
    "$probe_build/$probe" | tee "$probe_build/$probe.run.log"
    "$cuda_root/bin/compute-sanitizer" --tool memcheck --error-exitcode 1 \
        "$probe_build/$probe" > "$probe_build/$probe.memcheck.log" 2>&1
    tail -1 "$probe_build/$probe.memcheck.log"
done
"$cuda_root/bin/cuobjdump" --dump-sass "$probe_build/resources" > "$probe_build/resources.sass"
"$cuda_root/bin/cuobjdump" --dump-resource-usage "$probe_build/resources" > "$probe_build/resources.usage"
