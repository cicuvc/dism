#!/usr/bin/env bash
set -euo pipefail
cuda_root=${CUDA_ROOT:-/usr/local/cuda}
glx_root=${GLX_ROOT:-/home/cicuvc/cs/projects/glx}
reverse_build=$(mktemp -d /tmp/dism-reverse.XXXXXX)
echo "Results: $reverse_build"
"$cuda_root/bin/nvcc" -std=c++20 -O3 -arch=sm_120a -I"$glx_root/include" \
    experiments/glx_reverse/reverse.cu -lcuda -o "$reverse_build/reverse"
"$reverse_build/reverse" | tee "$reverse_build/run.log"
"$cuda_root/bin/cuobjdump" --dump-sass "$reverse_build/reverse" > "$reverse_build/sass.txt"
"$cuda_root/bin/cuobjdump" --dump-resource-usage "$reverse_build/reverse" > "$reverse_build/resources.txt"
for checker in memcheck racecheck synccheck; do
    "$cuda_root/bin/compute-sanitizer" --tool "$checker" --error-exitcode 1 \
        "$reverse_build/reverse" > "$reverse_build/$checker.log" 2>&1
    tail -2 "$reverse_build/$checker.log"
done
