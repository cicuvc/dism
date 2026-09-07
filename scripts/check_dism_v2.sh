#!/usr/bin/env bash
set -euo pipefail
dism_python=${DISM_PYTHON:-/home/cicuvc/miniconda3/envs/blkw/bin/python}
cuda_root=${CUDA_ROOT:-/usr/local/cuda}
check_dir=$(mktemp -d /tmp/dism-v2-check.XXXXXX)
echo "Results: $check_dir"
"$dism_python" -m pytest -q -p no:cacheprovider \
    tests/test_dism_v2_reference.py tests/test_dism_v2_build.py \
    tests/test_dism_v2_core.py tests/test_dism_v2_codegen.py tests/test_dism_v2_precision.py \
    tests/test_dism_v2_embedding_precision.py \
    tests/test_dism_v2_boundaries.py \
    tests/test_dism_v2_recompute.py \
    tests/test_dism_v2_reverse.py \
    tests/test_dism_v2_backward.py \
    --junitxml="$check_dir/pytest.xml" -o junit_family=legacy \
    | tee "$check_dir/pytest.log"
for checker in memcheck racecheck synccheck; do
    # Check every new Dism kernel; do not instrument unrelated PyTorch oracle
    # kernels (the latter greatly increases sanitizer runtime).
    "$cuda_root/bin/compute-sanitizer" --tool "$checker" \
        --kernel-name kns=_ZN7dism_v2 --error-exitcode 1 \
        "$dism_python" -m pytest -x -q -p no:cacheprovider \
        tests/test_dism_v2_core.py tests/test_dism_v2_recompute.py tests/test_dism_v2_backward.py \
        > "$check_dir/$checker.log" 2>&1
    tail -3 "$check_dir/$checker.log"
done
