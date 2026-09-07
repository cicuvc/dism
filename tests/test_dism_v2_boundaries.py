"""Compile/run sparse forward boundaries and independent transposed rescan."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_glx_sparse_boundaries(tmp_path):
    if torch.cuda.get_device_capability() != (12,0):
        pytest.skip("sm120a probe")
    root=Path(__file__).resolve().parents[1]
    cuda=Path(CUDA_HOME)
    glx=Path(os.environ.get("GLX_ROOT","/home/cicuvc/cs/projects/glx"))
    executable=tmp_path/"boundaries"
    compile=subprocess.run([str(cuda/"bin/nvcc"),"-std=c++20","-O3","-arch=sm_120a",
        "-I"+str(glx/"include"),str(root/"experiments/glx_boundaries/boundaries.cu"),
        "-o",str(executable)],capture_output=True,text=True)
    assert compile.returncode==0,compile.stdout+compile.stderr
    run=subprocess.run([str(executable)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert "PASS cases=40" in run.stdout
    for shape in ("16x16","16x32","32x16","32x32","16x64","16x128"):
        assert f"PASS shape={shape}" in run.stdout
    for flag,file in (("--dump-sass","sass.txt"),("--dump-resource-usage","resources.txt")):
        result=subprocess.check_output([str(cuda/"bin/cuobjdump"),flag,str(executable)],text=True)
        (tmp_path/file).write_text(result)
    subprocess.run([sys.executable,str(root/"experiments/glx_boundaries/check_codegen.py"),str(tmp_path)],check=True)
