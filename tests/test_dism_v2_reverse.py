"""Reverse affine, paired mailboxes, 32-key summaries and reverse passing."""
import os
from pathlib import Path
import re
import subprocess
import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_reverse_affine_pipeline(tmp_path):
    if torch.cuda.get_device_capability()!=(12,0): pytest.skip("sm120a probe")
    root=Path(__file__).resolve().parents[1]
    cuda=Path(CUDA_HOME)
    glx=Path(os.environ.get("GLX_ROOT","/home/cicuvc/cs/projects/glx"))
    binary=tmp_path/"reverse"
    subprocess.run([str(cuda/"bin/nvcc"),"-std=c++20","-O3","-arch=sm_120a",
        "-I"+str(glx/"include"),str(root/"experiments/glx_reverse/reverse.cu"),
        "-lcuda","-o",str(binary)],check=True,capture_output=True,text=True)
    output=subprocess.check_output([str(binary)],text=True)
    assert output.count("PASS n=")==45 and "FAIL" not in output
    sass=subprocess.check_output([str(cuda/"bin/cuobjdump"),"--dump-sass",str(binary)],text=True)
    assert not re.search(r"\b(?:CALL|LDL|STL)(?:\.|\s)",sass)
    resources=subprocess.check_output([str(cuda/"bin/cuobjdump"),"--dump-resource-usage",str(binary)],text=True)
    sizes=re.findall(r"(?:STACK|LOCAL):(\d+)",resources)
    assert sizes and all(int(x)==0 for x in sizes)
