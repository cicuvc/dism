"""Ensure boundary extraction adds stores, not shuffle instructions or calls."""
import re
import sys
from pathlib import Path

root=Path(sys.argv[1])
sass=(root/"sass.txt").read_text()
assert not re.search(r"\bCALL(?:\.|\s)",sass)
assert not re.search(r"\b(?:LDL|STL)(?:\.|\s)",sass)
resources=(root/"resources.txt").read_text()
sizes=re.findall(r"(?:LOCAL|STACK):(\d+)",resources)
assert sizes and all(int(x)==0 for x in sizes)
counts={}
for section in sass.split("Function : ")[1:]:
    name=section.splitlines()[0].strip()
    counts[name]=len(re.findall(r"\bSHFL(?:\.|\s)",section))
base=[v for k,v in counts.items() if "forward_probeILb0" in k]
export=[v for k,v in counts.items() if "forward_probeILb1" in k]
assert len(base)==len(export)==1,counts
assert base==export,counts
for shape in ((16,16),(16,32),(32,32),(16,64),(32,16),(16,128)):
    prefix=f"shape_probeILi{shape[0]}ELi{shape[1]}ELb"
    b=[v for k,v in counts.items() if prefix+"0" in k]
    e=[v for k,v in counts.items() if prefix+"1" in k]
    assert len(b)==len(e)==1 and b==e,(shape,counts)
print("PASS codegen: no CALL/local/stack; equal baseline/export SHFL counts",counts)
