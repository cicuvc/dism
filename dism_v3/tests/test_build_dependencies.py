"""A header changing its includes must invalidate the dependency file itself."""
import asyncio
import importlib.util
import os
from pathlib import Path


def test_transitive_header_refresh(tmp_path):
    spec = importlib.util.spec_from_file_location('dism_build', Path(__file__).parents[1] / 'build.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.dep_path = tmp_path / 'deps'
    source = tmp_path / 'sample.cpp'
    first = tmp_path / 'first.h'
    second = tmp_path / 'second.h'
    source.write_text('#include "first.h"\n')
    first.write_text('#pragma once\n')
    second.write_text('#pragma once\n')

    def node():
        return build.CXXDepFile(build.CXXSourceFile(source), ['-std=c++20'])

    original = node()
    asyncio.run(original.materialize())
    assert original.status == 'ok' and original.check()
    assert str(second) not in original.file.read_text()

    first.write_text('#pragma once\n#include "second.h"\n')
    timestamp = original.file.stat().st_mtime_ns + 1
    os.utime(first, ns=(timestamp, timestamp))
    changed = node()
    assert not changed.check()
    asyncio.run(changed.materialize())
    assert changed.status == 'ok' and changed.check()
    assert str(second) in changed.file.read_text()

    timestamp = changed.file.stat().st_mtime_ns + 1
    os.utime(second, ns=(timestamp, timestamp))
    assert not node().check()
    second.unlink()
    assert not node().check()
