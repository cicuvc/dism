"""Setuptools bridge to the maintained clang/CUDA build graph."""
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parent


class BuildCUDA(build_ext):
    def build_extension(self, ext):
        if ext.name != 'cu_flash_dism':
            component = 'prefill' if ext.name == '_dism_prefill' else 'decode'
            subprocess.run([
                sys.executable, str(ROOT / 'python/flash_dism/inference/build.py'),
                '--component', component,
                '--output', str(Path(self.get_ext_fullpath(ext.name)).resolve()),
            ], check=True)
            return
        # Always consult the native dependency/configuration graph, including
        # on editable installs. Never silently package an unrelated stale .so.
        subprocess.run([sys.executable, str(ROOT / 'build.py')], cwd=ROOT, check=True)
        source = ROOT / 'python' / (ext.name + sysconfig.get_config_var('EXT_SUFFIX'))
        target = Path(self.get_ext_fullpath(ext.name)).resolve()
        if not source.is_file():
            raise RuntimeError(f'build.py did not produce {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target:
            shutil.copy2(source, target)


class BuildPython(build_py):
    def find_package_modules(self, package, package_dir):
        # Some historical kernel-local smoke scripts live alongside modules.
        # They are not runtime dependencies and use standalone-only imports.
        return [entry for entry in super().find_package_modules(package, package_dir)
                if not entry[1].startswith('test_')]


setup(
    ext_modules=[Extension(name, sources=[]) for name in
                 ('cu_flash_dism', '_dism_prefill', 'dism_decode_cuda')],
    cmdclass={'build_ext': BuildCUDA, 'build_py': BuildPython},
)
