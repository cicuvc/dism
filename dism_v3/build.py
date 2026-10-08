from abc import abstractmethod
import asyncio
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from functools import wraps
from itertools import product
from typing import Literal


build_path = Path('build')
object_path = build_path / "object"
dep_path = build_path / "deps"

clang_path = shutil.which('clang')
linker_path = shutil.which('clang')
fatbin_path = shutil.which('fatbinary') or '/usr/local/cuda/bin/fatbinary'

Status = Literal['pending', 'ok', 'failed']

compile_slots = None
def limited_compile(function):
    @wraps(function)
    async def run(*args, **kwargs):
        async with compile_slots:
            return await function(*args, **kwargs)
    return run

class BuildGraphNode:
    def __init__(self, file: Path):
        self.file = Path(file)
        self.dependencies: list[BuildGraphNode] = []
        self.status: Status = 'pending'

        self.report_cached = True

    @abstractmethod
    async def build(self, **kwargs) -> bool: ...

    def pre_build(self, **kwargs): 
        pass

    def clean(self):
        if self.file.exists():
            os.remove(str(self.file))
        for i in self.dependencies:
            i.clean()

    def pre_materialize(self, **kwargs):
        for i in self.dependencies:
            i.pre_materialize(**kwargs)
        self.pre_build(**kwargs)

    async def materialize(self, **kwargs):
        if self.status != 'pending':
            return
        
        if self.check():
            if self.report_cached:
                print(f"File {self.file} exists")
            self.status = 'ok'
            return
        
        await asyncio.gather(*[i.materialize(**kwargs) for i in self.dependencies])

        for i in self.dependencies:
            if i.status != 'ok':
                self.status = 'failed'
                return
            
        if await self.build(**kwargs):
            print(f"Build file {self.file}")
            self.status = 'ok'
        else:
            print(f"Build file {self.file} failed")
            self.status = 'failed'

    def check(self) -> bool:
        if self.status == 'ok':
            return True
        
        if not self.file.exists():
            return False

        for i in self.dependencies:
            if not i.check():
                return False
            if self.file.stat().st_mtime < i.file.stat().st_mtime:
                return False
        return True

class CXXSourceFile(BuildGraphNode):
    def __init__(self, file: Path, is_header: bool = False):
        super().__init__(file)

        self.is_header = is_header
        self.report_cached = not is_header

    async def build(self, **kwargs) -> bool:
        return True

    def clean(self):
        pass
    
    def check(self) -> bool:
        return True

class CXXOptionsFile(BuildGraphNode):
    def __init__(self, target_file: Path, cxx_args: list[str]):
        optfile = dep_path / f"{target_file.name}.opt"
        super().__init__(optfile)
        self.cxx_args = cxx_args

    async def build(self, **kwargs) -> bool:
        self.file.parent.mkdir(parents=True, exist_ok=True)

        self.file.write_text(json.dumps(self.cxx_args))
        return True

    def check(self) -> bool:
        if not super().check():
            return False
        return self.file.read_text() == json.dumps(self.cxx_args)

class CXXDepFile(BuildGraphNode):
    def __init__(self, cxxfile: CXXSourceFile, cxx_args: list[str]):
        depfile = dep_path / f"{cxxfile.file.name}.d"
        super().__init__(depfile)
        self.cxxfile = cxxfile
        self.cxx_args = cxx_args
        self.dependencies = [cxxfile, CXXOptionsFile(depfile, cxx_args)]

    def check(self) -> bool:
        if not super().check():
            return False
        # A changed header may add/remove transitive includes. Refresh the
        # dependency file itself, not just the object built from the OLD list.
        deps = self.file.read_text()
        names = deps[deps.find(': ') + 2:].replace('\\', '\n').split()
        timestamp = self.file.stat().st_mtime_ns
        return all(Path(name).exists() and Path(name).stat().st_mtime_ns <= timestamp
                   for name in names)
    
    async def build(self, **kwargs) -> bool:
        self.file.parent.mkdir(parents=True, exist_ok=True)

        dep_args = [clang_path] + self.cxx_args + [str(self.cxxfile.file.absolute()), '-MM', '-E', '-Wno-unknown-cuda-version']
        dep_proc = await asyncio.create_subprocess_exec(
            *dep_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output, diagnostics = await dep_proc.communicate()
        if diagnostics:
            print(diagnostics.decode(), end='')
        if dep_proc.returncode != 0:
            self.file.unlink(missing_ok=True)
        else:
            self.file.write_bytes(output)
        return dep_proc.returncode == 0

class CXXObjectFile(BuildGraphNode):
    def __init__(self, cxxfile: CXXSourceFile, cxx_args: list[str], objfile: Path | None = None, depfile: CXXDepFile | None = None):
        objfile = objfile or (object_path / f"{cxxfile.file.name}.o")
        super().__init__(objfile)
        self.cxxfile = cxxfile
        self.cxx_args = cxx_args
        self.depfile = depfile or CXXDepFile(cxxfile, cxx_args)
        self.optfile = CXXOptionsFile(objfile, cxx_args)
        self.dependencies.extend([self.depfile, self.optfile])
        self.type = 'object'

        if not self.depfile.check():
            asyncio.run((self.depfile.materialize()))
            if self.depfile.status != 'ok':
                self.status = 'failed'
                return

        deps = Path(self.depfile.file).read_text()
        dep_files = [os.path.normpath(i) for i in deps[2 + deps.find(': '):].replace('\\','\n').split() if i != '']
        self.dependencies.extend([CXXSourceFile(i, is_header=True) for i in dep_files if not Path(i).samefile(self.cxxfile.file)])

    def pre_build(self, **kwargs):
        args = [clang_path] + self.cxx_args + [str(self.cxxfile.file.absolute()), '-c', '-o', self.file]
        if 'compile_commands' in kwargs:
            kwargs['compile_commands'].add(str(self.cxxfile.file), args)
    @limited_compile
    async def build(self, **kwargs) -> bool:
        self.file.parent.mkdir(parents=True, exist_ok=True)

        
        args = [clang_path] + self.cxx_args + [str(self.cxxfile.file.absolute()), '-c', '-o', self.file]

        proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        await proc.wait()
        print(str(await proc.stdout.read(), encoding='utf-8'), end='')
        return proc.returncode == 0

class CUDAObjectFile(BuildGraphNode):
    def __init__(self, cufile: CXXSourceFile, cxx_args: list[str], archs: list[int] = []):
        objfile = object_path / f"{cufile.file.name}.o"
        super().__init__(objfile)

        self.device_objects = []

        self.depfile = CXXDepFile(cufile, cxx_args)
        self.optfile = CXXOptionsFile(objfile, cxx_args + archs)

        for i in archs:
            cuda_dev_args = [*cxx_args, '--cuda-device-only', f'--cuda-gpu-arch=sm_{i}']
            device_obj = object_path / f"{Path(cufile.file).name}.dev.sm{i}.o"
            node = CXXObjectFile(cufile, cuda_dev_args, device_obj, depfile=self.depfile)
            self.device_objects.append((i, node.file))
            self.dependencies.append(node)
        self.dependencies.append(cufile)
        self.dependencies.append(self.depfile)
        self.dependencies.append(self.optfile)

        self.cufile = cufile
        self.archs = archs
        self.cxx_args = cxx_args
        self.fatbin_path = object_path / f"{cufile.file.name}.fatbin"

        self.type = 'object'

    def pre_materialize(self, **kwargs):
        pass
    def pre_build(self, **kwargs):
        args = [clang_path, *self.cxx_args, '--cuda-host-only', '-c', '-o', str(self.file), '-Xclang', "-fcuda-include-gpubinary", "-Xclang", str(self.fatbin_path), self.cufile.file, "-fPIC"]
        if 'compile_commands' in kwargs:
            kwargs['compile_commands'].add(str(self.cufile.file), args)

    @limited_compile
    async def build(self, **kwargs) -> bool:
        self.file.parent.mkdir(parents=True, exist_ok=True)

        fatbin_args = [fatbin_path, '-64', '--create', self.fatbin_path, *[f'--image3=kind=elf,sm={arch},file={obj}' for arch, obj in self.device_objects]]
        fatbin_proc = await asyncio.create_subprocess_exec(*fatbin_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        await fatbin_proc.wait()
        if fatbin_proc.returncode != 0:
            print(str(await fatbin_proc.stdout.read(), encoding='utf-8'))
            return False

        args = [clang_path, *self.cxx_args, '--cuda-host-only', '-c', '-o', str(self.file), '-Xclang', "-fcuda-include-gpubinary", "-Xclang", str(self.fatbin_path), self.cufile.file, "-fPIC"]
        
        proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        await proc.wait()
        print(str(await proc.stdout.read(), encoding='utf-8'), end='')

        return proc.returncode == 0

class DynamicLibrary(BuildGraphNode):
    def __init__(self, lib: Path, libpath: list[Path], links: list[str]):
        lib = Path(lib)

        super().__init__(lib)
        
        self.libpath = [Path(i) for i in libpath]
        self.links = links

        self.opt_file = CXXOptionsFile(lib, [])
        self.dependencies.append(self.opt_file)

    def add_cxx(self, cxxfile: Path, cxx_args: list[str] = []):
        cxx_src = CXXSourceFile(cxxfile)
        obj_file = CXXObjectFile(cxx_src, cxx_args)
        self.dependencies.append(obj_file)

        self.opt_file.cxx_args.append(str(obj_file.file))

    def add_cuda(self, cxxfile: Path, cxx_args: list[str] = [], archs: list[int] = []):
        cxx_src = CXXSourceFile(cxxfile)
        obj_file = CUDAObjectFile(cxx_src, cxx_args, archs)
        self.dependencies.append(obj_file)

        self.opt_file.cxx_args.append(str(obj_file.file))

    async def build(self) -> bool:
        self.file.parent.mkdir(parents=True, exist_ok=True)

        obj_files = [str(i.file) for i in self.dependencies if hasattr(i, 'type') and getattr(i, 'type') == 'object']

        args = [linker_path, '-fPIC', '-shared', '-o', str(self.file), *obj_files, *[f'-L{i.absolute()}' for i in self.libpath], *[f"-Wl,-rpath,{i.absolute()}" for i in self.libpath], *[f'-l{i}' for i in self.links]]
        proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        await proc.wait()
        if proc.returncode != 0:
            print(str(await proc.stdout.read(), encoding='utf-8'))

            print(args)
        return proc.returncode == 0

class CompileCommandsService:
    def __init__(self, json_path: Path):
        self.json_path = json_path
        self.commands = []

    def add(self, cxxfile: Path, args: list[str]):
        self.commands.append({
            'directory': str(Path('.').absolute()),
            'file': str(Path(cxxfile).absolute()),
            'arguments': [str(i) for i in args]
        })

    def dump(self):
        self.json_path.write_text(json.dumps(self.commands, indent=4))

class CXXOptionGenerator:
    def __init__(self, *flags: str):
        self.flags = list(flags)
        self.incdir = []

    def add_includedir(self, dir: str):
        self.incdir.append(dir)

    def get_cxxflags(self) -> list[str]:
        return self.flags + [f"-I{i}" for i in self.incdir]
    

def build(*targets: BuildGraphNode):
    compile_commands = CompileCommandsService(build_path / ('compile_commands.json'))
    
    
    for i in targets:
        i.pre_materialize(compile_commands = compile_commands)

    async def run():
        global compile_slots
        compile_slots = asyncio.Semaphore(int(os.environ.get('DISM_BUILD_JOBS', '4')))
        tasks = [asyncio.ensure_future(i.materialize()) for i in targets]
        await asyncio.gather(*tasks)

    asyncio.run(run())

    compile_commands.dump()
    for i in targets:
        if i.status == 'ok':
            print(f"Target {i.file} successfully built")
        else:
            print(f"Target {i.file} build failed")
    return all(i.status == 'ok' for i in targets)

def clean(*targets: BuildGraphNode):
    for i in targets:
        i.clean()

if __name__ == "__main__":
    import sysconfig
    import glob

    package_name = 'cu_flash_dism'
    output_path = Path('python')
    archs = ['120a'] # Architecture-specific setmaxnreg on RTX5090.

    python_version = sysconfig.get_python_version()
    python_include = Path(sysconfig.get_path('include'))
    python_lib = Path(sysconfig.get_config_var('LIBDIR'))

    libpython_name = f"python{python_version}"

    torch_path = Path(str(importlib.import_module('torch').__path__[0]))

    python_links = [libpython_name]
    torch_links = ["torch", "c10", "torch_cpu", "torch_cuda", "c10_cuda", "torch_python"]

    cxx_build_args = CXXOptionGenerator('-O3', '-std=c++20', '-fPIC', f"-DPACKAGE_NAME={package_name}")
    cxx_build_args.add_includedir(Path('.') / 'include')
    # CUDA13 packages Thrust/CUB under cccl; ATen CUDA headers include Thrust.
    cxx_build_args.add_includedir(Path('/usr/local/cuda/include/cccl'))
    cxx_build_args.add_includedir(torch_path / 'include/torch/csrc/api/include')
    cxx_build_args.add_includedir(torch_path / 'include')
    cxx_build_args.add_includedir(python_include)

    target = DynamicLibrary(output_path / f"{package_name}{sysconfig.get_config_var('EXT_SUFFIX')}", [python_lib, torch_path/'lib'], python_links + torch_links + ['m', 'stdc++'])

    configs = list(product((16,32),(32,64),(32,64)))
    if os.environ.get('DISM_BUILD_CONFIGS'):
        configs = [tuple(map(int, item.split(','))) for item in os.environ['DISM_BUILD_CONFIGS'].split(';')]
        if any(c not in list(product((16,32),(32,64),(32,64))) for c in configs):
            raise ValueError('unsupported build configuration')
    registry = '#define DISM_CONFIGS(X) ' + ' '.join(f'X({r},{d},{v})' for r,d,v in configs) + '\n'
    build_path.mkdir(exist_ok=True)
    registry_path = build_path / 'config_registry.h'
    if not registry_path.exists() or registry_path.read_text() != registry:
        registry_path.write_text(registry)
    backward_debug = int(os.environ.get('DISM_BACKWARD_DEBUG', '0'))
    fp32 = int(os.environ.get('DISM_ENABLE_FP32', '0'))
    probes = int(os.environ.get('DISM_BUILD_PROBES', '0'))
    if probes not in (0,1):
        raise ValueError('DISM_BUILD_PROBES must be0 or1')
    base_flags = cxx_build_args.get_cxxflags() + ['-Ibuild', f'-DDISM_ENABLE_FP32={fp32}']
    target.add_cxx('src/module.cpp', base_flags)
    target.add_cxx('src/frontend.cpp', base_flags)
    cxx_files = [p for p in glob.glob('src/**/*.cpp', recursive=True)
                 if p not in ('src/module.cpp','src/frontend.cpp') and (probes or not p.startswith('src/probe/'))]
    cuda_files = glob.glob('src/*.cu')
    if probes:
        cuda_files += glob.glob('src/probe/*.cu')
    for r,d,v in configs:
        tag = f'r{r}_d{d}_v{v}'
        object_path = build_path / 'object' / tag
        dep_path = build_path / 'deps' / tag
        flags = base_flags + [f'-DDISM_VARIANT=dism_{tag}',
                 f'-DDISM_READOUT_DIM={r}', f'-DDISM_KC_KEY_DIM={d}', f'-DDISM_KC_HEAD_DIM={v}',
                 f'-DDISM_BACKWARD_DEBUG={backward_debug}']
        if os.environ.get('DISM_LINEINFO') == '1':
            flags.append('-gline-tables-only')
        for p in cxx_files:
            # Toolkit type declarations use CUDA syntax, but this is host-only:
            # Torch is never parsed by a device compiler.
            host_flags = flags + (['-x', 'cuda', '--cuda-host-only', '-Wno-unknown-cuda-version']
                                  if '/host/' in p else [])
            if p in ('src/interface.cpp','src/varlen_interface.cpp'):
                host_flags += [f'-DDISM_BUILD_PROBES={probes}']
            target.add_cxx(p, host_flags)
        for p in cuda_files:
            target.add_cuda(p, flags + ['-Wno-unknown-cuda-version','-Xcuda-ptxas','-v'], archs)

    sys.exit(0 if build(target) else 1)
