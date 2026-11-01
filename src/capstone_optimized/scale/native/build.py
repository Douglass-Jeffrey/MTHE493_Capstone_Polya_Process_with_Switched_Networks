"""Builds the native helpers (feeder.dll, cpu_step.dll) with MSVC if a source is newer than its DLL."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VCVARS = r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
TARGETS = {"feeder": ["feeder.cpp"], "cpu_step": ["cpu_step.cpp"]}


def build(name, force=False):
    srcs = [os.path.join(HERE, s) for s in TARGETS[name]]
    dll = os.path.join(HERE, name + ".dll")
    if not force and os.path.exists(dll) and all(os.path.getmtime(dll) >= os.path.getmtime(s) for s in srcs):
        return dll
    cmd = (f'call "{VCVARS}" >nul && cl /nologo /O2 /EHsc /std:c++17 /openmp:llvm /arch:AVX2 /LD '
           + " ".join(f'"{s}"' for s in srcs) + f' /Fe:"{dll}" /Fo:"{HERE}\\\\"')
    r = subprocess.run(cmd, shell=True, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"building {name} failed:\n{r.stdout}\n{r.stderr}")
    for ext in (".obj", ".lib", ".exp"):
        for s in TARGETS[name]:
            pth = os.path.join(HERE, os.path.splitext(s)[0] + ext)
            if os.path.exists(pth):
                os.remove(pth)
        pth = os.path.join(HERE, name + ext)
        if os.path.exists(pth):
            os.remove(pth)
    return dll


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["feeder"]:
        print(build(nm, force=True))
