"""Run a command while holding the machine-wide GPU lock, so that experiments running in parallel never share
the GPU (timings stay clean and two 5 GB jobs never meet in 6 GB of VRAM). Any run that uses the GPU, or more
than ~2 GB of host RAM, goes through this.

    python gpulock.py <tag> -- python script.py args...

The lock is a directory (mkdir is atomic). A lock whose owner process has died is taken over.
"""
import os
import subprocess
import sys
import time

import psutil

LOCK = os.path.join(os.environ.get("POLYA_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "work"), "gpu.lock")
OWNER = os.path.join(LOCK, "owner")


def _owner_alive():
    try:
        pid = int(open(OWNER).read().split()[1])
    except (OSError, ValueError, IndexError):
        return True                 # being written right now; assume alive
    return psutil.pid_exists(pid)


def acquire(tag):
    waited = 0.0
    while True:
        try:
            os.mkdir(LOCK)
            with open(OWNER, "w") as f:
                f.write(f"{tag} {os.getpid()} {time.strftime('%H:%M:%S')}")
            if waited:
                print(f"[gpulock] {tag}: acquired after {waited:.0f} s", flush=True)
            return
        except FileExistsError:
            if not _owner_alive():
                print(f"[gpulock] {tag}: taking over a stale lock", flush=True)
                release()
                continue
            if waited == 0:
                try:
                    print(f"[gpulock] {tag}: waiting for {open(OWNER).read()}", flush=True)
                except OSError:
                    pass
            time.sleep(3)
            waited += 3


def release():
    try:
        os.remove(OWNER)
    except OSError:
        pass
    try:
        os.rmdir(LOCK)
    except OSError:
        pass


if __name__ == "__main__":
    if "--" not in sys.argv:
        sys.exit(__doc__)
    i = sys.argv.index("--")
    tag, cmd = (sys.argv[1] if i > 1 else "anon"), sys.argv[i + 1:]
    if cmd and os.path.basename(cmd[0]).lower() in ("python", "python.exe", "python3"):
        cmd[0] = sys.executable          # the PATH lookup in a subprocess can find a different Python
    acquire(tag)
    t0 = time.time()
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        rc = subprocess.call(cmd, env=env)
    finally:
        release()
    print(f"[gpulock] {tag}: held {time.time() - t0:.0f} s, exit {rc}", flush=True)
    sys.exit(rc)
