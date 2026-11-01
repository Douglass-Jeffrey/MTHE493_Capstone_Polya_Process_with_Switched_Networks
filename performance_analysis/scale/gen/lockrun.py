"""gpulock.py with a 0.2 s poll instead of 3 s (same lock directory and protocol). Used for the long ba_1b run,
which otherwise kept losing the lock to back-to-back short runs.

    python lockrun.py <tag> -- python script.py args...
"""
import os
import runpy
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gpulock  # noqa: E402

_sleep = time.sleep
gpulock.time.sleep = lambda s: _sleep(min(s, 0.2))
sys.argv[0] = gpulock.__file__
runpy.run_path(gpulock.__file__, run_name="__main__")
