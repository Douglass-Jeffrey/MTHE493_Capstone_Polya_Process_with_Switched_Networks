"""
Where the scale experiments keep their data: the folder named by the POLYA_WORK environment variable, or work/
at the top of the repository (ignored by git).

    work/data/<name>/      split-format graphs (scale.gen, split_format.from_csr) and finite-memory ring files
    work/cache/<name>/     CSR caches (indptr.npy, indices.npy, meta.json), as run_sim.py --save-csr writes them
    work/base_src/src/     a copy of an earlier version of the code, for the e2e comparisons
                           (git archive polya-step-experiments src | tar -x -C work/base_src)
    work/orig_src/         the original code (fd79e4d), for the fused-kernel and profiling harnesses
                           (git archive fd79e4d src/capstone_optimized | tar -x --strip-components=1 -C work/orig_src)
    work/reports/, e2e/    Nsight reports and e2e outputs written by the harnesses in performance_analysis/
"""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ROOT = os.environ.get("POLYA_WORK") or os.path.join(REPO, "work")
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "cache")
BASE_SRC = os.path.join(ROOT, "base_src", "src")
