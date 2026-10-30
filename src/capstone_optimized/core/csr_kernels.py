from ..cupy_fallback import cp, CUPY_AVAILABLE
import numpy as np

"""
Custom CUDA kernels for the Polya step on an unweighted (0/1) CSR adjacency matrix.

The cuSPARSE product `adj_matrix @ node_urns` upcasts everything to float64 and streams the whole
matrix (values, indices, indptr) through memory every step. Since every edge weight is 1, a mega urn is
just a node's own urn plus the sum of its neighbours' urns, so we only need indptr and indices:
  - draw_two_colour : mega urn gather + 2-colour draw fused in one pass (no mega/probs temporaries)
  - mega_urns       : same values as adj_matrix @ node_urns (+ node_urns) for any colour count <= 8
  - apply_draws     : node_urns[i, draws[i]] += delta[i, draws[i]] without fancy-indexing temporaries
Sums are done in int64, so they are exact (the float64 SpMM is exact too, the results are identical).

The neighbour lookups are random reads, which is what limits this step. Indices are read in order, so
they can sit in pinned host RAM (zero copy) and be read over PCIe at no extra cost when VRAM is tight.
"""

_SRC = r"""
#define FULL 0xffffffffu

// LANES threads share one row: avg degree is ~6, so 4 lanes keeps most threads busy while hub rows
// (degree 10^4+) are still split up.
template<int LANES>
__global__ void draw_two_colour(const int* __restrict__ indptr, const int* __restrict__ indices,
                                const int2* __restrict__ urns, const double* __restrict__ rand_vals,
                                int* __restrict__ draws, long long n)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long row = t / LANES;
    int lane = (int)(t & (LANES - 1));
    bool valid = row < n;
    long long s0 = 0, s1 = 0;
    if (valid) {
        for (long long k = indptr[row] + lane; k < indptr[row + 1]; k += LANES) {
            int2 v = urns[indices[k]];
            s0 += v.x; s1 += v.y;
        }
    }
    for (int off = LANES / 2; off > 0; off >>= 1) {
        s0 += __shfl_down_sync(FULL, s0, off, LANES);
        s1 += __shfl_down_sync(FULL, s1, off, LANES);
    }
    if (valid && lane == 0) {
        int2 u = urns[row];
        s0 += u.x; s1 += u.y;
        // same float64 arithmetic as the original step: probs[:, 0] = mega0 / row_sum, row_sum 0 -> 1
        double m0 = (double)s0, row_sum = m0 + (double)s1;
        if (row_sum == 0.0) row_sum = 1.0;
        draws[row] = rand_vals[row] > m0 / row_sum ? 1 : 0;
    }
}

#define MAXC 8
template<int LANES>
__global__ void mega_urns(const int* __restrict__ indptr, const int* __restrict__ indices,
                          const int* __restrict__ urns, double* __restrict__ out, int c, int include_self,
                          long long n)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long row = t / LANES;
    int lane = (int)(t & (LANES - 1));
    bool valid = row < n;
    long long acc[MAXC];
    for (int j = 0; j < MAXC; ++j) acc[j] = 0;
    if (valid) {
        for (long long k = indptr[row] + lane; k < indptr[row + 1]; k += LANES) {
            const int* v = urns + (long long)indices[k] * c;
            for (int j = 0; j < c; ++j) acc[j] += v[j];
        }
    }
    for (int j = 0; j < c; ++j) {
        for (int off = LANES / 2; off > 0; off >>= 1) acc[j] += __shfl_down_sync(FULL, acc[j], off, LANES);
        if (valid && lane == 0)
            out[row * c + j] = (double)(acc[j] + (include_self ? urns[row * c + j] : 0));
    }
}
"""
LANES = 4
MAX_COLORS = 8
_BLOCK = 256
_kernels = {}


def _get(name):
    if not _kernels:
        names = [f"draw_two_colour<{LANES}>", f"mega_urns<{LANES}>"]
        mod = cp.RawModule(code=_SRC, options=("-std=c++14",), name_expressions=names)
        for nm in names:
            _kernels[nm.split("<")[0]] = mod.get_function(nm)
        _kernels["apply_draws"] = cp.ElementwiseKernel(
            "D d, raw T delta, int64 c", "raw U urns",
            "urns[i * c + d] += delta[i * c + d]", "polya_apply_draws")
    return _kernels[name]


def _ptr(arr):
    # indices may live in pinned host memory (numpy view), the GPU reads it directly through its address
    return arr if isinstance(arr, cp.ndarray) else np.uint64(arr.ctypes.data)


def _grid(n):
    return ((int(n) * LANES + _BLOCK - 1) // _BLOCK,)


def urns_supported(urns):
    """Kernels read urn rows directly, so they need a C-contiguous int32 (n, c) GPU array."""
    return (CUPY_AVAILABLE and isinstance(urns, cp.ndarray) and urns.dtype == cp.int32 and urns.ndim == 2
            and urns.flags.c_contiguous and urns.data.ptr % 8 == 0)


def draw_two_colour(indptr, indices, urns, rand_vals):
    n = urns.shape[0]
    draws = cp.empty(n, dtype=cp.int32)
    _get("draw_two_colour")(_grid(n), (_BLOCK,), (indptr, _ptr(indices), urns, rand_vals, draws, np.int64(n)))
    return draws


def mega_urns(indptr, indices, urns, include_self=True):
    n, c = urns.shape
    out = cp.empty((n, c), dtype=cp.float64)
    _get("mega_urns")(_grid(n), (_BLOCK,), (indptr, _ptr(indices), urns, out, np.int32(c),
                                            np.int32(bool(include_self)), np.int64(n)))
    return out


def apply_draws(draws, delta, urns):
    _get("apply_draws")(draws, delta, np.int64(urns.shape[1]), urns)
