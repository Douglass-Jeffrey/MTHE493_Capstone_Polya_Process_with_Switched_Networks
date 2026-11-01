"""Experimental kernels for the Polya step on a 0/1 CSR adjacency (all weights 1).

draw2<LANES>  : fused mega-urn gather + 2-colour draw. LANES threads share a row (1 = thread per row).
gather<LANES> : mega urns (float64, row-major) for any colour count <= 8, same values as A @ U (+ U).
Both take (row0, nrows, idx_base) so the same code runs on a chunk of rows whose indices sit in a
separate buffer: k-th neighbour of row r is indices[indptr[r] - idx_base + k].
"""
import numpy as np
import cupy as cp

SRC = r"""
#define FULL 0xffffffffu

template<int LANES>
__global__ void draw2(const int* __restrict__ indptr, const int* __restrict__ indices, long long idx_base,
                      const int2* __restrict__ urns, const double* __restrict__ rnd, int* __restrict__ draws,
                      long long row0, long long nrows)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long r = t / LANES;
    int lane = (int)(t & (LANES - 1));
    bool valid = r < nrows;
    long long row = row0 + r;
    long long s0 = 0, s1 = 0;
    if (valid) {
        long long beg = (long long)indptr[row] - idx_base, end = (long long)indptr[row + 1] - idx_base;
        for (long long k = beg + lane; k < end; k += LANES) {
            int2 v = urns[indices[k]];
            s0 += v.x; s1 += v.y;
        }
    }
    if (LANES > 1) {
        for (int off = LANES / 2; off > 0; off >>= 1) {
            s0 += __shfl_down_sync(FULL, s0, off, LANES);
            s1 += __shfl_down_sync(FULL, s1, off, LANES);
        }
    }
    if (valid && lane == 0) {
        int2 u = urns[row];
        s0 += u.x; s1 += u.y;
        // same float64 arithmetic as Polya_Process.step: probs[:, 0] = mega0 / row_sum (row_sum 0 -> 1)
        double m0 = (double)s0, rs = m0 + (double)s1;
        if (rs == 0.0) rs = 1.0;
        draws[row] = rnd[row] > m0 / rs ? 1 : 0;
    }
}

#define MAXC 8
template<int LANES>
__global__ void gather(const int* __restrict__ indptr, const int* __restrict__ indices, long long idx_base,
                       const int* __restrict__ urns, double* __restrict__ out, int c, int include_self,
                       long long row0, long long nrows)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long r = t / LANES;
    int lane = (int)(t & (LANES - 1));
    bool valid = r < nrows;
    long long row = row0 + r;
    long long acc[MAXC];
    for (int j = 0; j < MAXC; ++j) acc[j] = 0;
    if (valid) {
        long long beg = (long long)indptr[row] - idx_base, end = (long long)indptr[row + 1] - idx_base;
        for (long long k = beg + lane; k < end; k += LANES) {
            const int* v = urns + (long long)indices[k] * c;
            for (int j = 0; j < c; ++j) acc[j] += v[j];
        }
    }
    for (int j = 0; j < c; ++j) {
        if (LANES > 1)
            for (int off = LANES / 2; off > 0; off >>= 1) acc[j] += __shfl_down_sync(FULL, acc[j], off, LANES);
        if (valid && lane == 0)
            out[row * c + j] = (double)(acc[j] + (include_self ? urns[row * c + j] : 0));
    }
}
"""
LANE_OPTS = (1, 2, 4, 8, 16, 32)
_names = [f"draw2<{l}>" for l in LANE_OPTS] + [f"gather<{l}>" for l in LANE_OPTS]
_mod = cp.RawModule(code=SRC, options=("-std=c++14",), name_expressions=_names)
_k = {n: _mod.get_function(n) for n in _names}
BLOCK = 256

apply_draws = cp.ElementwiseKernel(
    "D d, raw T delta, int64 c", "raw U urns",
    "urns[i * c + d] += delta[i * c + d]", "polya_apply_draws")


def _grid(nrows, lanes):
    return ((int(nrows) * lanes + BLOCK - 1) // BLOCK,)


def draw2(indptr, indices, urns, rnd, draws, lanes=8, row0=0, nrows=None, idx_base=0, stream=None):
    nrows = urns.shape[0] - row0 if nrows is None else nrows
    args = (indptr, indices, np.int64(idx_base), urns, rnd, draws, np.int64(row0), np.int64(nrows))
    _k[f"draw2<{lanes}>"](_grid(nrows, lanes), (BLOCK,), args, stream=stream)


def gather(indptr, indices, urns, include_self=True, lanes=8):
    n, c = urns.shape
    out = cp.empty((n, c), dtype=cp.float64)
    args = (indptr, indices, np.int64(0), urns, out, np.int32(c), np.int32(include_self), np.int64(0), np.int64(n))
    _k[f"gather<{lanes}>"](_grid(n, lanes), (BLOCK,), args)
    return out
