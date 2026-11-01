"""
Experimental variants of the fused gather-and-draw step (csr_kernels.draw_two_colour) for workstream A:

  draw_i32   int2 urns (as the repo), random numbers from an array or from Philox
  draw_u16   urns packed as one 32-bit word per node and run (red in the low 16 bits, black in the high 16),
             R runs ("replicas") of the process on the same graph stored side by side, so one gather of a
             neighbour's R words serves all R runs; random numbers from an array or Philox (stream = run)
  apply_*    add delta to the drawn colour; apply_u16_mem also keeps the finite memory as a ring of bits
             (one bit per node and step instead of an int32), subtracting the draw from M steps ago

Semantics as in the repo: super urn = own + neighbours, p = red / (red + black) in float64 (0 -> 1),
draw = 1 (black) if rand > p else 0 (red).
"""
import numpy as np
import cupy as cp

from .philox import PHILOX_SRC

_SRC = PHILOX_SRC + r"""
#define FULL 0xffffffffu

template<int LANES, bool PHILOX>
__global__ void draw_i32(const int* __restrict__ indptr, const int* __restrict__ indices,
                         const int2* __restrict__ urns, const double* __restrict__ rand_vals,
                         unsigned long long seed, unsigned int step, unsigned char* __restrict__ draws, long long n)
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
        double m0 = (double)s0, row_sum = m0 + (double)s1;
        if (row_sum == 0.0) row_sum = 1.0;
        double r = PHILOX ? philox_uniform(seed, step, (unsigned long long)row, 0) : rand_vals[row];
        draws[row] = r > m0 / row_sum ? 1 : 0;
    }
}

template<int R>
__device__ __forceinline__ void load_words(const unsigned int* __restrict__ u, unsigned int* w)
{
    if constexpr (R == 1) {
        w[0] = __ldg(u);
    } else if constexpr (R == 2) {
        uint2 v = __ldg(reinterpret_cast<const uint2*>(u));
        w[0] = v.x; w[1] = v.y;
    } else {
        #pragma unroll
        for (int q = 0; q < R / 4; ++q) {
            uint4 v = __ldg(reinterpret_cast<const uint4*>(u) + q);
            w[4 * q] = v.x; w[4 * q + 1] = v.y; w[4 * q + 2] = v.z; w[4 * q + 3] = v.w;
        }
    }
}

template<int LANES, int R, bool PHILOX>
__global__ void draw_u16(const int* __restrict__ indptr, const int* __restrict__ indices,
                         const unsigned int* __restrict__ urns, const double* __restrict__ rand_vals,
                         unsigned long long seed, unsigned int step, unsigned char* __restrict__ draws, long long n)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long row = t / LANES;
    int lane = (int)(t & (LANES - 1));
    bool valid = row < n;
    unsigned int s0[R], s1[R], w[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) { s0[r] = 0; s1[r] = 0; }
    if (valid) {
        for (long long k = indptr[row] + lane; k < indptr[row + 1]; k += LANES) {
            load_words<R>(urns + (long long)indices[k] * R, w);
            #pragma unroll
            for (int r = 0; r < R; ++r) { s0[r] += w[r] & 0xffffu; s1[r] += w[r] >> 16; }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r)
        for (int off = LANES / 2; off > 0; off >>= 1) {
            s0[r] += __shfl_down_sync(FULL, s0[r], off, LANES);
            s1[r] += __shfl_down_sync(FULL, s1[r], off, LANES);
        }
    if (valid && lane == 0) {
        load_words<R>(urns + row * R, w);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            double m0 = (double)(s0[r] + (w[r] & 0xffffu)), row_sum = m0 + (double)(s1[r] + (w[r] >> 16));
            if (row_sum == 0.0) row_sum = 1.0;
            double x = PHILOX ? philox_uniform(seed, step, (unsigned long long)row, r) : rand_vals[row * R + r];
            draws[row * R + r] = x > m0 / row_sum ? 1 : 0;
        }
    }
}

extern "C" __global__ void apply_i32(int2* __restrict__ urns, const unsigned char* __restrict__ draws, int delta,
                                     long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { if (draws[i]) urns[i].y += delta; else urns[i].x += delta; }
}

extern "C" __global__ void apply_u16(unsigned int* __restrict__ urns, const unsigned char* __restrict__ draws,
                                     unsigned int delta, long long total)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) urns[i] += draws[i] ? (delta << 16) : delta;
}

// One run (R = 1) with finite memory as a ring of bits: ring row `slot` holds the draws of M steps ago, one
// bit per node. If `expire`, that old draw's balls are removed; then the row is overwritten with this step's
// draws. A warp handles 32 consecutive nodes, so each warp reads and writes one 32-bit word of the row.
extern "C" __global__ void apply_u16_mem(unsigned int* __restrict__ urns, const unsigned char* __restrict__ draws,
                                         unsigned int* __restrict__ ring, long long words_per_row, int slot,
                                         int expire, unsigned int delta, unsigned int decay, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool d = i < n && draws[i];
    unsigned int word = __ballot_sync(FULL, d);
    long long wi = i >> 5;
    unsigned int* slot_row = ring + (long long)slot * words_per_row;
    if (i < n) {
        unsigned int u = urns[i] + (d ? (delta << 16) : delta);
        if (expire) u -= ((slot_row[wi] >> lane) & 1u) ? (decay << 16) : decay;
        urns[i] = u;
    }
    __syncwarp();
    if (lane == 0 && wi < words_per_row) slot_row[wi] = word;
}
"""

_mod = None
_BLOCK = 256


def _module():
    global _mod
    if _mod is None:
        names = [f"draw_i32<{L},{p}>" for L in (4,) for p in ("false", "true")]
        names += [f"draw_u16<{L},{R},{p}>" for L in (1, 2, 4, 8) for R in (1, 2, 4, 8, 16) for p in ("false", "true")]
        _mod = (cp.RawModule(code=_SRC, options=("-std=c++17",), name_expressions=names), names)
    return _mod[0]


def _grid(n, lanes=4):
    return ((int(n) * lanes + _BLOCK - 1) // _BLOCK,)


def set_l2_fetch_granularity(nbytes):
    """Hint for how many bytes the L2 fetches from DRAM per miss (0-128). Returns the value now in effect."""
    lim = getattr(cp.cuda.runtime, "cudaLimitMaxL2FetchGranularity", 0x05)
    cp.cuda.runtime.deviceSetLimit(lim, int(nbytes))
    return cp.cuda.runtime.deviceGetLimit(lim)


def draw_i32(indptr, indices, urns, draws, rand_vals=None, seed=0, step=0):
    n = urns.shape[0]
    philox = rand_vals is None
    f = _module().get_function(f"draw_i32<4,{'true' if philox else 'false'}>")
    f(_grid(n), (_BLOCK,), (indptr, indices, urns, rand_vals if not philox else np.uint64(0),
                            np.uint64(seed), np.uint32(step), draws, np.int64(n)))


def draw_u16(indptr, indices, words, R, draws, rand_vals=None, seed=0, step=0, lanes=4):
    """words: uint32 array of n * R packed urns (node-major). lanes: threads per row (1, 2, 4 or 8)."""
    n = words.size // R
    philox = rand_vals is None
    f = _module().get_function(f"draw_u16<{lanes},{R},{'true' if philox else 'false'}>")
    f(_grid(n, lanes), (_BLOCK,), (indptr, indices, words, rand_vals if not philox else np.uint64(0),
                            np.uint64(seed), np.uint32(step), draws, np.int64(n)))


def apply_i32(urns, draws, delta=1):
    n = urns.shape[0]
    _module().get_function("apply_i32")(((n + 255) // 256,), (256,), (urns, draws, np.int32(delta), np.int64(n)))


def apply_u16(words, draws, delta=1):
    t = words.size
    _module().get_function("apply_u16")(((t + 255) // 256,), (256,), (words, draws, np.uint32(delta), np.int64(t)))


def apply_u16_mem(words, draws, ring, slot, expire, delta=1, decay=1):
    n = words.size
    wpr = ring.shape[1]
    _module().get_function("apply_u16_mem")(((n + 255) // 256,), (256,),
                                            (words, draws, ring, np.int64(wpr), np.int32(slot),
                                             np.int32(bool(expire)), np.uint32(delta), np.uint32(decay),
                                             np.int64(n)))


def pack_u16(urns_i32):
    """(n, 2) int32 -> n uint32 words (red | black << 16)."""
    u = urns_i32.astype(cp.uint32)
    return u[:, 0] | (u[:, 1] << 16)


def unpack_u16(words):
    return cp.stack([(words & 0xFFFF).astype(cp.int32), (words >> 16).astype(cp.int32)], axis=-1)
