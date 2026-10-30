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

# CSR build from the (m, 2) edge list (rows, cols): count, place, then sort each row by column so the
# result is identical to scipy's coo -> csr. The edge list may be in pinned host memory (read in order).
_BUILD_SRC = r"""
extern "C" __global__ void csr_count(const int2* __restrict__ edges, long long m, int* __restrict__ deg)
{
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < m; e += (long long)gridDim.x * blockDim.x)
        atomicAdd(&deg[edges[e].x], 1);
}

extern "C" __global__ void csr_fill(const int2* __restrict__ edges, long long m, int* __restrict__ cursor,
                                   int* __restrict__ indices)
{
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < m; e += (long long)gridDim.x * blockDim.x) {
        int2 rc = edges[e];
        indices[atomicAdd(&cursor[rc.x], 1)] = rc.y;
    }
}

// sort key (row - r0, column) for every entry of rows r0..r1, one thread per row
extern "C" __global__ void csr_row_keys(const int* __restrict__ indptr, const int* __restrict__ indices,
                                       long long r0, long long r1, long long* __restrict__ keys)
{
    long long r = r0 + (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= r1) return;
    long long base = indptr[r0], hi = (r - r0) << 32;
    for (long long k = indptr[r]; k < indptr[r + 1]; ++k)
        keys[k - base] = hi | (unsigned int)indices[k];
}

// One level of a breadth-first search. Each unvisited node with a neighbour in level `prev` gets the key
// (smallest rank among those neighbours, node): a queue-based BFS over sorted rows reaches it first from
// that neighbour, and orders a neighbour's children by node, so sorting the keys gives scipy's order
extern "C" __global__ void bfs_level(const int* __restrict__ indptr, const int* __restrict__ indices,
                                    const int* __restrict__ level, const int* __restrict__ rank, int prev,
                                    long long n, long long* __restrict__ key)
{
    long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    long long best = 0x7fffffffffffffffLL;
    if (level[v] < 0) {
        for (long long k = indptr[v]; k < indptr[v + 1]; ++k) {
            int u = indices[k];
            if (level[u] == prev && rank[u] < best) best = rank[u];
        }
        if (best != 0x7fffffffffffffffLL) best = (best << 32) | v;
    }
    key[v] = best;
}

// rows r0..r1 of the renumbered matrix: old row order[r] with every column mapped through inv, written to
// out from position new_indptr[r0]
extern "C" __global__ void relabel_rows(const int* __restrict__ indptr, const int* __restrict__ indices,
                                       const int* __restrict__ order, const int* __restrict__ inv,
                                       const int* __restrict__ new_indptr, long long r0, long long r1,
                                       int* __restrict__ out)
{
    long long r = r0 + (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= r1) return;
    long long o = order[r], b = new_indptr[r] - new_indptr[r0] - indptr[o];
    for (long long k = indptr[o]; k < indptr[o + 1]; ++k)
        out[b + k] = inv[indices[k]];
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


_build = {}
_STRIDE_GRID = (4096,)


def _get_build(name):
    if not _build:
        mod = cp.RawModule(code=_BUILD_SRC, options=("-std=c++14",))
        for nm in ("csr_count", "csr_fill", "csr_row_keys", "bfs_level", "relabel_rows"):
            _build[nm] = mod.get_function(nm)
    return _build[name]


def csr_from_edges(edges, n, indices_on_host):
    """
    indptr, indices of the CSR matrix of an (m, 2) int32 edge list, with every row's columns sorted: the
    same arrays scipy builds. `edges` may be a pinned host array. indices_on_host(m) says where the
    indices go. Returns None if the list has a duplicate edge (scipy would sum it into a weight of 2).
    """
    m = edges.shape[0]
    eptr = _ptr(edges)
    deg = cp.zeros(n + 1, dtype=cp.int32)
    _get_build("csr_count")(_STRIDE_GRID, (_BLOCK,), (eptr, np.int64(m), deg[1:]))
    indptr = cp.cumsum(deg, dtype=cp.int32)
    del deg
    cursor = indptr[:-1].copy()
    if indices_on_host(m):
        indices = np.frombuffer(cp.cuda.alloc_pinned_memory(max(4 * m, 1)), dtype=np.int32, count=m)
    else:
        indices = cp.empty(m, dtype=cp.int32)
    _get_build("csr_fill")(_STRIDE_GRID, (_BLOCK,), (eptr, np.int64(m), cursor, _ptr(indices)))
    del cursor
    # the atomics leave each row's columns in arbitrary order
    if not sort_rows(indptr, indices, n):
        return None
    return indptr, indices


def sort_rows(indptr, indices, n):
    """Sort every row's columns in place, a chunk of rows at a time. Returns False on a duplicate entry."""
    indptr_h = cp.asnumpy(indptr)
    free, _ = cp.cuda.runtime.memGetInfo()
    chunk = max(1 << 20, int(0.4 * free) // 16)        # int64 keys plus the sort's buffer of the same size
    r0 = 0
    while r0 < n:
        r1 = int(np.searchsorted(indptr_h, indptr_h[r0] + chunk, side="right")) - 1
        r1 = min(n, max(r1, r0 + 1))
        s, t = int(indptr_h[r0]), int(indptr_h[r1])
        if t > s:
            keys = cp.empty(t - s, dtype=cp.int64)
            _get_build("csr_row_keys")(((r1 - r0 + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                                       (indptr, _ptr(indices), np.int64(r0), np.int64(r1), keys))
            keys.sort()
            if bool((keys[1:] == keys[:-1]).any()):
                return False
            cols = (keys & 0xFFFFFFFF).astype(cp.int32)
            del keys
            if isinstance(indices, cp.ndarray):
                indices[s:t] = cols
            else:
                cols.get(out=indices[s:t])
        r0 = r1
    return True


def _grid1(n):
    return ((int(n) + _BLOCK - 1) // _BLOCK,)


def bfs_order(indptr, indices, n, source):
    """
    Breadth-first order from `source` (order[new] = old), identical to scipy's breadth_first_order on a CSR
    with sorted rows. Nodes it cannot reach go last, in index order. One kernel launch per BFS level.
    """
    level = cp.full(n, -1, dtype=cp.int32)
    rank = cp.full(n, -1, dtype=cp.int32)
    key = cp.empty(n, dtype=cp.int64)
    order = cp.empty(n, dtype=cp.int32)
    level[source], rank[source], order[0] = 0, 0, source
    filled, lvl = 1, 0
    while filled < n:
        _get_build("bfs_level")(_grid1(n), (_BLOCK,), (indptr, _ptr(indices), level, rank, np.int32(lvl),
                                                      np.int64(n), key))
        found = cp.flatnonzero(key != np.int64(0x7fffffffffffffff))
        cnt = int(found.size)
        if cnt == 0:
            break
        nodes = (cp.sort(key[found]) & 0xFFFFFFFF).astype(cp.int32)
        del found
        order[filled:filled + cnt] = nodes
        rank[nodes] = cp.arange(filled, filled + cnt, dtype=cp.int32)
        level[nodes] = lvl + 1
        filled += cnt
        lvl += 1
    if filled < n:
        order[filled:] = cp.flatnonzero(level < 0).astype(cp.int32)
    return order


def relabel(indptr, indices, order):
    """
    indptr and indices of the graph with node order[r] renamed r, rows not yet sorted (see sort_rows). The
    indices are returned in pinned host memory, so the old and new copies are never both in VRAM. They are
    built a chunk of rows at a time in VRAM and copied out in bulk: the kernel's scattered stores would
    each be a separate PCIe write if it wrote to host memory directly (6 s instead of 0.5 s at 71.6M nodes).
    """
    n = order.size
    m = int(indptr[-1])
    inv = cp.empty(n, dtype=cp.int32)
    inv[order] = cp.arange(n, dtype=cp.int32)
    new_indptr = cp.zeros(n + 1, dtype=cp.int32)
    cp.cumsum(cp.diff(indptr)[order], out=new_indptr[1:])
    out = np.frombuffer(cp.cuda.alloc_pinned_memory(max(4 * m, 1)), dtype=np.int32, count=m)
    new_indptr_h = cp.asnumpy(new_indptr)
    chunk = 64 * 2**20                                 # entries per chunk (256 MB)
    buf = cp.empty(min(chunk, max(m, 1)), dtype=cp.int32)
    r0 = 0
    while r0 < n:
        r1 = int(np.searchsorted(new_indptr_h, new_indptr_h[r0] + chunk, side="right")) - 1
        r1 = min(n, max(r1, r0 + 1))
        s, t = int(new_indptr_h[r0]), int(new_indptr_h[r1])
        if t - s > buf.size:                           # a single row larger than a chunk
            buf = cp.empty(t - s, dtype=cp.int32)
        _get_build("relabel_rows")(_grid1(r1 - r0), (_BLOCK,), (indptr, _ptr(indices), order, inv, new_indptr,
                                                               np.int64(r0), np.int64(r1), buf))
        buf[:t - s].get(out=out[s:t])
        r0 = r1
    return new_indptr, out
