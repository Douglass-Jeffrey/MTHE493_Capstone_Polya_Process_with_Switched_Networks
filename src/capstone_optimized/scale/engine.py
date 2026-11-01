"""
Out-of-core Polya step on the split adjacency format (split_format.py), for graphs whose adjacency does not fit
in VRAM, or in RAM.

Only the urns live in VRAM, as 16-bit red/black counts (4 B per node, 4 GB at 10^9 nodes). The adjacency is
processed in blocks of rows, and each block is read by the draw kernel directly from wherever it lives:
  vram    the whole adjacency in device memory (small graphs)
  pinned  resident in pinned host RAM, read over PCIe by the kernel (zero copy)
  stream  read from the NVMe drive every step by native threads (feeder.dll) into a ring of pinned buffers,
          which the kernel then reads zero copy; a buffer is refilled only after its kernel has finished
  mixed   as many blocks as fit a pinned-RAM budget stay resident, the rest are streamed
The in part's row pointers are kept as one byte per row (in-degree, 255 = look up the exception list) in
pinned RAM, and turned into 32-bit block offsets on the GPU before each block's draw kernel.

Each step: per block, one fused kernel sums the own urn, the m out neighbours and the in neighbours, draws with
the same float64 rule as csr_kernels.draw_two_colour (1 if rand > red/(red+black)) and stores the draw as one
bit; then one kernel adds delta to the drawn colour of every urn, flagging (not wrapping) a 16-bit overflow.
rand is Philox(seed, step, node) computed in the kernel, or, to compare with the repo's step, read from a given
float64 array.
"""
import ctypes
import os
import threading
import time

import cupy as cp
import numpy as np
from cupy.cuda import nvtx

from . import split_format as sf
from .feeder import ALIGN, Feeder
from .philox import PHILOX_SRC

_SRC = PHILOX_SRC + r"""
#define FULL 0xffffffffu

// One block of rows [row0, row0 + nrows). LANES threads per row; 32 / LANES rows per warp, packed into bits.
template<int LANES, int EXT>
__global__ void draw_block(const int* __restrict__ outp, long long out_row0, int m,
                           const unsigned int* __restrict__ in_off, const int* __restrict__ in_idx,
                           const ushort2* __restrict__ urns, long long row0, long long nrows, long long init,
                           unsigned long long seed, unsigned int step, const double* __restrict__ rand_vals,
                           unsigned char* __restrict__ draw_bytes)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long lr = t / LANES;
    int lane = (int)(t & (LANES - 1));
    bool valid = lr < nrows;
    long long row = row0 + lr;
    unsigned long long s0 = 0, s1 = 0;
    if (valid) {
        int mo = row >= init ? m : 0;
        const int* o = outp + (row - out_row0) * m;          // only read when mo > 0
        unsigned int a = in_off[lr];
        int deg = mo + (int)(in_off[lr + 1] - a);
        for (int k = lane; k < deg; k += LANES) {
            int u = k < mo ? o[k] : in_idx[a + (unsigned int)(k - mo)];
            ushort2 v = urns[u];
            s0 += v.x; s1 += v.y;
        }
    }
    for (int off = LANES / 2; off > 0; off >>= 1) {
        s0 += __shfl_down_sync(FULL, s0, off, LANES);
        s1 += __shfl_down_sync(FULL, s1, off, LANES);
    }
    int draw = 0;
    if (valid && lane == 0) {
        ushort2 u = urns[row];
        s0 += u.x; s1 += u.y;
        double m0 = (double)s0, rs = m0 + (double)s1;
        if (rs == 0.0) rs = 1.0;
        double r = EXT ? rand_vals[row] : philox_uniform(seed, step, (unsigned long long)row, 0u);
        draw = r > m0 / rs ? 1 : 0;
    }
    unsigned int bal = __ballot_sync(FULL, draw);
    if ((threadIdx.x & 31) == 0) {
        const int R = 32 / LANES;                             // rows in this warp, row0 is a multiple of 8
        long long wrow = row0 + t / LANES;
        unsigned int w = 0;
        #pragma unroll
        for (int i = 0; i < R; ++i) w |= ((bal >> (i * LANES)) & 1u) << i;
        #pragma unroll
        for (int j = 0; j < R / 8; ++j)
            if (wrow + 8 * j < row0 + nrows) draw_bytes[(wrow >> 3) + j] = (unsigned char)(w >> (8 * j));
    }
}

// Push variant: rows in [init, K) ("early" rows, the hubs) have no in part here. Every out entry (row -> u) with
// u in [init, K) adds the row's urn to S[u] (red in the low 32 bits, black in the high 32), which collects u's
// in-neighbour sum without reading u's in-list; early rows also add their out-part sum to their own S slot and
// are drawn afterwards by draw_early. Integer sums, so the result equals the pull version exactly.
template<int LANES, int EXT>
__global__ void draw_block_push(const int* __restrict__ outp, long long out_row0, int m,
                                const unsigned int* __restrict__ in_off, const int* __restrict__ in_idx,
                                const ushort2* __restrict__ urns, long long row0, long long nrows, long long init,
                                unsigned long long seed, unsigned int step, const double* __restrict__ rand_vals,
                                unsigned char* __restrict__ draw_bytes, long long K,
                                unsigned long long* __restrict__ S)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long lr = t / LANES;
    int lane = (int)(t & (LANES - 1));
    bool valid = lr < nrows;
    long long row = row0 + lr;
    bool early = valid && row >= init && row < K;
    unsigned long long s0 = 0, s1 = 0;
    ushort2 me = make_ushort2(0, 0);
    if (valid) {
        me = urns[row];
        unsigned long long mine = (unsigned long long)me.x | ((unsigned long long)me.y << 32);
        int mo = row >= init ? m : 0;
        const int* o = outp + (row - out_row0) * m;
        unsigned int a = in_off[lr];
        int deg = mo + (int)(in_off[lr + 1] - a);
        for (int k = lane; k < deg; k += LANES) {
            int u = k < mo ? o[k] : in_idx[a + (unsigned int)(k - mo)];
            ushort2 v = urns[u];
            s0 += v.x; s1 += v.y;
            if (k < mo && u >= init && u < K) atomicAdd(&S[u], mine);
        }
    }
    for (int off = LANES / 2; off > 0; off >>= 1) {
        s0 += __shfl_down_sync(FULL, s0, off, LANES);
        s1 += __shfl_down_sync(FULL, s1, off, LANES);
    }
    int draw = 0;
    if (valid && lane == 0) {
        if (early) {
            atomicAdd(&S[row], s0 | (s1 << 32));
        } else {
            s0 += me.x; s1 += me.y;
            double m0 = (double)s0, rs = m0 + (double)s1;
            if (rs == 0.0) rs = 1.0;
            double r = EXT ? rand_vals[row] : philox_uniform(seed, step, (unsigned long long)row, 0u);
            draw = r > m0 / rs ? 1 : 0;
        }
    }
    unsigned int bal = __ballot_sync(FULL, draw);
    if ((threadIdx.x & 31) == 0) {
        const int R = 32 / LANES;
        long long wrow = row0 + t / LANES;
        unsigned int w = 0;
        #pragma unroll
        for (int i = 0; i < R; ++i) w |= ((bal >> (i * LANES)) & 1u) << i;
        #pragma unroll
        for (int j = 0; j < R / 8; ++j)       // bytes of early rows are rewritten by draw_early
            if (wrow + 8 * j < row0 + nrows) draw_bytes[(wrow >> 3) + j] = (unsigned char)(w >> (8 * j));
    }
}

// Draws of the early rows [init, K) (init and K multiples of 8) from S plus their own urn, one byte per thread.
template<int EXT>
__global__ void draw_early(const unsigned long long* __restrict__ S, const ushort2* __restrict__ urns, long long init,
                           long long K, unsigned long long seed, unsigned int step,
                           const double* __restrict__ rand_vals, unsigned char* __restrict__ draw_bytes)
{
    long long b = (init >> 3) + (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b * 8 >= K) return;
    unsigned int w = 0;
    for (int j = 0; j < 8; ++j) {
        long long v = b * 8 + j;
        unsigned long long x = S[v];
        ushort2 u = urns[v];
        double m0 = (double)((x & 0xffffffffULL) + u.x), rs = m0 + (double)((x >> 32) + u.y);
        if (rs == 0.0) rs = 1.0;
        double r = EXT ? rand_vals[v] : philox_uniform(seed, step, (unsigned long long)v, 0u);
        if (r > m0 / rs) w |= 1u << j;
    }
    draw_bytes[b] = (unsigned char)w;
}

extern "C" __global__ void deg8_to_u32(const unsigned char* __restrict__ deg8, long long n,
                                       unsigned int* __restrict__ out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = deg8[i];
}

extern "C" __global__ void apply_bits(ushort2* __restrict__ urns, const unsigned char* __restrict__ bits,
                                      long long n, int delta, unsigned int* __restrict__ flag)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = (bits[i >> 3] >> (i & 7)) & 1;
    ushort2 u = urns[i];
    int x = (d ? (int)u.y : (int)u.x) + delta;
    if (x > 0xFFFF || x < 0) { atomicOr(flag, 1u); x = x < 0 ? 0 : 0xFFFF; }
    if (d) u.y = (unsigned short)x; else u.x = (unsigned short)x;
    urns[i] = u;
}

// Finite memory, as Polya_Process.memory_step: add delta to the drawn colour, then take decay off the colour
// drawn M steps ago (old bits from the host ring).
extern "C" __global__ void apply_bits_mem(ushort2* __restrict__ urns, const unsigned char* __restrict__ bits,
                                          const unsigned char* __restrict__ old_bits, long long n, int delta,
                                          int decay, unsigned int* __restrict__ flag)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = (bits[i >> 3] >> (i & 7)) & 1, o = (old_bits[i >> 3] >> (i & 7)) & 1;
    int c[2] = {(int)urns[i].x, (int)urns[i].y};
    c[d] += delta;
    c[o] -= decay;
    for (int j = 0; j < 2; ++j)
        if (c[j] > 0xFFFF || c[j] < 0) { atomicOr(flag, 1u); c[j] = c[j] < 0 ? 0 : 0xFFFF; }
    urns[i] = make_ushort2((unsigned short)c[0], (unsigned short)c[1]);
}
"""

_BLOCK = 256
_mods = {}


def _kernels(lanes):
    if lanes not in _mods:
        names = [f"draw_block<{lanes},0>", f"draw_block<{lanes},1>", f"draw_block_push<{lanes},0>",
                 f"draw_block_push<{lanes},1>", "draw_early<0>", "draw_early<1>"]
        mod = cp.RawModule(code=_SRC, options=("-std=c++14",), name_expressions=names)
        _mods[lanes] = (mod.get_function(names[0]), mod.get_function(names[1]),
                        mod.get_function("deg8_to_u32"), mod.get_function("apply_bits"),
                        mod.get_function("apply_bits_mem"),
                        [mod.get_function(nm) for nm in names[2:]])
    return _mods[lanes]


class Pinned:
    """Page-locked host buffer (cudaHostAlloc), freed with the object."""

    def __init__(self, nbytes):
        self.nbytes = int(nbytes)
        self.ptr = cp.cuda.runtime.hostAlloc(max(self.nbytes, 1), 0)

    def array(self, dtype, count, offset=0):
        import ctypes
        n = int(count) * np.dtype(dtype).itemsize
        return np.frombuffer((ctypes.c_char * n).from_address(self.ptr + offset), dtype=dtype, count=count)

    def free(self):
        if self.ptr:
            cp.cuda.runtime.freeHost(self.ptr)
            self.ptr = 0

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


def _al_down(x):
    return x // ALIGN * ALIGN


def _al_up(x):
    return (x + ALIGN - 1) // ALIGN * ALIGN


class Engine:
    def __init__(self, d, placement="auto", block_rows=1 << 22, block_mb=None, pinned_budget_gb=8.0, ram_budget_gb=0.0,
                 slots=6, threads=4, chunk_mb=8, lanes=4, seed=0, delta=1, l2_fetch=None, vram_margin_gb=0.4,
                 push_k=0, log=print):
        """block_mb: size blocks by adjacency bytes (early rows hold the hubs' long in-lists) instead of by rows;
        block_rows is then the cap on rows per block.
        push_k: rows [init, push_k) get their in-neighbour sums pushed from the out parts into a VRAM accumulator
        (8 B per row) instead of reading their in-lists, which are then never read; -1 = as many rows as fit in
        VRAM. In a BA graph the first K nodes hold sqrt(K/n) of all edge ends, so this drops a large share of
        the in part from what is read each step."""
        assert block_rows % 256 == 0 and lanes in (1, 2, 4)
        self.d, self.lanes, self.seed, self.delta, self.log = d, lanes, int(seed), int(delta), log
        meta = sf.read_meta(d)
        self.meta = meta
        n, m, init = meta["n"], meta["m"], meta["init"]
        self.n, self.m, self.init = n, m, init
        p = sf.paths(d)
        self.paths = p
        t0 = time.time()
        if l2_fetch is not None:
            cp.cuda.runtime.deviceSetLimit(0x05, int(l2_fetch))     # cudaLimitMaxL2FetchGranularity

        in_ptr = np.memmap(p["in_ptr.i64"], np.int64, "r", shape=(n + 1,))

        # in-degrees as one byte per row in pinned RAM (255 = exception), exceptions per block on the GPU
        self.deg8 = Pinned(n)
        deg8 = self.deg8.array(np.uint8, n)
        esc_rows, esc_vals = [], []
        for c0 in range(0, n, 1 << 25):
            c1 = min(n, c0 + (1 << 25))
            dg = np.diff(np.asarray(in_ptr[c0:c1 + 1]))
            big = np.flatnonzero(dg >= 255)
            deg8[c0:c1] = np.minimum(dg, 255).astype(np.uint8)
            esc_rows.append(big + c0)
            esc_vals.append(dg[big])
        esc_rows, esc_vals = np.concatenate(esc_rows), np.concatenate(esc_vals)

        # push rows: [init, K), K a multiple of 256, sized to the VRAM left after the urns and draw bits
        K = 0
        if push_k:
            assert init % 8 == 0
            if push_k < 0:
                free0 = cp.cuda.runtime.memGetInfo()[0]
                spare = free0 - n * 4 - n // 8 - 8 * (block_rows + 1) - vram_margin_gb * 2 ** 30
                push_k = int(max(0, spare) // 8)
            K = min(n, push_k) // 256 * 256
            # packed 32-bit halves: (in-degree + m) * 65535 must stay below 2^32
            big = (esc_rows >= init) & (esc_rows < K) & (esc_vals + m >= 65535)
            if big.any():
                raise ValueError(f"push rows up to {K} include in-degree {esc_vals[big].max()}, too large to pack")
            if K <= init:
                K = 0
        self.K = K
        if K:
            deg8[init:K] = 0
            keep = (esc_rows < init) | (esc_rows >= K)
            esc_rows, esc_vals = esc_rows[keep], esc_vals[keep]
        ip_init = int(in_ptr[init]) if K else 0
        cut = (int(in_ptr[K]) - ip_init) if K else 0

        def in_eff(r):
            # in_ptr with the push rows' in-lists removed (they are never read)
            r = np.asarray(r)
            v = np.asarray(in_ptr[r]).astype(np.int64)
            if K:
                v = np.where(r <= init, v, np.where(r <= K, ip_init, v - cut))
            return v

        # blocks of rows (multiples of 256), with their byte ranges in out.i32 and in_idx.i32
        if block_mb:
            G = 256
            grid = np.append(np.arange(0, n, G, dtype=np.int64), n)
            cum = m * 4 * np.maximum(grid - init, 0) + 4 * in_eff(grid)
            bounds, gi = [0], 0
            while grid[gi] < n:
                gj = int(np.searchsorted(cum, cum[gi] + block_mb * 2 ** 20, side="right")) - 1
                gi = min(max(gj, gi + 1), gi + block_rows // G, grid.size - 1)
                bounds.append(int(grid[gi]))
        else:
            bounds = list(range(0, n, block_rows)) + [n]
        if K:
            bounds = sorted(set(bounds) | {init, K})
        self.blocks = []
        for r0, r1 in zip(bounds[:-1], bounds[1:]):
            o0, o1 = (max(r0, init) - init) * m * 4, (max(r1, init) - init) * m * 4
            if K and init <= r0 and r1 <= K:
                i0 = i1 = ip_init * 4                       # push rows: no in part
            else:
                i0, i1 = int(in_ptr[r0]) * 4, int(in_ptr[r1]) * 4
            assert (i1 - i0) // 4 < 2 ** 32
            self.blocks.append(dict(r0=r0, r1=r1, out_row0=max(r0, init), o0=o0, o1=o1, i0=i0, i1=i1))
        nb = len(self.blocks)
        for bk in self.blocks:
            s = (esc_rows >= bk["r0"]) & (esc_rows < bk["r1"])
            bk["esc"] = ((cp.asarray(esc_rows[s] - bk["r0"]), cp.asarray(esc_vals[s].astype(np.uint32)))
                         if s.any() else None)
        del in_ptr
        self.n_escapes = int(esc_rows.size)

        # device state
        self.urns = cp.ones((n, 2), dtype=cp.uint16)
        self.bits = cp.zeros((n + 7) // 8 + 8, dtype=cp.uint8)
        self.flag = cp.zeros(1, dtype=cp.uint32)
        maxrows = max(bk["r1"] - bk["r0"] for bk in self.blocks)
        self.deg32 = cp.empty(maxrows, dtype=cp.uint32)
        self.off = cp.zeros(maxrows + 1, dtype=cp.uint32)
        (self.k_draw_philox, self.k_draw_ext, self.k_deg, self.k_apply, self.k_apply_mem,
         (self.k_push_philox, self.k_push_ext, self.k_early_philox, self.k_early_ext)) = _kernels(lanes)
        self.mem_M = 0
        self.S = cp.zeros(K, dtype=cp.uint64) if K else None

        # placement
        out_b = os.path.getsize(p["out.i32"]) if meta["n_out"] else 0
        in_b = os.path.getsize(p["in_idx.i32"])
        free = cp.cuda.runtime.memGetInfo()[0]
        if placement == "auto":
            if out_b + in_b < free - vram_margin_gb * 2 ** 30:
                placement = "vram"
            elif out_b + in_b < pinned_budget_gb * 2 ** 30:
                placement = "pinned"
            else:
                placement = "mixed"
        self.placement = placement
        for bk in self.blocks:
            bk["where"] = {"vram": "vram", "pinned": "pinned", "stream": "stream"}.get(placement)
        if placement == "mixed":
            # tiers: pinned (read by the GPU in place), ram (pageable, copied into a pinned slot by the feeder
            # threads each step), stream (read from the drive each step). Each tier's blocks are spread evenly
            # over the order, so the drive keeps reading while the GPU works on resident blocks
            cost = [_al_up(bk["o1"]) - _al_down(bk["o0"]) + _al_up(bk["i1"]) - _al_down(bk["i0"]) + 2 * ALIGN
                    for bk in self.blocks]

            def spread(cands, budget):
                # the largest k for which k blocks spread evenly over cands fit the budget
                best = []
                for k in range(1, len(cands) + 1):
                    pick = [cands[j] for j in np.unique(np.linspace(0, len(cands) - 1, k).round().astype(int))]
                    if sum(cost[i] for i in pick) > budget:
                        break
                    best = pick
                return set(best)

            pin = spread(list(range(nb)), pinned_budget_gb * 2 ** 30)
            ram = spread([i for i in range(nb) if i not in pin], ram_budget_gb * 2 ** 30)
            for i, bk in enumerate(self.blocks):
                bk["where"] = "pinned" if i in pin else ("ram" if i in ram else "stream")

        self.feeder = Feeder(threads=threads, slots=max(slots, 1) + 1, chunk=chunk_mb << 20)
        self.f_out = self.feeder.open(p["out.i32"]) if out_b else -1
        self.f_in = self.feeder.open(p["in_idx.i32"])
        self.dev = {}
        self.pinned_bytes = 0
        if placement == "vram":
            self.dev["out"] = cp.empty(max(out_b // 4, 1), dtype=cp.int32)
            self.dev["in"] = cp.empty(max(in_b // 4, 1), dtype=cp.int32)
            mm_out = np.memmap(p["out.i32"], np.int32, "r") if out_b else None
            mm_in = np.memmap(p["in_idx.i32"], np.int32, "r")
            for arr, mm in ((self.dev["out"], mm_out), (self.dev["in"], mm_in)):
                if mm is None:
                    continue
                for c0 in range(0, mm.size, 1 << 26):
                    arr[c0:c0 + (1 << 26)].set(np.ascontiguousarray(mm[c0:c0 + (1 << 26)]))
            for bk in self.blocks:
                bk["outp"] = self.dev["out"].data.ptr + bk["o0"]
                bk["inp"] = self.dev["in"].data.ptr + bk["i0"]
        # resident pinned blocks: one buffer each, filled by the feeder through its spare slot
        load_slot = slots
        self.ram_bytes = 0
        self._ram = []
        for bk in self.blocks:
            if bk["where"] not in ("pinned", "ram"):
                continue
            buf, reads, outp, inp = self._layout(bk)
            if bk["where"] == "pinned":
                bk["buf"] = Pinned(buf)
                base = bk["buf"].ptr
                self.pinned_bytes += buf
            else:
                a = np.empty(buf + ALIGN, dtype=np.uint8)
                self._ram.append(a)
                base = _al_up(a.ctypes.data)
                bk["ram_ptr"], bk["ram_len"] = base, buf
                self.ram_bytes += buf
            for f, a0, a1, dst in reads:
                self.feeder.submit(load_slot, f, a0, a1 - a0, base + dst)
            bk["outp"], bk["inp"] = base + outp, base + inp
        self.feeder.wait(load_slot)

        # streamed blocks: ring of pinned slot buffers
        self.streamed = [i for i, bk in enumerate(self.blocks) if bk["where"] in ("stream", "ram")]
        self.nslots = slots if self.streamed else 0
        self.slots, self.slot_ev = [], []
        if self.streamed:
            cap = max(self._layout(self.blocks[i])[0] for i in self.streamed)
            self.slots = [Pinned(cap) for _ in range(self.nslots)]
            self.slot_ev = [None] * self.nslots
            self.pinned_bytes += cap * self.nslots
        self.lookahead = max(1, self.nslots - 2)
        self.j_launch = self.j_submit = 0

        # processing order: streamed blocks spread among resident ones
        self.order = list(range(nb))
        self.step_no = 1
        self.load_s = time.time() - t0
        self.log(f"engine: n={n:,} blocks={nb} placement={placement} "
                 f"(pinned {sum(b['where'] == 'pinned' for b in self.blocks)}, "
                 f"ram {sum(b['where'] == 'ram' for b in self.blocks)}, "
                 f"stream {sum(b['where'] == 'stream' for b in self.blocks)}) "
                 f"pinned={self.pinned_bytes / 2**30:.2f} GiB ram={self.ram_bytes / 2**30:.2f} GiB "
                 f"escapes={self.n_escapes} push K={self.K:,} load {self.load_s:.1f} s")

    def _layout(self, bk):
        """Aligned reads for one block into one buffer: (buffer bytes, reads, out offset, in offset)."""
        reads, pos = [], 0
        outp = 0
        if bk["o1"] > bk["o0"]:
            a0, a1 = _al_down(bk["o0"]), _al_up(bk["o1"])
            reads.append((self.f_out, a0, a1, pos))
            outp = pos + bk["o0"] - a0
            pos += a1 - a0
        a0, a1 = _al_down(bk["i0"]), _al_up(max(bk["i1"], bk["i0"] + 1))
        reads.append((self.f_in, a0, a1, pos))
        inp = pos + bk["i0"] - a0
        pos += a1 - a0
        return pos + ALIGN, reads, outp, inp

    def _submit(self, j):
        s = j % self.nslots
        bk = self.blocks[self.streamed[j % len(self.streamed)]]
        if self.slot_ev[s] is not None:
            self.slot_ev[s].synchronize()
        if bk["where"] == "ram":
            self.feeder.submit_copy(s, bk["ram_ptr"], bk["ram_len"], self.slots[s].ptr)
            return
        _, reads, _, _ = self._layout(bk)
        for f, a0, a1, dst in reads:
            self.feeder.submit(s, f, a0, a1 - a0, self.slots[s].ptr + dst)

    def _launch(self, bk, outp, inp, rand):
        r0, nrows = bk["r0"], bk["r1"] - bk["r0"]
        self.k_deg(((nrows + 255) // 256,), (256,), (np.uint64(self.deg8.ptr + r0), np.int64(nrows), self.deg32))
        if bk["esc"] is not None:
            self.deg32[bk["esc"][0]] = bk["esc"][1]
        cp.cumsum(self.deg32[:nrows], dtype=cp.uint32, out=self.off[1:nrows + 1])
        grid = ((nrows * self.lanes + _BLOCK - 1) // _BLOCK,)
        common = (np.uint64(outp), np.int64(bk["out_row0"]), np.int32(self.m), self.off, np.uint64(inp),
                  self.urns, np.int64(r0), np.int64(nrows), np.int64(self.init), np.uint64(self.seed),
                  np.uint32(self.step_no))
        if self.K:
            extra = (self.bits, np.int64(self.K), self.S)
            if rand is None:
                self.k_push_philox(grid, (_BLOCK,), common + (np.uint64(0),) + extra)
            else:
                self.k_push_ext(grid, (_BLOCK,), common + (rand,) + extra)
        elif rand is None:
            self.k_draw_philox(grid, (_BLOCK,), common + (np.uint64(0), self.bits))
        else:
            self.k_draw_ext(grid, (_BLOCK,), common + (rand, self.bits))

    def enable_memory(self, M, decay=None, ring_path=None):
        """
        Finite memory of M steps (Polya_Process with memory_enabled): each step's draw bits are kept in a ring
        of M rows of n bits in host memory (or a file, ring_path, at 10^9 nodes the ring is 9 GB for M = 72);
        from step M + 1 on, the draw from M steps ago has decay taken off again. One ring row goes to the GPU
        and one comes back per step.
        """
        assert getattr(self, "cpu", None) is None, "memory with the CPU share is not implemented"
        self.mem_M, self.mem_decay = int(M), int(self.delta if decay is None else decay)
        nbytes = self.bits.size
        if ring_path:
            self.ring = np.memmap(ring_path, np.uint8, "w+", shape=(self.mem_M, nbytes))
        else:
            self.ring = np.zeros((self.mem_M, nbytes), np.uint8)
        self.old_bits = cp.zeros(nbytes, cp.uint8)
        self.ring_io = Pinned(nbytes)
        self.ring_buf = self.ring_io.array(np.uint8, nbytes)

    def enable_cpu(self, nblocks, threads=8):
        """
        CPU+GPU: the last nblocks blocks are drawn by CPU threads (native/cpu_step.dll) from a host copy of the
        urns, while the GPU draws the rest. Each step the CPU's draw bits go to the GPU before the urn update,
        and all draw bits come back so the CPU can update its copy (at the start of the next step, while the
        GPU is already drawing). Philox only.
        """
        assert not self.K, "the CPU share with push rows is not implemented"
        if not 0 < nblocks <= len(self.blocks):
            raise ValueError(f"the CPU can draw 1 to {len(self.blocks)} blocks of this graph, not {nblocks}")
        from .native import build as _build
        lib = ctypes.CDLL(_build.build("cpu_step"))
        P, I64, U64 = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_ulonglong
        lib.cpu_draw.argtypes = [P, I64, ctypes.c_int, P, P, P, I64, I64, I64, U64, ctypes.c_uint, P, ctypes.c_int]
        lib.cpu_apply.argtypes = [P, P, I64, ctypes.c_int, ctypes.c_int]
        lib.cpu_apply.restype = ctypes.c_int
        meta, out, in_ptr, in_idx = sf.open_arrays(self.d)
        self.cpu_blocks = list(range(len(self.blocks) - nblocks, len(self.blocks)))
        for i in self.cpu_blocks:
            bk = self.blocks[i]
            r0, r1 = bk["r0"], bk["r1"]
            bk["cpu"] = True
            bk["c_out"] = np.ascontiguousarray(out[bk["o0"] // 4:bk["o1"] // 4]) if bk["o1"] > bk["o0"] else np.zeros(1, np.int32)
            ip = np.asarray(in_ptr[r0:r1 + 1])
            bk["c_ptr"] = ip - ip[0]
            bk["c_in"] = np.ascontiguousarray(in_idx[ip[0]:ip[-1]]) if ip[-1] > ip[0] else np.zeros(1, np.int32)
        self.cpu = lib
        self.cpu_threads = threads
        self.h_urns = self.urns.get()
        self.h_bits_buf = Pinned(self.bits.nbytes)
        self.h_bits = self.h_bits_buf.array(np.uint8, self.bits.size)
        self.cpu_b0 = self.blocks[self.cpu_blocks[0]]["r0"] // 8
        self.cpu_pending = False
        self.cpu_times = []

    def _cpu_work(self):
        t0 = time.time()
        if self.cpu_pending:
            if self.cpu.cpu_apply(self.h_urns.ctypes.data, self.h_bits.ctypes.data, self.n, self.delta,
                                  self.cpu_threads):
                self.cpu_overflow = True
        t1 = time.time()
        for i in self.cpu_blocks:
            bk = self.blocks[i]
            self.cpu.cpu_draw(bk["c_out"].ctypes.data, bk["out_row0"], self.m, bk["c_ptr"].ctypes.data,
                              bk["c_in"].ctypes.data, self.h_urns.ctypes.data, bk["r0"], bk["r1"], self.init,
                              self.seed, self.step_no, self.h_bits.ctypes.data, self.cpu_threads)
        self.cpu_times.append((t1 - t0, time.time() - t1))

    def step(self, rand=None):
        nvtx.RangePush(f"step {self.step_no}")
        th = None
        if getattr(self, "cpu", None) is not None:
            assert rand is None, "the CPU share uses Philox"
            th = threading.Thread(target=self._cpu_work)
            th.start()
        if self.K:
            self.S.fill(0)
        for i in self.order:
            bk = self.blocks[i]
            if bk.get("cpu"):
                continue
            if bk["where"] in ("stream", "ram"):
                j = self.j_launch
                while self.j_submit <= j + self.lookahead:
                    self._submit(self.j_submit)
                    self.j_submit += 1
                s = j % self.nslots
                nvtx.RangePush("wait for block data")
                self.feeder.wait(s)
                nvtx.RangePop()
                _, _, outp, inp = self._layout(bk)
                self._launch(bk, self.slots[s].ptr + outp, self.slots[s].ptr + inp, rand)
                ev = cp.cuda.Event(disable_timing=True)
                ev.record()
                self.slot_ev[s] = ev
                self.j_launch += 1
            else:
                self._launch(bk, bk["outp"], bk["inp"], rand)
        if self.K:
            nbytes = (self.K - self.init) // 8
            args = (self.S, self.urns, np.int64(self.init), np.int64(self.K), np.uint64(self.seed),
                    np.uint32(self.step_no))
            if rand is None:
                self.k_early_philox(((nbytes + 255) // 256,), (256,), args + (np.uint64(0), self.bits))
            else:
                self.k_early_ext(((nbytes + 255) // 256,), (256,), args + (rand, self.bits))
        if th is not None:
            th.join()
            self.bits[self.cpu_b0:].set(self.h_bits[self.cpu_b0:])
        slot = self.step_no % self.mem_M if self.mem_M else 0
        if self.mem_M and self.step_no > self.mem_M:
            self.ring_buf[:] = self.ring[slot]
            self.old_bits.set(self.ring_buf)
            self.k_apply_mem(((self.n + 255) // 256,), (256,), (self.urns, self.bits, self.old_bits, np.int64(self.n),
                                                                 np.int32(self.delta), np.int32(self.mem_decay),
                                                                 self.flag))
        else:
            self.k_apply(((self.n + 255) // 256,), (256,), (self.urns, self.bits, np.int64(self.n),
                                                             np.int32(self.delta), self.flag))
        if self.mem_M:
            self.bits.get(out=self.ring_buf)
            self.ring[slot] = self.ring_buf
        if th is not None:
            self.bits.get(out=self.h_bits)
            self.cpu_pending = True
        self.step_no += 1
        nvtx.RangePop()

    def run(self, steps):
        """Runs steps, returns the time of each (from GPU events, the pipeline is not drained between steps)."""
        evs = [cp.cuda.Event()]
        evs[0].record()
        t0 = time.time()
        for _ in range(steps):
            self.step()
            e = cp.cuda.Event()
            e.record()
            evs.append(e)
        evs[-1].synchronize()
        wall = time.time() - t0
        per = [cp.cuda.get_elapsed_time(a, b) / 1e3 for a, b in zip(evs[:-1], evs[1:])]
        return per, wall

    def overflowed(self):
        return bool(int(self.flag.get()[0])) or getattr(self, "cpu_overflow", False)

    def close(self):
        if self.feeder is not None:
            self.feeder.close()
            self.feeder = None
        for bk in self.blocks:
            if "buf" in bk:
                bk["buf"].free()
        for s in self.slots:
            s.free()
        self.deg8.free()
        self.dev.clear()
        self._ram.clear()
        if self.mem_M:
            self.ring_io.free()
            self.ring = None
