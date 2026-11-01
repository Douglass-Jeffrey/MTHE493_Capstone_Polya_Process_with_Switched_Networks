"""
Out-of-core batched Barabasi-Albert generator writing the split format (split_format.py) directly.

It reproduces Barabasi_Albert_Growth.grow_batch exactly: the same CuPy random calls in the same order (the batch's
randint, then the duplicate-resampling loop), so for the same seed and batch schedule the graph is identical to
the repo's growth + build_csr. What changes is storage. The repo keeps the stub list in the edge buffer (both
directions of every edge, 16 B per edge) and builds a CSR from it. Here only the targets each node chose are kept
(the out part, 4 B per edge), and a stub is resolved arithmetically:

  stub r <  head_size       the initial clique's stubs (src of every ordered pair, then dst)
  stub r >= head_size       t = r - head_size is in the stub region of some earlier batch b, which holds
                            m * size_b source stubs (node start_b + o // m) and then m * size_b target stubs
                            (the batch's final targets, in order), i.e. out[m * (start_b - init) + o - m * size_b]

The out part lives in pinned host memory (written to disk at the end), with a prefix cached in VRAM for the
lookups. The in part (who chose each node, plus the clique for clique nodes) is built afterwards: in-degrees are
counted on the GPU, and the rows are filled in ranges whose entries fit in VRAM, sorted, and written to disk.
"""
import json
import os
import queue
import threading
import time

import numpy as np
import cupy as cp

from . import split_format as sf

SIZES = [8, 128, 2048, 32768, 524288]
GB = 2**30

_SRC = r"""
#define FULL 0xffffffffu

// value of every stub index in idx; nb = number of completed batches
extern "C" __global__ void resolve_stubs(const long long* __restrict__ idx, long long n,
        const int* __restrict__ head, long long head_size,
        const long long* __restrict__ boff, const long long* __restrict__ bstart, const long long* __restrict__ bsize,
        int nb, const int* __restrict__ out_dev, long long dev_cap,
        const unsigned long long* __restrict__ host_chunks, int shift, int m, int init, int* __restrict__ res)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long r = idx[i];
    if (r < head_size) { res[i] = head[r]; return; }
    long long t = r - head_size;
    int lo = 0, hi = nb - 1;                       // largest b with boff[b] <= t
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (boff[mid] <= t) lo = mid; else hi = mid - 1;
    }
    long long o = t - boff[lo], mb = (long long)m * bsize[lo];
    if (o < mb) { res[i] = (int)(bstart[lo] + o / m); return; }
    long long g = (long long)m * (bstart[lo] - init) + (o - mb);
    res[i] = g < dev_cap ? out_dev[g]
                         : ((const int*)host_chunks[g >> shift])[g & ((1LL << shift) - 1)];
}

// same flag as cp.any(cp.diff(cp.sort(targets, axis=1), axis=1) == 0, axis=1): some pair of a row's m targets equal
extern "C" __global__ void row_has_dupes(const int* __restrict__ t, int m, long long rows, bool* __restrict__ out)
{
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int* x = t + r * m;
    bool d = false;
    for (int i = 0; i < m; ++i)
        for (int j = i + 1; j < m; ++j)
            d |= x[i] == x[j];
    out[r] = d;
}

extern "C" __global__ void count_in(const int* __restrict__ out, long long n_out, int* __restrict__ deg)
{
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x; g < n_out; g += stride)
        atomicAdd(&deg[out[g]], 1);
}
// n_out entries of one chunk, whose first entry is global entry g_base

// keys ((t - a) << 32 | v) for every edge v -> t with a <= t < b, appended at *cnt (warp-aggregated)
extern "C" __global__ void collect_in(const int* __restrict__ out, long long n_out, long long g_base, int m,
                                      int init, long long a, long long b, unsigned long long* __restrict__ cnt,
                                      long long* __restrict__ keys)
{
    int lane = threadIdx.x & 31;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long g0 = (long long)blockIdx.x * blockDim.x + threadIdx.x - lane; g0 < n_out; g0 += stride) {
        long long g = g0 + lane;
        bool p = false;
        long long key = 0;
        if (g < n_out) {
            long long t = out[g];
            if (t >= a && t < b) {
                p = true;
                key = ((t - a) << 32) | (long long)(init + (g_base + g) / m);
            }
        }
        unsigned bal = __ballot_sync(FULL, p);
        if (bal) {
            int leader = __ffs(bal) - 1;
            unsigned long long base = 0;
            if (lane == leader) base = atomicAdd(cnt, (unsigned long long)__popc(bal));
            base = __shfl_sync(FULL, base, leader);
            if (p) keys[base + __popc(bal & ((1u << lane) - 1))] = key;
        }
    }
}
"""
_mod = None


def _k(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC)
    return _mod.get_function(name)


def schedule(batches, sizes=SIZES):
    """Batch sizes in growth order, as build_graph.py / test_benchmark run them."""
    return [int(bs) for bs in sizes for _ in range(int(batches))]


def clique_head(init):
    """The stub head _initialize_clique builds: src then dst of every ordered pair i != j."""
    nodes = np.arange(init, dtype=np.int32)
    row, col = np.repeat(nodes, init), np.tile(nodes, init)
    mask = row != col
    return np.concatenate([row[mask], col[mask]]).astype(np.int32), row[mask], col[mask]


class _Mem:
    """Peak host RSS / system-available and VRAM-free sampler."""

    def __init__(self):
        import psutil
        self.p, self.vm = psutil.Process(), psutil.virtual_memory
        self.peak_rss = 0
        self.min_avail = 1e30
        self.min_vram_free = 1e30
        self.by_phase = {}
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def sample(self):
        self.peak_rss = max(self.peak_rss, self.p.memory_info().rss)
        self.min_avail = min(self.min_avail, self.vm().available)

    def vram(self, phase="?"):
        free = cp.cuda.runtime.memGetInfo()[0]
        self.min_vram_free = min(self.min_vram_free, free)
        pool = cp.get_default_memory_pool()
        cur = self.by_phase.get(phase, (1e30, 0))
        self.by_phase[phase] = (min(cur[0], free), max(cur[1], pool.total_bytes()))

    def _run(self):
        while not self._stop.wait(0.2):
            self.sample()

    def stop(self):
        self._stop.set()
        self.th.join()
        self.sample()
        return {"peak_rss_gb": round(self.peak_rss / GB, 2), "min_avail_ram_gb": round(self.min_avail / GB, 2),
                "min_vram_free_gb": round(self.min_vram_free / GB, 2),
                "vram_by_phase_gb (min free, max pool)": {k: (round(a / GB, 2), round(b / GB, 2))
                                                          for k, (a, b) in self.by_phase.items()}}


class PinnedChunks:
    """
    A 1-D int32 array in pinned host memory, made of 1 GiB allocations: on this machine (WDDM, 32 GB RAM) a single
    cudaHostAlloc of 12 GiB fails while ~15 GiB is reachable in 1 GiB pieces. Kernels index it through ptrs, a
    device table of chunk addresses (entry g is in chunk g >> SHIFT).
    """
    SHIFT = 28

    def __init__(self, n):
        ch = 1 << self.SHIFT
        self.n = n
        self.mems = [cp.cuda.alloc_pinned_memory(4 * min(ch, n - a)) for a in range(0, n, ch)] or             [cp.cuda.alloc_pinned_memory(4)]
        self.views = [np.frombuffer(mm, np.int32, min(ch, n - c * ch)) for c, mm in enumerate(self.mems)]
        self.ptrs = cp.asarray(np.array([mm.ptr for mm in self.mems], np.uint64))

    def segments(self, g0, k):
        """(chunk, offset in chunk, offset in source, count) pieces of the global range [g0, g0 + k)."""
        ch, o = 1 << self.SHIFT, 0
        while o < k:
            c, off = divmod(g0 + o, ch)
            cnt = min(k - o, ch - off)
            yield c, off, o, cnt
            o += cnt

    def write(self, path, piece=256 * 2**20):
        with open(path, "wb", buffering=0) as f:
            for v in self.views:
                b = v.view(np.uint8)
                for o in range(0, b.size, piece):
                    f.write(memoryview(b[o:o + piece]))

    def free(self):
        self.views, self.mems, self.ptrs = [], [], None


class _Writer:
    """Writes host arrays to open files on a background thread, so the GPU's next pass overlaps the disk."""

    def __init__(self, depth=2):
        self.q = queue.Queue(maxsize=depth)
        self.err = []
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            f, arr = item
            try:
                if not self.err:
                    f.write(memoryview(arr).cast("B"))
            except Exception as e:
                self.err.append(e)
            self.q.task_done()

    def put(self, f, arr):
        if self.err:
            raise self.err[0]
        self.q.put((f, arr))

    def drain(self):
        self.q.join()
        if self.err:
            raise self.err[0]

    def close(self):
        self.q.put(None)
        self.th.join()


def generate(out_dir, batches, seed=493, m=3, init=64, sizes=SIZES, dev_cache_gb=3.0, key_cap=None,
             key_bytes=30, log=print, write_out_async=True):
    """
    Grows the repo's batched BA graph (Graph(num_nodes=init), Barabasi_Albert_Growth(m, initial_nodes=init),
    cp.random.seed(seed), grow_batch(bs) for bs in schedule) and writes it in the split format to out_dir.
    Returns the meta dict (with timings and memory peaks).
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = sf.paths(out_dir)
    pool, ppool = cp.get_default_memory_pool(), cp.get_default_pinned_memory_pool()
    mem = _Mem()
    T = {}
    t_all = time.time()

    sched = schedule(batches, sizes)
    nb_total = len(sched)
    bsize = np.array(sched, np.int64)
    bstart = init + np.concatenate([[0], np.cumsum(bsize)[:-1]]).astype(np.int64)
    boff = 2 * m * (bstart - init)
    n = int(init + bsize.sum())
    n_out = (n - init) * m
    assert n < 2**31, "node ids must fit in int32"
    log(f"[gen] n={n:,} batches={nb_total} n_out={n_out:,} ({n_out * 4 / GB:.2f} GB)")

    head_h, clq_src, clq_dst = clique_head(init)
    head = cp.asarray(head_h)
    head_size = head_h.size
    boff_d, bstart_d, bsize_d = cp.asarray(boff), cp.asarray(bstart), cp.asarray(bsize)

    # out part: pinned host (all of it, in 1 GiB chunks) + VRAM prefix cache for the stub lookups
    host = PinnedChunks(n_out)
    free_vram = cp.cuda.runtime.memGetInfo()[0]
    dev_cap = int(min(n_out, max(0, min(dev_cache_gb * GB, free_vram - 1.0 * GB)) // 4))
    out_d = cp.empty(max(1, dev_cap), cp.int32)
    log(f"[gen] VRAM cache {dev_cap * 4 / GB:.2f} GB ({dev_cap / max(1, n_out):.0%} of the out part)")
    resolve = _k("resolve_stubs")

    dupes_k = _k("row_has_dupes")

    def has_duplicates(targets):
        rows = targets.shape[0]
        out = cp.empty(rows, cp.bool_)
        if rows:
            dupes_k(((rows + 255) // 256,), (256,), (targets, np.int32(m), np.int64(rows), out))
        return out

    def stub_values(idx, nb):
        res = cp.empty(idx.shape, cp.int32)
        nn = int(idx.size)
        if nn:
            resolve(((nn + 255) // 256,), (256,),
                    (idx, np.int64(nn), head, np.int64(head_size), boff_d, bstart_d, bsize_d, np.int32(nb),
                     out_d, np.int64(dev_cap), host.ptrs, np.int32(host.SHIFT), np.int32(m), np.int32(init),
                     res))
        return res

    # ---- growth: same random calls as grow_batch ----
    cp.random.seed(seed)
    t0 = time.time()
    num_stubs = head_size
    resamples = 0
    stuck_rows = 0
    stream = cp.cuda.get_current_stream()
    for b, bs in enumerate(sched):
        rand_indices = cp.random.randint(0, num_stubs, size=(bs, m), dtype=cp.int64)
        targets = stub_values(rand_indices, b)
        has_dupes = has_duplicates(targets)
        loop_limit = 0
        while cp.any(has_dupes) and loop_limit < 10:
            bad_indices = cp.where(has_dupes)[0]
            num_bad = bad_indices.size
            new_rands = cp.random.randint(0, num_stubs, size=(num_bad, m), dtype=cp.int64)
            targets[bad_indices] = stub_values(new_rands, b)
            has_dupes = has_duplicates(targets)
            loop_limit += 1
            resamples += 1
        if loop_limit == 10:
            stuck_rows += int(cp.count_nonzero(has_dupes))
        flat = cp.ascontiguousarray(targets).reshape(-1)
        g0 = m * (int(bstart[b]) - init)
        k = flat.size
        if g0 < dev_cap:
            kd = min(k, dev_cap - g0)
            out_d[g0:g0 + kd] = flat[:kd]
        for c, off, o, cnt in host.segments(g0, k):
            cp.cuda.runtime.memcpyAsync(host.mems[c].ptr + 4 * off, flat.data.ptr + 4 * o, 4 * cnt,
                                        cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
        num_stubs += 2 * m * bs
        if b % 512 == 0 or b == nb_total - 1:
            mem.vram("growth")
            stream.synchronize()
            log(f"[gen]   batch {b + 1}/{nb_total}  nodes {int(bstart[b]) + bs:,}  {time.time() - t0:.1f} s")
    stream.synchronize()
    T["growth_s"] = time.time() - t0
    log(f"[gen] growth {T['growth_s']:.1f} s, resample rounds {resamples}, rows left with duplicates {stuck_rows}")
    del out_d, targets, flat, rand_indices, has_dupes
    pool.free_all_blocks()

    # write the out part in the background while the GPU builds the in part (both only read the pinned out part)
    t_w = time.time()
    writer_err = []

    def write_out():
        try:
            host.write(paths["out.i32"])
        except Exception as e:      # reported after join
            writer_err.append(e)
    wt = threading.Thread(target=write_out)
    if write_out_async:
        wt.start()
    else:
        write_out()

    # ---- in part, phase 1: in-degrees -> in_ptr ----
    t0 = time.time()
    deg = cp.zeros(n, cp.int32)
    mem.vram("in_count")
    for c, v in enumerate(host.views if n_out else []):
        _k("count_in")((24 * 32,), (256,), (np.uint64(host.mems[c].ptr), np.int64(v.size), deg))
    deg[:init] += init - 1                                   # clique neighbours
    nnz_in = n_out + init * (init - 1)
    chunk = 1 << 25
    carry = 0
    wr = _Writer()
    with open(paths["in_ptr.i64"], "wb", buffering=0) as f:
        f.write(np.zeros(1, np.int64).tobytes())
        for c0 in range(0, n, chunk):
            cs = cp.cumsum(deg[c0:c0 + chunk], dtype=cp.int64)
            cs += carry
            if c0 == 0:
                mem.vram("in_count")
            h = cp.asnumpy(cs)
            carry = int(h[-1])
            wr.put(f, h)
            del cs
        wr.drain()
    assert carry == nnz_in, (carry, nnz_in)
    del deg
    pool.free_all_blocks()
    T["in_count_s"] = time.time() - t0
    log(f"[gen] in-degrees + in_ptr {T['in_count_s']:.1f} s, nnz_in {nnz_in:,}")

    # ---- phase 2: fill row ranges whose keys fit in VRAM ----
    t0 = time.time()
    in_ptr = np.memmap(paths["in_ptr.i64"], np.int64, "r", shape=(n + 1,))
    free_vram = cp.cuda.runtime.memGetInfo()[0]
    if key_cap is None:
        # per entry: keys 8 B + thrust's radix-sort scratch (measured ~16 B: the pool peaked at 5.06 GiB with
        # 22 B budgeted, and WDDM then puts the overflow in system RAM) + the int32 result 4 B; keep 0.5 GB spare
        key_cap = int(max(1 << 20, (free_vram - 0.5 * GB) // key_bytes))
    bounds = [0]
    while bounds[-1] < n:
        a = bounds[-1]
        b_ = int(np.searchsorted(in_ptr, in_ptr[a] + key_cap, side="right")) - 1
        bounds.append(min(n, max(b_, a + 1)))
    keys = cp.empty(int(min(key_cap, nnz_in)) + 1, cp.int64)
    cnt = cp.zeros(1, cp.uint64)
    collect = _k("collect_in")
    with open(paths["in_idx.i32"], "wb", buffering=0) as f:
        for a, b_ in zip(bounds[:-1], bounds[1:]):
            want = int(in_ptr[b_] - in_ptr[a])
            cnt.fill(0)
            for c, v in enumerate(host.views if n_out else []):
                collect((24 * 32,), (256,), (np.uint64(host.mems[c].ptr), np.int64(v.size),
                                             np.int64(c << host.SHIFT), np.int32(m), np.int32(init),
                                             np.int64(a), np.int64(b_), cnt, keys))
            got = int(cnt.get()[0])
            if a < init:                                       # clique entries of rows a .. min(b, init)
                sel = (clq_src >= a) & (clq_src < b_)
                ck = ((clq_src[sel].astype(np.int64) - a) << 32) | clq_dst[sel].astype(np.int64)
                keys[got:got + ck.size] = cp.asarray(ck)
                got += ck.size
            assert got == want, (a, b_, got, want)
            kv = keys[:got]
            kv.sort()
            idx = cp.ascontiguousarray(kv.view(cp.int32)[::2])    # low halves = node ids (little-endian)
            mem.vram("in_fill")
            wr.put(f, cp.asnumpy(idx))
            del kv, idx
        wr.drain()
    wr.close()
    del keys, cnt
    pool.free_all_blocks()
    T["in_fill_s"] = time.time() - t0
    T["in_passes"] = len(bounds) - 1
    log(f"[gen] in part filled in {len(bounds) - 1} passes, {T['in_fill_s']:.1f} s")

    if write_out_async:
        wt.join()
    if writer_err:
        raise writer_err[0]
    T["out_write_wait_s"] = time.time() - t_w
    del in_ptr
    host.free()
    ppool.free_all_blocks()
    T["total_s"] = time.time() - t_all
    memstats = mem.stop()
    disk = sum(os.path.getsize(paths[k]) for k in ("out.i32", "in_ptr.i64", "in_idx.i32"))
    meta = sf.write_meta(out_dir, n, m, init, n_out, nnz_in, seed=int(seed), batches=int(batches),
                         sizes=list(map(int, sizes)), source="scale.gen (bit-identical batched BA growth)",
                         timings={k: (round(v, 2) if isinstance(v, float) else v) for k, v in T.items()},
                         memory=memstats, disk_gb=round(disk / GB, 2),
                         dev_cache_gb=round(dev_cap * 4 / GB, 2), key_cap=int(key_cap),
                         resample_rounds=resamples, rows_with_duplicates=stuck_rows)
    log(f"[gen] done {T['total_s']:.1f} s  {memstats}  disk {disk / GB:.2f} GB")
    return meta
