"""
Several runs of the networked Polya process on one graph, stepped together (workstream A, 2026-10-31).

The step is limited by the random reads of neighbours' urns: in growth order nearly every read is its own
cache miss, which brings in 64 bytes to use 8. Storing the urns of R runs side by side (R x 4 bytes per node,
16-bit counts) lets one miss serve all R runs, so R runs cost little more than one: at 71.6M nodes, 8 runs
take 43 ms per run per step instead of 269 ms for one (31 ms vs 186 ms on a breadth-first numbered graph;
the repo's fused step takes 290 ms).

Each run has its own random stream (Philox, stream = run index), its own delta and memory decay, and can be
intervened on separately (inject), so runs that share the network but differ in those parameters, or are
repetitions with different seeds, can be batched. Runs that change the network itself (switch-node growth)
cannot share it.

Counts are 16-bit: a run fails loudly (OverflowError) before any count could pass 65535.
"""
import numpy as np
import cupy as cp

from . import quick_kernels as qk

_SRC = r"""
#define FULL 0xffffffffu
// urns, draws: n * R, node-major. ring: M rows of ceil(n * R / 32) words, bit (node * R + run).
extern "C" __global__ void apply_replicas(unsigned int* __restrict__ urns, const unsigned char* __restrict__ draws,
                                          unsigned int* __restrict__ ring, long long words_per_row, int slot,
                                          int use_ring, int expire, const unsigned int* __restrict__ delta,
                                          const unsigned int* __restrict__ decay, int R, long long total)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool d = i < total && draws[i];
    if (i < total) {
        int r = (int)(i % R);
        unsigned int u = urns[i] + (d ? (delta[r] << 16) : delta[r]);
        if (use_ring && expire)
            u -= ((ring[(long long)slot * words_per_row + (i >> 5)] >> lane) & 1u) ? (decay[r] << 16) : decay[r];
        urns[i] = u;
    }
    if (use_ring) {
        unsigned int word = __ballot_sync(FULL, d);
        if (lane == 0 && (i >> 5) < words_per_row) ring[(long long)slot * words_per_row + (i >> 5)] = word;
    }
}
"""
_mod = None


def _apply_kernel():
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC)
    return _mod.get_function("apply_replicas")


class ReplicaPolya:
    """
    R runs of the two-colour process on one unweighted CSR graph (indptr/indices in VRAM, int32).

    deltas, decays: length-R sequences (balls added per draw, removed when a draw expires).
    memory: M > 0 keeps each run's last M draws (one bit each) and removes the draw from M steps ago, as
    Polya_Process(memory_enabled=True, memory_decay_time=M) does; 0 = no memory.
    urns: optional (n, R, 2) initial urns, default one ball of each colour.
    """

    # threads per row: with more runs per neighbour each thread has more work per load, and fewer threads per
    # row put more rows (more independent misses) in flight. Measured at 71.6M nodes (results.md).
    LANES = {1: 4, 2: 2, 4: 2, 8: 1, 16: 1}

    def __init__(self, indptr, indices, R, deltas=None, decays=None, memory=0, seed=0, urns=None):
        assert R in self.LANES, f"R must be one of {sorted(self.LANES)}"
        self.indptr, self.indices, self.R = indptr, indices, R
        self.n = indptr.size - 1
        self.seed, self.step_count = int(seed), 0
        self.deltas = np.asarray(deltas if deltas is not None else [1] * R, np.int64)
        self.decays = np.asarray(decays if decays is not None else self.deltas, np.int64)
        assert self.deltas.shape == (R,) and self.decays.shape == (R,) and (self.deltas >= 0).all()
        self._delta_d = cp.asarray(self.deltas.astype(np.uint32))
        self._decay_d = cp.asarray(self.decays.astype(np.uint32))
        if urns is None:
            self.words = cp.full(self.n * R, 1 | (1 << 16), cp.uint32)
            self._max_count = 1
        else:
            urns = cp.asarray(urns)
            assert urns.shape == (self.n, R, 2) and int(urns.min()) >= 0
            self._max_count = int(urns.max())
            self._check(self._max_count)
            self.words = qk.pack_u16(urns.reshape(-1, 2))
        self.draws = cp.empty(self.n * R, cp.uint8)
        self.M = int(memory)
        wpr = (self.n * R + 31) // 32
        self.ring = cp.zeros((max(self.M, 1), wpr), cp.uint32) if self.M else cp.zeros((1, 1), cp.uint32)

    @staticmethod
    def _check(c):
        if c > 0xFFFF:
            raise OverflowError(f"an urn count could reach {c}, more than 16-bit counts hold")

    def step(self):
        # bound on any count after this step: each step adds at most max(delta) to a count, and with memory
        # a count holds at most M steps' worth of draws on top of what it started with
        grown = (self.step_count + 1) if not self.M else min(self.step_count + 1, self.M)
        self._check(self._max_count + grown * int(self.deltas.max()))
        k = self.step_count + 1                      # the repo's step_count is 1-based
        qk.draw_u16(self.indptr, self.indices, self.words, self.R, self.draws, None, seed=self.seed,
                    step=self.step_count, lanes=self.LANES[self.R])
        total = self.n * self.R
        _apply_kernel()(((total + 255) // 256,), (256,),
                        (self.words, self.draws, self.ring, np.int64(self.ring.shape[1]),
                         np.int32(k % self.M if self.M else 0), np.int32(bool(self.M)),
                         np.int32(bool(self.M) and k > self.M), self._delta_d, self._decay_d,
                         np.int32(self.R), np.int64(total)))
        self.step_count += 1

    def urns(self, run=None):
        """(n, 2) int32 urns of one run, or (n, R, 2) for all."""
        u = qk.unpack_u16(self.words).reshape(self.n, self.R, 2)
        return u if run is None else u[:, run]

    def inject(self, run, nodes, red=0, black=0):
        """Adds balls to the given nodes' urns in one run (e.g. the direct-injection intervention)."""
        idx = cp.asarray(nodes, dtype=cp.int64) * self.R + run
        self._max_count += max(int(red), int(black))
        self._check(self._max_count)
        self.words[idx] += np.uint32(int(red) | (int(black) << 16))
