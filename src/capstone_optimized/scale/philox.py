"""
Counter-based random numbers for CUDA kernels (Philox4x32-10, the generator cuRAND's Philox uses).

A kernel computes the random number for (seed, step, node) on the spot instead of reading it from an array,
so a step needs no N-element float64 buffer (573 MB at 71.6M nodes, 8 GB at 1B) and no pass to fill it.
The same (seed, step, node) always gives the same number, on any block layout, so runs are reproducible.

PHILOX_SRC is pasted at the top of a RawModule source. uniform_ref() is a NumPy reference used by the tests.
"""
import numpy as np

PHILOX_SRC = r"""
#ifndef POLYA_PHILOX
#define POLYA_PHILOX
__device__ __forceinline__ uint4 philox4x32_10(uint4 c, uint2 k)
{
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        unsigned int hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
        unsigned int hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
        c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
        k.x += 0x9E3779B9u; k.y += 0xBB67AE85u;
    }
    return c;
}

// Uniform double in [0, 1) with 53 random bits, for (seed, step, node, stream).
__device__ __forceinline__ double philox_uniform(unsigned long long seed, unsigned int step,
                                                 unsigned long long node, unsigned int stream)
{
    uint4 r = philox4x32_10(make_uint4((unsigned int)node, (unsigned int)(node >> 32), step, stream),
                            make_uint2((unsigned int)seed, (unsigned int)(seed >> 32)));
    return ((double)(r.x >> 5) * 67108864.0 + (double)(r.y >> 6)) * (1.0 / 9007199254740992.0);
}
#endif
"""


def _mulhilo(a, b):
    p = np.uint64(a) * np.uint64(b)
    return np.uint32(p >> np.uint64(32)), np.uint32(p & np.uint64(0xFFFFFFFF))


def uniform_ref(seed, step, node, stream=0):
    """NumPy reference for philox_uniform, for one node (slow; tests only)."""
    with np.errstate(over="ignore"):
        c = [np.uint32(node & 0xFFFFFFFF), np.uint32(node >> 32), np.uint32(step), np.uint32(stream)]
        k = [np.uint32(seed & 0xFFFFFFFF), np.uint32(seed >> 32)]
        for _ in range(10):
            hi0, lo0 = _mulhilo(0xD2511F53, c[0])
            hi1, lo1 = _mulhilo(0xCD9E8D57, c[2])
            c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0]
            k = [np.uint32(k[0] + np.uint32(0x9E3779B9)), np.uint32(k[1] + np.uint32(0xBB67AE85))]
    return (float(c[0] >> np.uint32(5)) * 67108864.0 + float(c[1] >> np.uint32(6))) / 9007199254740992.0
