// CPU share of the out-of-core Polya step (scale/engine.py, hybrid mode): the same gather, Philox draw and
// urn update as the GPU kernels, on a host copy of the 16-bit urns, with OpenMP threads. Integer sums and the
// same float64 rule give the same draws as the GPU for the same (seed, step, node).
#include <algorithm>
#include <cstdint>
#include <immintrin.h>

static inline double philox_uniform(uint64_t seed, uint32_t step, uint64_t node)
{
    uint32_t c0 = (uint32_t)node, c1 = (uint32_t)(node >> 32), c2 = step, c3 = 0;
    uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
    for (int i = 0; i < 10; ++i) {
        uint64_t p0 = (uint64_t)0xD2511F53u * c0, p1 = (uint64_t)0xCD9E8D57u * c2;
        uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0, n2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1;
        c0 = n0; c1 = (uint32_t)p1; c2 = n2; c3 = (uint32_t)p0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return ((double)(c0 >> 5) * 67108864.0 + (double)(c1 >> 6)) * (1.0 / 9007199254740992.0);
}

extern "C" {

// Draws for rows [r0, r1) (r0 a multiple of 8) into bits (one bit per row, indexed by global row).
// out: out part starting at row out_row0 (m per row); in_ptr: r1 - r0 + 1 offsets into in_idx for these rows.
// Each thread walks a run of rows in order and prefetches the urns of the neighbours PF rows ahead, so several
// random urn reads are in flight per thread.
__declspec(dllexport) void cpu_draw(const int32_t* out, int64_t out_row0, int m, const int64_t* in_ptr,
                                    const int32_t* in_idx, const uint16_t* urns, int64_t r0, int64_t r1,
                                    int64_t init, uint64_t seed, uint32_t step, uint8_t* bits, int threads)
{
    const int64_t RUN = 4096, PF = 6;
    int64_t nrun = (r1 - r0 + RUN - 1) / RUN;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int64_t q = 0; q < nrun; ++q) {
        int64_t a = r0 + q * RUN, b = std::min(r1, a + RUN);
        uint32_t byte = 0;
        for (int64_t row = a; row < b; ++row) {
            int64_t pr = row + PF;
            if (pr < b) {
                if (pr >= init) {
                    const int32_t* o = out + (pr - out_row0) * m;
                    for (int k = 0; k < m; ++k) _mm_prefetch((const char*)(urns + 2 * (int64_t)o[k]), _MM_HINT_T0);
                }
                for (int64_t k = in_ptr[pr - r0], e = in_ptr[pr - r0 + 1]; k < e && k < in_ptr[pr - r0] + 16; ++k)
                    _mm_prefetch((const char*)(urns + 2 * (int64_t)in_idx[k]), _MM_HINT_T0);
            }
            uint64_t s0 = urns[2 * row], s1 = urns[2 * row + 1];
            if (row >= init) {
                const int32_t* o = out + (row - out_row0) * m;
                for (int k = 0; k < m; ++k) {
                    int64_t u = o[k];
                    s0 += urns[2 * u]; s1 += urns[2 * u + 1];
                }
            }
            for (int64_t k = in_ptr[row - r0], e = in_ptr[row - r0 + 1]; k < e; ++k) {
                int64_t u = in_idx[k];
                s0 += urns[2 * u]; s1 += urns[2 * u + 1];
            }
            double m0 = (double)s0, rs = m0 + (double)s1;
            if (rs == 0.0) rs = 1.0;
            if (philox_uniform(seed, step, (uint64_t)row) > m0 / rs) byte |= 1u << (row & 7);
            if ((row & 7) == 7 || row + 1 == b) {
                bits[row >> 3] = (uint8_t)byte;
                byte = 0;
            }
        }
    }
}

// urns[i, bit_i] += delta for every row; returns 1 if a count left the 16-bit range (clamped).
__declspec(dllexport) int cpu_apply(uint16_t* urns, const uint8_t* bits, int64_t n, int delta, int threads)
{
    int bad = 0;
#pragma omp parallel for schedule(static) num_threads(threads) reduction(| : bad)
    for (int64_t i = 0; i < n; ++i) {
        int d = (bits[i >> 3] >> (i & 7)) & 1;
        int x = (int)urns[2 * i + d] + delta;
        if (x > 0xFFFF || x < 0) { bad = 1; x = x < 0 ? 0 : 0xFFFF; }
        urns[2 * i + d] = (uint16_t)x;
    }
    return bad;
}

}  // extern "C"
