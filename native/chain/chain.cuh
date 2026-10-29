// Derived numerical primitives: TensorFold (see LICENSES); chain scheduling: Spark1017 Apache-2.0.
#pragma once
#include "../persistent/experts_grouped.cuh"
namespace tf_exl3x {
template <int CB, int K2, int S, int KB, bool FOLD>
__device__ __forceinline__ void chain_slice(const uint32_t* __restrict__ T, int NTILES, int KT, int nt0,
                                         const half* __restrict__ X, int K, const int* rows_sh, int live, int warp,
                                         int lane, half* As, uint32_t* Ts, float (&tot)[2][4][4], int per_range, int WK) {
    float acc[2][4][4] = {}, sums[2][4][4] = {};
    int left=per_range, chain=0;
    constexpr int TW = Fmt<K2>::TW;
    constexpr int LD = KB * 16 + 8;                           // halfs a staged row (padded)
    constexpr int CPR = KB * 2;                               // 16-byte chunks a row a stage
    constexpr int TCH = 8 * TW / 4;                    // 16-byte chunks of a k tile's words for the program
    const LaneMap<K2> map(lane);
    const int NS = KT / KB;
    const uint32_t* tsrc = T + (size_t)nt0 * TW;
    // ldmatrix: lane l reads row (l & 7) + 8 ((l >> 3) & 1) of a 16-row tile, columns 8 (l >> 4) of a k tile
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;

    auto load = [&](int st) {
        const int slot = st % S;
        half* as = As + (size_t)slot * 32 * LD;
        const int kb = st * KB * 16;
        for (int c = threadIdx.x; c < live * 16 * CPR; c += 128) {
            const int r = c / CPR, q = c % CPR;
            const int p = rows_sh[r];
            cp_async16(as + r * LD + q * 8, X + (size_t)(p < 0 ? 0 : p) * K + kb + q * 8, p >= 0);
        }
        uint32_t* ts = Ts + (size_t)slot * KB * 8 * TW;
        for (int c = threadIdx.x; c < KB * TCH; c += 128) {
            const int kk = c / TCH, q = c % TCH;
            cp_async16(ts + kk * 8 * TW + q * 4, tsrc + ((size_t)(st * KB + kk) * NTILES) * TW + q * 4, true);
        }
    };
#pragma unroll
    for (int st = 0; st < S - 1; ++st) {
        if (st < NS) load(st);
        cp_async_commit();
    }
    for (int st = 0; st < NS; ++st) {
        cp_async_wait<S - 2>();
        __syncthreads();                                      // stage st landed; every warp is done with st - 1
        if (st + S - 1 < NS) load(st + S - 1);
        cp_async_commit();
        const int slot = st % S;
        const half* as = As + (size_t)slot * 32 * LD;
        const uint32_t* ts = Ts + (size_t)slot * KB * 8 * TW + (size_t)(2 * warp) * TW;
#pragma unroll
        for (int kk = 0; kk < KB; ++kk) {
            uint32_t bb[2][4];
            m2_decode<CB, K2>(ts + kk * 8 * TW, TW, map, lane, bb);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                if (i < live) {
                    uint32_t a[4];
                    ldmatrix_x4(a, as + (size_t)(16 * i + lrow) * LD + kk * 16 + lcol);
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const uint32_t bl[2] = {bb[j][0], bb[j][1]}, bh[2] = {bb[j][2], bb[j][3]};
                        mma16816(acc[i][2 * j], a, bl);
                        mma16816(acc[i][2 * j + 1], a, bh);
                    }
                }
            }
        }
        left -= KB;
        if (left == 0) {
#pragma unroll
            for (int i=0;i<2;++i)
#pragma unroll
                for (int q=0;q<4;++q)
#pragma unroll
                    for (int j=0;j<4;++j) {
                        sums[i][q][j] = chain == 0 ? acc[i][q][j] : sums[i][q][j] + acc[i][q][j];
                        acc[i][q][j] = 0.f;
                    }
            left = per_range;
            if (++chain == WK) {
                chain=0;
#pragma unroll
                for (int i=0;i<2;++i)
#pragma unroll
                    for (int q=0;q<4;++q)
#pragma unroll
                        for (int j=0;j<4;++j)
                            tot[i][q][j] = FOLD ? tot[i][q][j] + sums[i][q][j] : sums[i][q][j];
            }
        }
    }
    cp_async_wait<0>();
}

template <int CB, int LO, int HI, bool FOLD, int KB>
__global__ void __launch_bounds__(128, 2) chain_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int maxm, int slots, int nexp, int SK, int WK, const int* work) {
    const int wi=blockIdx.z % nexp;
    const int u=work[2*wi];
    if (u >= ucount[0]) return;
    const int mb=work[2*wi+1];
    const int mat = blockIdx.z / nexp;
    extern __shared__ __align__(16) unsigned char m2_smem[];
    __shared__ int rows_sh[32];
    __shared__ int live_sh;
    if (threadIdx.x < 32) {
        const int m = mb * 32 + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                               // members come first, so this block is empty
    if (threadIdx.x == 0) {
        int n = 0;
        while (n < 32 && rows_sh[n] >= 0) ++n;
        live_sh = (n + 15) >> 4;
    }
    __syncthreads();
    const int live = live_sh;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, t = lane & 3;
    const int nt0 = blockIdx.y * 8;
    half* As = reinterpret_cast<half*>(m2_smem);
    uint32_t* Ts = reinterpret_cast<uint32_t*>(m2_smem + (size_t)3 * 32 * (KB*16+8) * sizeof(half));

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][q][c] = 0.f;

    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            chain_slice<CB, K2_, 3, KB, FOLD>(T, N >> 4, K >> 4, nt0, X, K, rows_sh, live, warp, lane, As, Ts, acc, (K >> 4)/(SK*WK), WK);       \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }
    // Z [mats, P, N]: rows g8 and g8 + 8 of each 16-row tile, columns 8 q + 2 t of the warp's 32
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (i >= live) break;
        const int pa = rows_sh[16 * i + g8], pb = rows_sh[16 * i + g8 + 8];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int col = nt0 * 16 + warp * 32 + q * 8 + 2 * t;
            if (pa >= 0)
                *reinterpret_cast<float2*>(Z + ((size_t)mat * P + pa) * N + col) = make_float2(acc[i][q][0], acc[i][q][1]);
            if (pb >= 0)
                *reinterpret_cast<float2*>(Z + ((size_t)mat * P + pb) * N + col) = make_float2(acc[i][q][2], acc[i][q][3]);
        }
    }
}


}
