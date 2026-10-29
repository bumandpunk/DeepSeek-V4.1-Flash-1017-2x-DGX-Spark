"""Generate direct-fragment prompt kernels with the upstream decode summation order.
Transforms TensorFold's documented one-chain mma2 algorithm into SK x WK chains,
plus compact worklists. This is an experimental kernel, disabled by default.
"""
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=a.source.read_text()
a.output.mkdir(parents=True,exist_ok=True)
start=s.index('template <int CB, int K2, int S, int KB>\n__device__ __forceinline__ void m2_slice');end=s.index('// Prompt chunks\' gate/up with the input rotation inside',start)
f=s[start:end]
f=f.replace('m2_slice','chain_slice').replace('grouped_mma2_kernel','chain_kernel').replace('M2_ROWS','32').replace('M2_COLS','128').replace('M2_TILES','8')
f=f.replace('template <int CB, int K2, int S, int KB>','template <int CB, int K2, int S, int KB, bool FOLD>')
f=f.replace('float (&acc)[4][4][4]) {','float (&tot)[2][4][4], int per_range, int WK) {\n    float acc[2][4][4] = {}, sums[2][4][4] = {};\n    int left=per_range, chain=0;')
f=f.replace('c += 256','c += 128').replace('i < 4; ++i','i < 2; ++i')
anchor='''    }
    cp_async_wait<0>();
}'''
new='''        left -= KB;
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
}'''
assert f.count(anchor)==1;f=f.replace(anchor,new)
f=f.replace('template <int CB, int LO, int HI>','template <int CB, int LO, int HI, bool FOLD, int KB>')
f=f.replace('__launch_bounds__(256, 1)','__launch_bounds__(128, 2)')
f=f.replace('int K, int N, int P, int maxm, int slots, int nexp) {','int K, int N, int P, int maxm, int slots, int nexp, int SK, int WK, const int* work) {')
f=f.replace('const int u = blockIdx.z % nexp;', 'const int wi=blockIdx.z % nexp;\n    const int u=work[2*wi];')
f=f.replace('const int mb = blockIdx.x;', 'const int mb=work[2*wi+1];')
f=f.replace('m2_rows_bytes<4, 4>()','(size_t)3 * 32 * (KB*16+8) * sizeof(half)')
f=f.replace('float acc[4][4][4];','float acc[2][4][4];')
f=f.replace('chain_slice<CB, K2_, 4, 4>(T, N >> 4, K >> 4, nt0, X, K, rows_sh, live, warp, lane, As, Ts, acc);','chain_slice<CB, K2_, 3, KB, FOLD>(T, N >> 4, K >> 4, nt0, X, K, rows_sh, live, warp, lane, As, Ts, acc, (K >> 4)/(SK*WK), WK);')
(a.output/'chain.cuh').write_text('// Derived numerical primitives: TensorFold (see LICENSES); chain scheduling: Spark1017 Apache-2.0.\n#pragma once\n#include "../persistent/experts_grouped.cuh"\nnamespace tf_exl3x {\n'+f+'\n}\n')
