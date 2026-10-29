// Spark1017 ready-expert work queue. Apache-2.0; numerical job body: TensorFold MIT.
#include <torch/extension.h>
#include "spark_job.cuh"
using namespace tf_exl3x;

// A worker only claims a down job after that expert published every GU column.
// Unready down jobs never occupy the finite worker pool. GU exhaustion cannot
// deadlock: any unfinished GU job is held by a worker and has no down dependency.
__global__ __launch_bounds__(128, 4) void expert_pipeline(
    GroupedArgs gu, GroupedArgs down, DecodeEpi eg, DecodeEpi ed, int* counters) {
    constexpr int RING = 4 * 3 * 8 * 4 * 10;
    constexpr int REDUCE = 4 * 8 * 8 * 16;
    constexpr int WORDS = RING > REDUCE ? RING : REDUCE;
    __shared__ __align__(16) uint32_t smem[WORDS];
    __shared__ int rows[16], last, kind, task;
    const int mt = (gu.maxm + 15) / 16;
    const int gy = gu.N / 128, gz = 2 * gu.SK * mt;
    const int dy = down.N / 128, dz = mt;
    const int nu = *gu.ucount;
    const int ng = nu * gy * gz, nd = nu * dy * dz;
    const int epoch = *eg.epoch;
    while (true) {
        if (threadIdx.x == 0) {
            kind = -1;
            while (kind < 0) {
                int d = atomicAdd(counters + 1, 0);
                if (d >= nd) { kind = 2; break; }
                int u = nu - 1 - d / (dy * dz);
                if (ld_acquire(eg.ready + u) == epoch && atomicCAS(counters + 1, d, d + 1) == d) {
                    kind = 1; task = d; break;
                }
                int g = atomicAdd(counters, 0);
                if (g < ng) {
                    g = atomicAdd(counters, 1);
                    if (g < ng) { kind = 0; task = g; break; }
                }
                __nanosleep(64);
            }
        }
        __syncthreads();
        if (kind == 2) return;
        if (kind == 0) {
            spark_cp_job<2,8,4,3,2,10,1>(gu.x0,gu.x1,gu.tp0,gu.tp1,gu.k2_0,gu.k2_1,gu.uids,gu.ucount,
                gu.members,gu.z,gu.K,gu.N,gu.P,gu.SK,gu.maxm,gu.slots,eg,
                task/(gy*gz),task%(gy*gz),gy,gz,smem,rows,last);
        } else {
            spark_cp_job<2,8,4,3,2,10,2>(down.x0,down.x1,down.tp0,down.tp1,down.k2_0,down.k2_1,
                down.uids,down.ucount,down.members,down.z,down.K,down.N,down.P,down.SK,
                down.maxm,down.slots,ed,task/(dy*dz),task%(dy*dz),dy,dz,smem,rows,last);
        }
        __syncthreads();
    }
}

void pipeline(std::vector<at::Tensor> t, std::vector<int64_t> p, double limit) {
    TORCH_CHECK(t.size()==28 && p.size()==12, "pipeline argument contract");
    for (const auto& x:t) TORCH_CHECK(x.is_cuda() && x.is_contiguous(),"CUDA contiguous tensors required");
    const int D=p[0],I=p[1],P=p[2],sk=p[3],slots=p[4],E=p[5];
    TORCH_CHECK(p[6]==2 && D%256==0 && I%128==0 && slots<=32 && t[11].size(1)<=128,"unsupported shape/codebook");
    c10::cuda::CUDAGuard guard(t[0].device());
    auto hp=[&](int i){return reinterpret_cast<const half*>(t[i].data_ptr());};
    auto ip=[&](int i){return t[i].data_ptr<int>();};
    auto fp=[&](int i){return t[i].data_ptr<float>();};
    auto lp=[&](int i){return t[i].data_ptr<int64_t>();};
    GroupedArgs g{};g.x0=hp(0);g.x1=hp(1);g.tp0=lp(3);g.tp1=lp(4);g.k2_0=ip(6);g.k2_1=ip(7);
    g.uids=ip(9);g.ucount=ip(10);g.members=ip(11);g.z=fp(12);g.K=D;g.N=I;g.P=P;g.SK=sk;g.slots=slots;g.maxm=t[11].size(1);
    GroupedArgs d=g;d.x0=d.x1=hp(2);d.tp0=d.tp1=lp(5);d.k2_0=d.k2_1=ip(8);d.K=I;d.N=D;d.SK=1;
    DecodeEpi eg{};eg.pick=ip(13);eg.E=E;eg.svh_g=hp(14);eg.svh_u=hp(15);eg.suh_d=hp(16);
    eg.xd=const_cast<half*>(hp(2));eg.limit=limit;eg.act_mode=p[9];eg.cnt=ip(22);
    eg.ready=ip(24);eg.ready_cnt=ip(25);eg.epoch=ip(26);eg.discard=p[10] && sk<=4;
    DecodeEpi ed=eg;ed.svh_d=hp(17);ed.y=fp(18);ed.wts=p[7]?fp(19):nullptr;ed.add=p[8]?fp(20):nullptr;
    ed.out=p[7]?fp(21):nullptr;ed.cnt=ip(23);ed.discard=p[10] && p[7] && slots<=8;
    const auto stream=at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(ip(27),0,2*sizeof(int),stream));
    int device;C10_CUDA_CHECK(cudaGetDevice(&device));
    int sms;C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    expert_pipeline<<<sms*p[11],128,0,stream>>>(g,d,eg,ed,ip(27));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
