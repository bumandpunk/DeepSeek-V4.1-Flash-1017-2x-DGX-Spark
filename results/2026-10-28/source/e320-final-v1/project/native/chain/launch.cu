#include <torch/extension.h>
#include "chain.cuh"
using namespace tf_exl3x;
template<bool FOLD,int KB>
void launch(std::vector<at::Tensor>& t,std::vector<int64_t>& p,cudaStream_t stream){
 const size_t smem=3*32*(KB*16+8)*sizeof(half)+3*KB*8*4*10*sizeof(uint32_t);
 auto fn=chain_kernel<2,2,10,FOLD,KB>;
 if(smem>48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 dim3 grid(1,p[2]/128,t[10].size(0)*p[0]);
 fn<<<grid,128,smem,stream>>>(reinterpret_cast<const half*>(t[0].data_ptr()),reinterpret_cast<const half*>(t[1].data_ptr()),t[2].data_ptr<int64_t>(),t[3].data_ptr<int64_t>(),t[4].data_ptr<int>(),t[5].data_ptr<int>(),t[6].data_ptr<int>(),t[7].data_ptr<int>(),t[8].data_ptr<int>(),t[9].data_ptr<float>(),p[1],p[2],p[3],t[8].size(1),p[5],t[10].size(0),p[4],p[7],t[10].data_ptr<int>());
}
void run(std::vector<at::Tensor> t,std::vector<int64_t> p){
 TORCH_CHECK(t.size()==11 && p.size()==12,"chain argument contract");
 for(const auto& x:t)TORCH_CHECK(x.is_cuda()&&x.is_contiguous(),"CUDA contiguous required");
 TORCH_CHECK(p[6]==2 && p[8]>=2 && p[9]<=10 && p[2]%128==0 && p[1]%(16*p[4]*p[7]*p[11])==0,"unsupported chain dimensions");
 c10::cuda::CUDAGuard guard(t[0].device());auto stream=at::cuda::getCurrentCUDAStream();
 if(p[10]){if(p[11]==4)launch<true,4>(t,p,stream);else launch<true,2>(t,p,stream);}
 else{if(p[11]==4)launch<false,4>(t,p,stream);else launch<false,2>(t,p,stream);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
