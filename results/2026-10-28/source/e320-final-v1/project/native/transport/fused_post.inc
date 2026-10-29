// Spark1017, Apache-2.0. The arithmetic order follows TensorFold's MIT _hc_post_only.
// Included in the pinned RDMA implementation's anonymous namespace.
__global__ void collect_post_kernel(const float4* own, int n4, int64_t offset4,
    const char* recv_ring, const uint64_t* flags, const uint64_t* seq, int rank,
    int slots, uint64_t capacity, const uint64_t* abort_word,
    __nv_bfloat16* residual, const float* post, const float* comb, int dims) {
    const uint64_t q=*seq;
    const int slot=(int)(q%slots), peer=1-rank;
    if(threadIdx.x==0)
        for(uint32_t spin=1;load_sys(flags+(size_t)slot*2+peer)<q;++spin)
            if((spin&1023)==0 && load_sys(abort_word))break;
    __syncthreads();
    const float4* remote=reinterpret_cast<const float4*>(recv_ring+((size_t)slot*2+peer)*capacity);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n4;i+=gridDim.x*blockDim.x){
        float4 av=own[i],bv=__ldcv(remote+i);
        const float *a=reinterpret_cast<const float*>(&av),*b=reinterpret_cast<const float*>(&bv);
        const int64_t index=(offset4+i)*4;
        const int row=(int)(index/dims),col=(int)(index%dims);
        float x[4][4],y[4];
#pragma unroll
        for(int j=0;j<4;++j){
            const float sum=rank==0?__fadd_rn(a[j],b[j]):__fadd_rn(b[j],a[j]);
            y[j]=__bfloat162float(__float2bfloat16_rn(sum));
#pragma unroll
            for(int k=0;k<4;++k)x[k][j]=__bfloat162float(residual[((int64_t)row*4+k)*dims+col+j]);
        }
        // Read every input stream before in-place stores. Columns never alias
        // another thread's inputs, including at segment and row boundaries.
#pragma unroll
        for(int k=0;k<4;++k){
            float c0=comb[row*16+k],c1=comb[row*16+4+k],c2=comb[row*16+8+k],c3=comb[row*16+12+k];
            const float ps=post[row*4+k];
#pragma unroll
            for(int j=0;j<4;++j){
                float v=__fmaf_rn(c0,x[0][j],__fmul_rn(c1,x[1][j]));
                v=__fmaf_rn(c2,x[2][j],v);
                v=__fmaf_rn(c3,x[3][j],v);
                v=__fmaf_rn(ps,y[j],v);
                residual[((int64_t)row*4+k)*dims+col+j]=__float2bfloat16_rn(v);
            }
        }
    }
}

void gather_post_impl(int64_t handle,const at::Tensor& send,at::Tensor residual,
                 const at::Tensor& post,const at::Tensor& comb,int depth){
    Gather& g=get(handle);
    TORCH_CHECK(g.world==2,"fused post requires exactly two ranks");
    TORCH_CHECK(send.is_cuda() && send.is_contiguous() && send.scalar_type()==at::kFloat && send.dim()==2,
                "fused post requires contiguous fp32 [rows,dim]");
    const int64_t rows=send.size(0),dims=send.size(1),total4=send.numel()/4;
    TORCH_CHECK(dims>0 && dims%4==0 && rows>0 && send.numel()*4<=(256ll<<20),"unsupported fused post shape");
    TORCH_CHECK(residual.is_contiguous() && residual.scalar_type()==at::kBFloat16 && residual.dim()==3 &&
                residual.size(0)==rows && residual.size(1)==4 && residual.size(2)==dims,"residual shape/type");
    TORCH_CHECK(post.is_contiguous() && comb.is_contiguous() && post.scalar_type()==at::kFloat &&
                comb.scalar_type()==at::kFloat && post.numel()==rows*4 && comb.numel()==rows*16,"coefficient shape/type");
    TORCH_CHECK(send.device()==residual.device() && send.device()==post.device() && send.device()==comb.device(),
                "fused post requires one device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(send.data_ptr())%16==0,"unaligned source");
    if(g.failed.load())TORCH_CHECK(false,"rdma gather: ",g.why);
    const auto stream=at::cuda::getCurrentCUDAStream();
    const int64_t capacity4=g.max_bytes/16;
    const float4* input=reinterpret_cast<const float4*>(send.data_ptr());
    // At most two outstanding segments, and at least twice that many slots:
    // completing the preceding batch proves that the peer passed the older
    // batch before these receive slots may be overwritten. Tickets bind each
    // collect to its own sequence even after the next staging launch advances it.
    depth=std::min(depth,g.slots/2);
    TORCH_CHECK(depth>=1 && depth<=2,"insufficient pipeline ring slots");
    for(int64_t first4=0;first4<total4;first4+=capacity4*depth){
      const int batch=(int)std::min((int64_t)depth,(total4-first4+capacity4-1)/capacity4);
      for(int step=0;step<batch;++step){
        const int64_t offset4=first4+step*capacity4;
        const int n4=(int)std::min(capacity4,total4-offset4);
        const int staging=std::max(1,std::min(16,(n4+1023)/1024));
        stage_kernel<<<staging,STAGE_THREADS,0,stream>>>(input+offset4,n4,dev(g.send_ring),dev(g.sizes),
            dev(g.doorbell),dev(g.sent),g.seq,g.slots,g.max_bytes,nullptr,dev(g.abort_word),g.staged,g.fence_mode,nullptr,g.tickets+step);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      for(int step=0;step<batch;++step){
        const int64_t offset4=first4+step*capacity4;
        const int n4=(int)std::min(capacity4,total4-offset4);
        const int blocks=std::max(1,std::min(128,(n4+255)/256));
        collect_post_kernel<<<blocks,256,0,stream>>>(input+offset4,n4,offset4,dev(g.recv_ring),dev(g.flags),
            g.tickets+step,g.rank,g.slots,g.max_bytes,dev(g.abort_word),reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),
            post.data_ptr<float>(),comb.data_ptr<float>(),(int)dims);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }
}

void gather_post(int64_t h,const at::Tensor& s,at::Tensor r,const at::Tensor& p,const at::Tensor& c){
    gather_post_impl(h,s,r,p,c,1);
}
void gather_post_pipeline(int64_t h,const at::Tensor& s,at::Tensor r,const at::Tensor& p,const at::Tensor& c){
    gather_post_impl(h,s,r,p,c,2);
}
