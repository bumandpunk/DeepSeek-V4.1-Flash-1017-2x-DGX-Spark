"""Direct warp fragments, preserving the decode path's FP32 chain order."""
from functools import lru_cache
from pathlib import Path
@lru_cache(None)
def extension():
 from tensorfold.cuda.build import load
 root=Path(__file__).resolve().parents[2]/'native/chain'
 return load(name='spark1017_chain_v1',sources=[str(root/'bindings.cpp'),str(root/'launch.cu')],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)
def run(x0,x1,p0,p1,b0,b1,ids,count,members,z,mats,K,N,P,SK,slots,cb,WK,lo,hi,fold,work):
 chain=K//16//SK//WK;kb=4 if chain%4==0 else 2
 extension().run([x0,x1,p0,p1,b0,b1,ids,count,members,z,work],[mats,K,N,P,SK,slots,cb,WK,lo,hi,fold,kb])
