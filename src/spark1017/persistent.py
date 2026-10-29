"""Experimental exact numerical kernels with ready-expert scheduling."""
from functools import lru_cache
from pathlib import Path
import os
import torch

@lru_cache(None)
def extension():
    from tensorfold.cuda.build import load
    root=Path(__file__).resolve().parents[2]/'native/persistent'
    return load(name='spark1017_persistent_v1',sources=[str(root/'bindings.cpp'),str(root/'queue.cu')],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)

def routed(ext,x,pick,wts,ex,s,out,R,ids,members,limit,act_mode,add,before_down=None):
    # Dedicated queue only supports the tested mul1 layout. Caller enforces guard.
    D,I,E,slots=ex.dims,ex.width,ex.count,s.slots
    if out is None and wts is not None:out=torch.empty((R,D),device=x.device,dtype=torch.float32)
    w=s.no_y if wts is None else wts;a=s.no_y if add is None else add;o=s.no_y if out is None else out
    if not hasattr(s,'spark_queue'):s.spark_queue=torch.zeros((2,),dtype=torch.int32,device=x.device)
    ext.decode_prep(x,x.stride(0),pick,ex.suh_g,ex.suh_u,s.xg,s.xu,ids,s.count,members,R,D,slots,E,w,s.y,a,o,int(wts is not None),int(add is not None),1,s.epoch,1)
    extension().pipeline([s.xg,s.xu,s.xd,ex.gate_ptr,ex.up_ptr,ex.down_ptr,ex.gate_k2,ex.up_k2,ex.down_k2,ids,s.count,members,s.z,pick,ex.svh_g,ex.svh_u,ex.suh_d,ex.svh_d,s.y,w,a,o,s.cnt_gu,s.cnt_d,s.ready,s.ready_cnt,s.epoch,s.spark_queue],
       [D,I,R*slots,s.cfg_gu[2],slots,E,ex.cb,int(wts is not None),int(add is not None),act_mode,1,int(os.environ.get('SPARK1017_WORKERS','4'))],float(limit))
    return out if wts is not None else s.y[:R*slots]
