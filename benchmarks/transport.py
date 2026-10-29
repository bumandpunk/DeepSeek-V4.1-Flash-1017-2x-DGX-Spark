"""Paired two-rank transport controls; byte equality before interleaved timing."""
import argparse,json,os,statistics,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--master',required=True)
p.add_argument('--port',type=int,default=29670);p.add_argument('--protocol',default='');p.add_argument('--out',required=True)
p.add_argument('--iters',type=int,default=16);a=p.parse_args()
if a.protocol:os.environ['NCCL_PROTO']=a.protocol
os.environ['NCCL_MIN_NCHANNELS']='4'
import torch
from tensorfold.cuda.comm import NCCL
from tensorfold.cuda.rdma import RdmaGather
from spark1017.transport import RdmaGather as Segmented
torch.cuda.set_device(0)
nccl=NCCL(a.rank,2,a.master,a.port)
cap=int(4.25*2**20)
small=Segmented(nccl.store,a.rank,2,max_bytes=cap,prefix='spark_test_segment')
large=RdmaGather(nccl.store,a.rank,2,max_bytes=96<<20,prefix='spark_test_large')
result={'protocol':a.protocol or 'NCCL default','rank':a.rank,'slot_bytes':cap,
        'registered_ring_bytes_small':cap*4*3,'registered_ring_bytes_large':(96<<20)*4*3,'cases':[]}
try:
 for nbytes in [16,196608,cap-16,cap,cap+16,8<<20,40<<20,80<<20]:
  n=nbytes//4;indices=torch.arange(n,dtype=torch.int32,device='cuda')
  tag=lambda r:0x40010000+r*0x1010000
  send=(indices^tag(a.rank)).view(torch.float32)
  expect=torch.cat([(indices^tag(r)).view(torch.float32) for r in range(2)])
  out=torch.empty_like(expect)
  funcs={'nccl':lambda:nccl.all_gather(send,out),
         'large_ring':lambda:large.all_gather(send,out),
         'segmented':lambda:small.ext.gather_segmented(small.h,send,out)}
  graphs={}
  for name,call in funcs.items():
   for _ in range(3):call()
   torch.cuda.synchronize()
   assert torch.equal(out.view(torch.int32),expect.view(torch.int32)),(name,nbytes,'eager')
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):call()
   for _ in range(5):graph.replay()
   torch.cuda.synchronize()
   assert torch.equal(out.view(torch.int32),expect.view(torch.int32)),(name,nbytes,'graph')
   graphs[name]=graph
  times={name:[] for name in funcs};wall={name:[] for name in funcs}
  for iteration in range(a.iters):
   for name in (list(funcs) if iteration%2==0 else list(reversed(funcs))):
    nccl.barrier()
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    t=time.perf_counter();begin.record();funcs[name]();end.record();torch.cuda.synchronize()
    times[name].append(begin.elapsed_time(end));wall[name].append((time.perf_counter()-t)*1000)
    assert torch.equal(out.view(torch.int32),expect.view(torch.int32)),(name,nbytes,iteration)
  row={'bytes':nbytes,'bit_exact':True,'graph_exact':True,'gpu_median_ms':{k:statistics.median(v) for k,v in times.items()},
       'wall_median_ms':{k:statistics.median(v) for k,v in wall.items()},'all_gpu_ms':times}
  result['cases'].append(row);Path(a.out).write_text(json.dumps(result,indent=2))
  print(json.dumps({k:v for k,v in row.items() if k!='all_gpu_ms'}),flush=True)
 nccl.barrier()
finally:
 small.ext.stop(small.h);large.ext.stop(large.h)
