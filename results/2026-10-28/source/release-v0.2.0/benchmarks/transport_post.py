"""Fused network collect + HC post against the unchanged rank-order reference."""
import argparse,json,os,statistics,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--master',required=True)
p.add_argument('--port',type=int,default=29670);p.add_argument('--out',required=True);p.add_argument('--iters',type=int,default=16)
a=p.parse_args();os.environ['NCCL_PROTO']='Simple';os.environ['NCCL_MIN_NCHANNELS']='4'
import torch
from tensorfold.cuda.comm import NCCL
from tensorfold.families.deepseek_v41.cuda.kernels import hc_post
from spark1017.transport import RdmaGather
torch.cuda.set_device(0);nccl=NCCL(a.rank,2,a.master,a.port)
rdma=RdmaGather(nccl.store,a.rank,2,max_bytes=int(4.25*2**20),prefix='fused_post_test')
result={'rank':a.rank,'slot_bytes':rdma.max_bytes,'cases':[]};D=5120
try:
 for R in [1,17,128,512,2048,4096]:
  g=torch.Generator(device='cuda').manual_seed(1017+R)
  residual=(torch.randn((R,4,D),device='cuda',generator=g)*100).bfloat16()
  post=torch.rand((R,4),device='cuda',generator=g)*2
  comb=torch.rand((R,4,4),device='cuda',generator=g)*2-1
  send=torch.randn((R,D),device='cuda',generator=g)*100*(a.rank+1)
  gathered=torch.empty((2,R,D),device='cuda',dtype=torch.float32)
  out=residual.clone()
  def base():nccl.all_gather(send.view(-1),gathered.view(-1));hc_post(gathered,out,post,comb,out)
  def split():rdma.ext.gather_segmented(rdma.h,send,gathered);hc_post(gathered,out,post,comb,out)
  funcs={'nccl_post':base,'segmented_post':split,'fused_post':lambda:rdma.ext.gather_post(rdma.h,send,out,post,comb),
         'pipelined_post':lambda:rdma.ext.gather_post_pipeline(rdma.h,send,out,post,comb)}
  reference=None
  for name,call in funcs.items():
   for _ in range(3):out.copy_(residual);call()
   torch.cuda.synchronize()
   if reference is None:reference=out.clone()
   diff=torch.count_nonzero(out.view(torch.int16)!=reference.view(torch.int16)).item()
   if diff:
    print(json.dumps({'R':R,'name':name,'different_values':diff,'max_abs':(out-reference).abs().max().item()}),flush=True)
   assert diff==0,(name,R,'eager',diff)
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out.copy_(residual);call()
   for _ in range(5):graph.replay()
   torch.cuda.synchronize()
   assert torch.equal(out.view(torch.int16),reference.view(torch.int16)),(name,R,'graph')
  times={name:[] for name in funcs}
  for iteration in range(a.iters):
   for name in (list(funcs) if iteration%2==0 else list(reversed(funcs))):
    out.copy_(residual);nccl.barrier()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record();funcs[name]();end.record();torch.cuda.synchronize()
    times[name].append(start.elapsed_time(end))
    assert torch.equal(out.view(torch.int16),reference.view(torch.int16)),(name,R,iteration)
  row={'R':R,'D':D,'bytes':send.numel()*4,'bit_exact':True,'graph_exact':True,
       'median_ms':{k:statistics.median(v) for k,v in times.items()},'all_ms':times}
  result['cases'].append(row);Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in row.items() if k!='all_ms'}),flush=True)
 # Changing transaction values detect stale/overwritten ring slots. Artificial
 # rank delays exercise skew; these calls are excluded from timing statistics.
 R=2048;send=torch.empty((R,D),device='cuda');out=torch.empty((R,4,D),device='cuda',dtype=torch.bfloat16)
 post=torch.ones((R,4),device='cuda');comb=torch.zeros((R,4,4),device='cuda')
 for i in range(100):
  send.fill_(i+a.rank);out.zero_()
  if i%7==a.rank:torch.cuda._sleep(1000000)
  rdma.ext.gather_post_pipeline(rdma.h,send,out,post,comb)
  assert bool(torch.all(out==(2*i+1))),(i,'transaction mixing')
 result['skew_stress_transactions']=100;Path(a.out).write_text(json.dumps(result,indent=2))
 nccl.barrier()
finally:rdma.ext.stop(rdma.h)
