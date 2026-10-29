"""Verify split dependency scheduling against the original prompt HC kernels."""
import argparse,json,statistics
from pathlib import Path
import torch
from tensorfold.families.deepseek_v41.cuda import kernels as K
from spark1017.postmix import after_post
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
D=5120;results=[]
with torch.inference_mode():
 for R in (1,17,128,512,2048):
  torch.manual_seed(R+1017)
  h=torch.randn((R,4,D),device='cuda').bfloat16();fn=torch.randn((24,4*D),device='cuda')*.01
  scale=torch.tensor([.1,.1,.1],device='cuda');base=torch.randn(24,device='cuda')*.1
  pre=torch.rand((R,4),device='cuda');norm=torch.randn(D,device='cuda').bfloat16()
  rows=[];times={}
  for name in ('original','split'):
   out=torch.empty((R,D),dtype=torch.bfloat16,device='cuda');pre_out=torch.empty_like(pre)
   post=torch.empty_like(pre);comb=torch.empty((R,4,4),device='cuda');part=torch.empty(R*K.HC_BLOCKS*32,device='cuda')
   side=torch.cuda.Stream()
   def run():
    args=(h,fn,scale,base,pre,norm,1e-6,1e-6,20,out,pre_out,post,comb,part)
    if name=='original':K.hc_pre(*args)
    else:
     after_post(*args,sink=side)
     torch.cuda.current_stream().wait_stream(side)
   run();torch.cuda.synchronize()
   values=[out,pre_out,post,comb,part.view(R,K.HC_BLOCKS,32)[:,:,:25]]
   expected=[x.clone() for x in values];graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):run()
   for _ in range(10):graph.replay()
   torch.cuda.synchronize()
   assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(values,expected)),(R,name,'graph')
   ms=[]
   for _ in range(12):
    b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    b.record();graph.replay();e.record();torch.cuda.synchronize();ms.append(b.elapsed_time(e))
   rows.append(expected);times[name]=statistics.median(ms)
  assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(*rows)),(R,'state')
  results.append({'rows':R,'all_state_bit_exact':True,'median_ms':times})
 Path(a.out).write_text(json.dumps(results,indent=2));print(json.dumps(results))
