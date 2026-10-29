"""Exact-state check for the side-stream schedule, including CUDA Graph replay."""
import argparse,json
from pathlib import Path
import torch
from tensorfold.families.deepseek_v41.cuda import kernels as K
from spark1017.postmix import after_post

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
D=5120;cases=[]
with torch.inference_mode():
 for R in (1,2,3,6,12,17,34):
  torch.manual_seed(1017+R)
  h=torch.randn((R,4,D),device='cuda').bfloat16()
  g=torch.randn((2,R,D),device='cuda')
  fn=torch.randn((24,4*D),device='cuda')*.01
  scale=torch.tensor([.1,.1,.1],device='cuda');base=torch.randn(24,device='cuda')*.1
  pre=torch.rand((R,4),device='cuda');norm=torch.randn(D,device='cuda').bfloat16()
  old_post=torch.rand((R,4),device='cuda');old_comb=torch.rand((R,4,4),device='cuda')
  outputs=[]
  for candidate in (False,True):
   hi=h.clone();ho=torch.empty_like(h);out=torch.empty((R,D),device='cuda',dtype=torch.bfloat16)
   post=old_post.clone();comb=old_comb.clone();pre_out=torch.empty_like(pre)
   part=torch.empty(R*K.HC_BLOCKS*32,device='cuda');sink=torch.cuda.Stream()
   def run():
    hi.copy_(h);post.copy_(old_post);comb.copy_(old_comb)
    if candidate:
     K.hc_post(g,hi,post,comb,hi)
     after_post(hi,fn,scale,base,pre,norm,1e-6,1e-6,20,out,pre_out,post,comb,part,sink=sink)
    else:
     K.hc_pre2(hi,fn,scale,base,pre,norm,1e-6,1e-6,20,out,pre_out,post,comb,part,gathered=g,h_out=ho,sink=sink)
    torch.cuda.current_stream().wait_stream(sink)
   run();torch.cuda.synchronize()
   values=[hi if candidate else ho,out,pre_out,post,comb,part.view(R,K.HC_BLOCKS,32)[:,:,:25]]
   eager=[x.clone() for x in values]
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):run()
   for _ in range(10):graph.replay()
   torch.cuda.synchronize()
   assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(values,eager)),(R,candidate,'graph')
   outputs.append(eager)
  assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(*outputs)),(R,'state')
  cases.append({'rows':R,'all_state_bit_exact':True,'graph_replays':10})
 Path(a.out).write_text(json.dumps(cases,indent=2));print(json.dumps(cases))
