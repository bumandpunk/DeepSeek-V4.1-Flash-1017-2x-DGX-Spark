"""Interleaved real-weight prompt microbenchmark, output equality before timing."""
import argparse, json, os, statistics, time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--layer',type=int,default=10);p.add_argument('--out',required=True);p.add_argument('--rows',default='128,512,2048');p.add_argument('--iters',type=int,default=16);a=p.parse_args()
from tensorfold.cuda.exl3 import experts as X
from tensorfold.families.deepseek_v41.config import Cfg
from tensorfold.families.deepseek_v41.cuda.weights import Shards,load_block
cfg=Cfg.read(a.model);lay=load_block(Shards(a.model),cfg,f'layers.{a.layer}',a.layer,0,2,cfg.n_routed);ex=lay.experts
E,D,slots=ex.count-1,ex.dims,cfg.topk+1
print(json.dumps({'E':E,'D':D,'I':ex.width,'cb':ex.cb,'gu':ex.k2_gu,'down':ex.k2_d,'cfg':X.default_config(D,ex.width,True)}),flush=True)
l2=torch.cuda.get_device_properties(0).L2_cache_size
flush=torch.empty(max(4*l2,128<<20)//4,dtype=torch.int32,device='cuda');sink=torch.empty((),dtype=torch.int64,device='cuda')
result={'layer':a.layer,'gpu':torch.cuda.get_device_name(),'rows':[]}
for R in map(int,a.rows.split(',')):
 for mode in ['uniform','clustered','gate']:
  g=torch.Generator(device='cuda').manual_seed(281028+R+['uniform','clustered','gate'].index(mode))
  x=(torch.randn((R,D),generator=g,device='cuda')*.5).bfloat16()
  if mode=='gate':q=(torch.nn.functional.softplus(x.float()@lay.gate_w.float().t()).sqrt()+lay.gate_b.float()).topk(cfg.topk,dim=-1).indices
  else:
   q=torch.rand((R,E if mode=='uniform' else min(E,24)),device='cuda',generator=g).topk(cfg.topk,dim=-1).indices
  pick=torch.cat([q,torch.full((R,1),E,device='cuda')],1).int().contiguous()
  w=torch.rand((R,slots),generator=g,device='cuda');w[:,-1]=1
  ss={v:X.Scratch(ex,max(R,64),slots,prompt=R>=64) for v in ['uniform','chain']}
  outs={}
  def call(v):
   os.environ['SPARK1017_OCCUPANCY']='0'
   os.environ['SPARK1017_CHAIN']='1' if v=='chain' else '0'
   os.environ['SPARK1017_CHAIN_DECODE']='1' if v=='chain' else '0'
   return X.routed(x,pick,w,ex,ss[v],None,R,limit=cfg.swiglu_limit)
  for v in ss:
   for _ in range(3):out=call(v)
   outs[v]=out.clone()
  torch.cuda.synchronize()
  equal=torch.equal(outs['uniform'],outs['chain'])
  diff=(outs['uniform']-outs['chain']).abs()
  row={'R':R,'routing':mode,'bit_equal':equal,'max_abs_error':diff.max().item(),'different_values':int(torch.count_nonzero(diff))}
  assert equal,row
  times={v:[] for v in ss}
  for it in range(a.iters+2):
   for v in (list(ss) if it%2==0 else list(reversed(ss))):
    torch.sum(flush,0,dtype=torch.int64,out=sink)
    st,en=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    st.record();call(v);en.record();torch.cuda.synchronize()
    if it>=2:times[v].append(st.elapsed_time(en))
  row['ms']=times;row['median_ms']={v:statistics.median(t) for v,t in times.items()};row['speedup']=row['median_ms']['uniform']/row['median_ms']['chain']
  result['rows'].append(row);Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in row.items() if k!='ms'}),flush=True)
