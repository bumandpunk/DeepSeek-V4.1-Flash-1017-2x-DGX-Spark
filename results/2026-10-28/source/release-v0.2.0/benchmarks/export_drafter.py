"""Copy only original drafter/embedding/head tensor bytes to an isolated helper pack."""
import argparse,pathlib,struct,json,hashlib,shutil,time
p=argparse.ArgumentParser();p.add_argument("--model",type=pathlib.Path,required=True);p.add_argument("--output",type=pathlib.Path,required=True);a=p.parse_args()
root=a.model
out=a.output;out.mkdir(parents=True,exist_ok=False)
entries=[];header={};at=0
for f in sorted(root.glob('*.safetensors')):
 with f.open('rb') as h:
  n=struct.unpack('<Q',h.read(8))[0];meta=json.loads(h.read(n))
 for name,e in meta.items():
  if name=='__metadata__' or not (name.startswith(('mtp.','head.')) or name in ('embed.weight','norm.weight')):continue
  assert name not in header, 'duplicate tensor name'
  begin,end=e['data_offsets'];size=end-begin
  assert 0<=begin<=end and n+8+end<=f.stat().st_size, 'invalid tensor offsets'
  header[name]={**e,'data_offsets':[at,at+size]};entries.append((name,f,n+8+begin,size));at+=size
raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
print(json.dumps({'tensors':len(entries),'gib':at/2**30}),flush=True)
digest=hashlib.sha256()
with (out/'draft.safetensors.partial').open('wb') as dst:
 for b in [struct.pack('<Q',len(raw)),raw]:dst.write(b);digest.update(b)
 for name,f,offset,size in entries:
  with f.open('rb') as src:
   src.seek(offset)
   while size:
    b=src.read(min(size,16<<20));assert b;dst.write(b);digest.update(b);size-=len(b)
(out/'draft.safetensors.partial').rename(out/'draft.safetensors')
shutil.copy2(root/'config.json',out/'config.json')
(out/'manifest.json').write_text(json.dumps({'tensors':len(entries),'bytes':at,'sha256':digest.hexdigest()},indent=2))
print((out/'manifest.json').read_text(),flush=True)
