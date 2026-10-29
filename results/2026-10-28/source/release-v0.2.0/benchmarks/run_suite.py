"""Run the pinned public benchmark unchanged, additionally preserving every full response."""
import argparse,datetime,hashlib,json,pathlib,types
import kit_bench as kit

p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--quick',action='store_true');p.add_argument('--profile-only',action='store_true');p.add_argument('--prefill-only',action='store_true');p.add_argument('--base',required=True);p.add_argument('--key-file');p.add_argument('--model',default='DeepSeek-V4.1-Flash-1017');p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent;out=a.output
out.mkdir(parents=True,exist_ok=False)
spec={'label':a.label,'quick':a.quick,'profile_only':a.profile_only,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'kit_sha256':hashlib.sha256((root/'kit_bench.py').read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'decode_set':'b','max_tokens':384,
      'repetitions':3 if a.quick else 9,'context_capacity':524288,'engine_slots':2,'thinking':False,'temperature':0,
      'driver':'standard-library HTTP streaming client','base':a.base,
      'quality_boundary':'Speed prompts are not training data. Quality evaluated separately.'}
(out/'manifest.json').write_text(json.dumps(spec,indent=2))
class RecordingClient(kit.Client):
 def stream(self,body,path):
  result=super().stream(body,path)
  request=json.dumps({'path':path,'body':body},sort_keys=True)
  record={'request_sha256':hashlib.sha256(request.encode()).hexdigest(),'request':{'path':path,'body':body},'response':result}
  with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
  if result.get('tokens') is None or result.get('ttft_s') is None:
   (out/'request_failure.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
   raise RuntimeError('Server returned no completed generation; raw request and partial response preserved. Check engine logs.')
  return result
c=RecordingClient(spec['base'],a.model,kit.load_key(a.key_file))
print(json.dumps({'event':'ready',**kit.cmd_ready(c,types.SimpleNamespace(timeout=600))}),flush=True)
summary={}
# The public driver salts random prefixes with wall time. Freeze only its
# filler seed here so engine comparisons receive byte-identical token IDs.
# The pinned kit source and every streaming timing operation stay unchanged.
_filler = kit.filler_ids
_prefix_occurrences = {}
def fixed_filler(client, length, seed):
 occurrence = _prefix_occurrences.get(length, 0)
 _prefix_occurrences[length] = occurrence + 1
 return _filler(client, length, seed=20261027 + (length << 4) + occurrence)
kit.filler_ids = fixed_filler
spec['prefix_seed_scheme'] = '20261027 + (length << 4) + occurrence; wrapper v2'
(out/'manifest.json').write_text(json.dumps(spec,indent=2))
def run(name,args):
 print(json.dumps({'event':'begin','section':name}),flush=True)
 result=getattr(kit,'cmd_'+name)(c,types.SimpleNamespace(**args));summary[name]=result
 (out/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps({'event':'end','section':name}),flush=True)
if not a.prefill_only:run('decode',dict(set='b',tokens=384,reps=3 if a.quick else 9))
if not a.profile_only:
 if not a.prefill_only:run('concurrent',dict(set='b',streams='2,4',tokens=384,reps=2 if a.quick else 3))
 run('prefill',dict(lengths='8192,32768' if a.quick else '8192,32768,131072',reps=1 if a.quick else 3))
 if not a.quick:run('depth',dict(lengths='131072',tokens=512))
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({'event':'complete','label':a.label}),flush=True)
