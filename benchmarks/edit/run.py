"""Source-preserving code edit measurements with independent executable checks."""
import argparse,hashlib,json,pathlib,re,subprocess,sys
r=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(r.parent));import kit_bench as kit
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--reps',type=int,default=2);p.add_argument('--base',required=True);p.add_argument('--key-file');p.add_argument('--executor-image',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
o=a.output.resolve();o.mkdir(parents=True,exist_ok=False)
c=kit.Client(a.base,'DeepSeek-V4.1-Flash-1017',kit.load_key(a.key_file))
tasks={
 'bisect':'Add a public function bisect_bounds(a, x, lo=0, hi=None, *, key=None) returning (left_index, right_index) by calling the existing bisect_left and bisect_right with the same arguments. Preserve every existing function and all existing behavior.',
 'graphlib':'Add TopologicalSorter.__len__(self), returning the total number of registered nodes, including predecessor-only nodes, independent of whether nodes are already marked done. Preserve every existing method and all existing behavior.',
 'textwrap':'Add a public function wrap_lines(text, width=70, **kwargs). For each line from text.splitlines(), extend the result with wrap(line, width=width, **kwargs), or with one empty string when that line yields no wrapped lines. Return the flattened list. Add wrap_lines to __all__. Preserve every existing function and all existing behavior.'}
image=subprocess.check_output(['docker','image','inspect',a.executor_image,'--format','{{.Id}}'],text=True).strip()
manifest={'label':a.label,'source':'python/cpython v3.12.9','tasks':tasks,'repetitions':a.reps,'temperature':0,'max_tokens':16384,'thinking':False,'image':image,'purpose':'Engineering fixtures for source-preserving edits, not a general coding benchmark or official score','sources':{n:hashlib.sha256((r/'sources'/(n+'.py')).read_bytes()).hexdigest() for n in tasks}}
(o/'manifest.json').write_text(json.dumps(manifest,indent=2))
summary=[]
for rep in range(a.reps):
 for name,instruction in tasks.items():
  source=(r/'sources'/(name+'.py')).read_text()
  prompt='Edit the following complete Python module. '+instruction+' Return the entire updated module as runnable Python code, without Markdown fences or explanations. Retain existing comments and docstrings.\n\nSOURCE:\n'+source
  body={'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':16384,'seed':1017,'enable_thinking':False,'reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':False}}
  result=c.stream(body,'/v1/chat/completions'); record={'name':name,'rep':rep,'request':body,'response':result}
  stem=name+'-'+str(rep);(o/(stem+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2))
  code=result['text'].strip()
  if code.startswith('```') and code.endswith('```'):code=code.split('\n',1)[1].rsplit('```',1)[0].strip()
  (o/(stem+'.py')).write_text(code);(o/(stem+'.py')).chmod(0o644);o.chmod(0o755)
  container='spark-edit-'+stem
  args=['docker','run','--rm','--name',container,'--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534','--pids-limit','64','--cpus','2','--memory','512m','--memory-swap','512m','--tmpfs','/tmp:rw,nosuid,size=64m','-v',str(o)+':/work:ro','-v',str(r/'check.py')+':/check.py:ro',image,'python3','-I','-B','/check.py',name,'/work/'+stem+'.py']
  try:
   check=subprocess.run(args,capture_output=True,text=True,timeout=30)
   grade=json.loads(check.stdout) if check.returncode==0 else {'passed':0,'total':None,'error':check.stderr[-3000:],'returncode':check.returncode}
  except subprocess.TimeoutExpired:
   subprocess.run(['docker','rm','-f',container],capture_output=True);grade={'passed':0,'total':None,'error':'timeout'}
  row={'name':name,'rep':rep,'tokens':result['tokens'],'tps':result['decode_tps'],'ttft_s':result['ttft_s'],'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),'text_sha256':hashlib.sha256(result['text'].encode()).hexdigest(),'grade':grade}
  summary.append(row);(o/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(row),flush=True)
  if not result['tokens']:raise RuntimeError('incomplete generation')
print('COMPLETE '+a.label,flush=True)
