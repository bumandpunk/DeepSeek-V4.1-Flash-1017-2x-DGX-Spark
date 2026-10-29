"""Portable I/O adaptation of the frozen validation-set pilot; generated code runs only in isolated Docker containers."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
BASE = KEY_FILE = H5_FILE = None
DOCKER = ['docker']
DATA = ROOT / 'problems_dev.jsonl'
TEMPLATE = (ROOT / 'multistep_template.txt').read_text()
CONFIG = {
    'dataset': 'SciCode1/SciCode', 'revision': '4510f6a6aa27c43fad7b43da2c59602a86e88480',
    'split': 'validation', 'background': True, 'repeats': 1,
    'max_tokens': 16384, 'temperature': 1.0, 'top_p': 1.0,
    'thinking': True, 'reasoning_effort': 'high', 'execution_timeout_seconds': 300,
    'generation_timeout_seconds': 1800, 'api_failure_retries': 2,
    'model': 'DeepSeek-V4.1-Flash-1017', 'dataset_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
    'purpose': 'Development pilot, not an Artificial Analysis leaderboard score',
}

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def code_from(content):
    blocks = re.findall(r'```python\s*\n(.*?)\n```', content or '', re.S)
    return '\n\n'.join(blocks)

def prompt_for(problem, index, previous):
    steps = problem['sub_steps']
    old = []
    for j in range(index):
        s = steps[j]
        old += [s['step_description_prompt']+'\n'+s.get('step_background',''), previous[j], '------']
    s = steps[index]
    next_step = s['step_description_prompt']+'\n'+s.get('step_background','')+'\n\n'+s['function_header']+'\n\n'+s['return_line']
    return TEMPLATE.format(problem_steps_str='\n\n'.join(old[:-1]), next_step_str=next_step,
                           dependencies=problem['required_dependencies'])

def generate(problem, out):
    previous = []
    key = KEY_FILE.read_text().strip().split(':',1)[-1].strip()
    for index, step in enumerate(problem['sub_steps']):
        sid = step['step_number']
        directory = out / sid
        directory.mkdir(parents=True, exist_ok=True)
        raw_path = directory / 'response.json'
        prompt = prompt_for(problem, index, previous)
        (directory / 'prompt.txt').write_text(prompt)
        seed = int(hashlib.sha256(('round1-20261027:'+sid).encode()).hexdigest()[:8],16) % (2**31)
        body = {'model':CONFIG['model'], 'messages':[{'role':'user','content':prompt}],
                'temperature':CONFIG['temperature'], 'top_p':CONFIG['top_p'],
                'max_tokens':CONFIG['max_tokens'], 'seed':seed,
                'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'high'}}
        dump(directory / 'request.json', body)
        if raw_path.exists():
            result = json.loads(raw_path.read_text())
        else:
            error = None
            for retry in range(3):
                started = time.monotonic()
                try:
                    req = urllib.request.Request(BASE.rstrip('/') + '/v1/chat/completions',
                        json.dumps(body).encode(), {'Content-Type':'application/json','Authorization':'Bearer '+key})
                    with urllib.request.urlopen(req,timeout=1800) as response:
                        result=json.load(response)
                    result['_wall_seconds']=time.monotonic()-started
                    result['_api_retries']=retry
                    dump(raw_path,result)
                    error=None
                    break
                except (OSError,urllib.error.URLError,ValueError) as exc:
                    error=repr(exc)
                    time.sleep(2)
            if error:
                dump(directory/'api_error.json',{'error':error})
                raise RuntimeError(sid+': '+error)
        choice=result['choices'][0]
        code=code_from(choice['message'].get('content'))
        (directory/'solution.py').write_text(problem['required_dependencies']+'\n'+'\n'.join(previous)+'\n'+code)
        (directory/'step_code.py').write_text(code)
        previous.append(code)
        print(json.dumps({'event':'generated','step':sid,'finish_reason':choice.get('finish_reason'),
                          'seconds':round(result.get('_wall_seconds',0),2),'usage':result.get('usage'),
                          'code_extracted':bool(code)}),flush=True)

def grade(problem, out, image, gold=False):
    previous=[]
    for index,step in enumerate(problem['sub_steps']):
        sid=step['step_number']; directory=out/sid; directory.mkdir(parents=True,exist_ok=True)
        if gold:
            code=step['ground_truth_code']; solution=problem['required_dependencies']+'\n'+'\n'.join(previous)+'\n'+code
            previous.append(code)
            (directory/'solution.py').write_text(solution)
        elif not (directory/'solution.py').exists():
            continue
        tests='\nfrom scicode_helpers import process_hdf5_to_tuple\n'
        tests+=f'targets=process_hdf5_to_tuple({sid!r},{len(step["test_cases"])},"/data/test_data.h5")\n'
        for j,case in enumerate(step['test_cases']):
            tests+=f'\ntarget=targets[{j}]\n'+case+'\n'
        (directory/'run.py').write_text((directory/'solution.py').read_text()+tests)
        (directory/'scicode_helpers.py').write_bytes((ROOT/'scicode_helpers.py').read_bytes())
        for f in directory.iterdir():
            if f.is_file(): f.chmod(0o644)
        directory.chmod(0o755)
        name='spark1017-scicode-'+('gold-' if gold else 'model-')+sid.replace('.','-')
        args=DOCKER+['run','--rm','--name',name,'--network','none','--read-only',
              '--user','65534:65534','--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','128',
              '--cpus','4','--memory','8g','--memory-swap','8g','--tmpfs','/tmp:rw,nosuid,size=512m',
              '-v',f'{directory}:/work:ro','-v',f'{H5_FILE}:/data/test_data.h5:ro',
              image,'python','-B','/work/run.py']
        started=time.monotonic()
        try:
            p=subprocess.run(args,capture_output=True,text=True,timeout=300)
            status='passed' if p.returncode==0 else ('infrastructure_error' if p.returncode in (125,126,127) else 'failed')
            result={'step':sid,'status':status,'returncode':p.returncode,'seconds':time.monotonic()-started,
                    'stdout':p.stdout[-6000:],'stderr':p.stderr[-6000:]}
        except subprocess.TimeoutExpired:
            subprocess.run(DOCKER+['rm','-f',name],capture_output=True)
            result={'step':sid,'status':'timeout','seconds':time.monotonic()-started}
        dump(directory/'grade.json',result)
        print(json.dumps({'event':'graded','gold':gold,**{k:v for k,v in result.items() if k not in ('stdout','stderr')}}),flush=True)

def main():
    global BASE, KEY_FILE, H5_FILE, DOCKER
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['generate','grade','gold']);parser.add_argument('--label',default='baseline-original');parser.add_argument('--parents',type=int,default=15)
    parser.add_argument('--base'); parser.add_argument('--key-file',type=pathlib.Path)
    parser.add_argument('--h5-file',type=pathlib.Path); parser.add_argument('--executor-image')
    parser.add_argument('--sudo-docker',action='store_true')
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args()
    DOCKER=['sudo','-n','docker'] if args.sudo_docker else ['docker']
    BASE, KEY_FILE = args.base, args.key_file
    H5_FILE = args.h5_file.resolve() if args.h5_file else None
    if args.mode=='generate' and (not BASE or not KEY_FILE):
        parser.error('generate requires --base and --key-file')
    if args.mode!='generate':
        if not H5_FILE or not args.executor_image:
            parser.error('grade/gold require --h5-file and --executor-image')
        with H5_FILE.open('rb') as handle:
            digest=hashlib.file_digest(handle,'sha256').hexdigest()
        if digest!='48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890':
            parser.error('numeric test data hash differs from the frozen protocol')
    if CONFIG['dataset_sha256']!='193968aff23b7ed931c8f6d196b611e2f6694adb49e6cc085444a60ad2fc4f7b':
        parser.error('development dataset hash differs from the frozen protocol')
    problems=[json.loads(x) for x in DATA.read_text().splitlines()][:args.parents]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    config={**CONFIG,'problem_ids':[p['problem_id'] for p in problems],
            'subproblems':sum(len(p['sub_steps']) for p in problems),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    lock=out/'config.json'
    if lock.exists():
        assert json.loads(lock.read_text())==config, 'Frozen run configuration changed; use a new label.'
    else: dump(lock,config)
    image=None
    if args.mode!='generate':
        image=subprocess.check_output(DOCKER+['image','inspect',args.executor_image,'--format','{{.Id}}'],text=True).strip()
        dump(out/'executor.json',{'image_id':image})
        assert H5_FILE.is_file()
    for p in problems:
        generate(p,out) if args.mode=='generate' else grade(p,out,image,args.mode=='gold')
    print(json.dumps({'event':'complete','mode':args.mode,'label':args.label,'subproblems':config['subproblems']}),flush=True)

if __name__=='__main__': main()
