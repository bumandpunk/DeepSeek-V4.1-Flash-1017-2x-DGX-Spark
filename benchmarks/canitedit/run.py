"""Frozen CanItEdit subset, HTTP generation and isolated executable tests."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import resource

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import kit_bench as kit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label', required=True)
parser.add_argument('--mode', choices=['model', 'gold', 'before'], default='model')
parser.add_argument('--base')
parser.add_argument('--key-file')
parser.add_argument('--executor-image', required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--reps', type=int, default=2)
parser.add_argument('--sudo-docker', action='store_true')
args = parser.parse_args()
if args.mode == 'model' and (not args.base or not args.key_file):
    parser.error('model mode requires --base and --key-file')
if not re.fullmatch(r'[a-zA-Z0-9-]+', args.label):
    parser.error('label must be alphanumeric or hyphens')
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
output.chmod(0o755)
docker = (['sudo', '-n'] if args.sudo_docker else []) + ['docker']
image = subprocess.check_output(docker + ['image', 'inspect', args.executor_image, '--format', '{{.Id}}'], text=True).strip()
cases = json.loads((ROOT / 'selected.json').read_text())
manifest = json.loads((ROOT / 'manifest.json').read_text())
manifest.update(label=args.label, mode=args.mode, repetitions=args.reps if args.mode=='model' else 1,
                executor_image_id=image, harness_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                scoring_seed=1017, scoring_timeout_seconds=60)
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
client = kit.Client(args.base, 'DeepSeek-V4.1-Flash-1017', kit.load_key(args.key_file)) if args.mode=='model' else None
summary = []
for rep in range(manifest['repetitions']):
    for case in cases:
        sid = str(case['id'])
        directory = output / (sid + '-' + str(rep))
        directory.mkdir(mode=0o755)
        if args.mode == 'model':
            # The upstream zero-shot chat prompt. Gold and tests are not sent.
            messages = [
                {'role':'system','content':'You are PythonEditGPT. You will be provided the original code snippet and an instruction that specifies the changes you need to make. You will produce the changed code, based on the original code and the instruction given. Only produce the code, do not include any additional prose.'},
                {'role':'user','content':f"## Code Before\n```py\n{case['before']}\n```\n\n## Instruction\n{case['instruction_descriptive']}\n\n## Code After"},
            ]
            request = {'messages':messages,'temperature':0,'top_p':1,'max_tokens':8192,'seed':1017+case['id'],
                       'enable_thinking':False,'reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':False}}
            response = client.stream(request, '/v1/chat/completions')
            (directory / 'record.json').write_text(json.dumps({'request':request,'response':response},ensure_ascii=False,indent=2))
            code = response['text'].strip()
            fence = re.search(r'```(?:python|py)?\s*\n(.*?)\n```', code, re.S)
            if fence:
                code = fence.group(1)
        else:
            request = None
            response = None
            code = case['after' if args.mode=='gold' else 'before']
        (directory / 'solution.py').write_text(code)
        sentinel = 'SPARK1017_TESTS_PASSED_SENTINEL'
        (directory / 'program.py').write_text(code+'\n\n'+case['tests']+'\nprint('+repr(sentinel)+')\n')
        for file in directory.iterdir():
            file.chmod(0o644)
        name = 'spark-canitedit-'+args.label+'-'+sid+'-'+str(rep)
        seed_and_run = "import random, numpy, torch, runpy; random.seed(1017); numpy.random.seed(1017); torch.manual_seed(1017); runpy.run_path('/work/program.py',run_name='__main__')"
        command = docker + ['run','--rm','--name',name,'--log-driver','none','--network','none','--read-only','--cap-drop','ALL',
                            '--security-opt','no-new-privileges','--user','65534:65534','--pids-limit','128',
                            '--cpus','2','--memory','2g','--memory-swap','2g','--tmpfs','/tmp:rw,nosuid,size=128m',
                            '-e','PYTHONHASHSEED=0','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1',
                            '-v',str(directory)+':/work:ro','--entrypoint','python3',image,'-I','-B','-c',seed_and_run]
        try:
            def limit_output():
                resource.setrlimit(resource.RLIMIT_FSIZE,(1<<20,1<<20))
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                result = subprocess.run(command,stdout=stdout,stderr=stderr,timeout=60,preexec_fn=limit_output)
                stdout.seek(0); stderr.seek(0)
                out_text=stdout.read(1<<20).decode(errors='replace')
                err_text=stderr.read(1<<20).decode(errors='replace')
            grade = {'passed':result.returncode==0 and sentinel in out_text.splitlines(),
                     'returncode':result.returncode,'stdout':out_text[-3000:],'stderr':err_text[-3000:]}
        except subprocess.TimeoutExpired:
            grade = {'passed':False,'timeout':True}
        finally:
            subprocess.run(docker+['rm','-f',name],capture_output=True,timeout=15)
        (directory / 'grade.json').write_text(json.dumps(grade,ensure_ascii=False,indent=2))
        row = {'id':case['id'],'name':case['name'],'rep':rep,'passed':grade['passed'],
               'solution_sha256':hashlib.sha256(code.encode()).hexdigest()}
        if response is not None:
            row.update(request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
                       text_sha256=hashlib.sha256(response['text'].encode()).hexdigest(),tokens=response['tokens'],
                       tps=response['decode_tps'],ttft_s=response['ttft_s'],wall_s=response['end']-response['start'],
                       possibly_truncated=response['tokens']==8192)
        summary.append(row)
        (output / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        print(json.dumps(row),flush=True)
(output/'complete.json').write_text(json.dumps({'cases':len(cases),'generations':len(summary),'passed':sum(x['passed'] for x in summary)}))
print('COMPLETE '+args.label,flush=True)
