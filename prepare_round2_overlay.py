"""Apply opt-in round-two hooks to the validated v1 overlay, in a new directory."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--v1-overlay',type=Path,required=True)
p.add_argument('--cuda-source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
expected=json.loads((root/'compat/round2-base.json').read_text())
for name,digest in expected.items():
    if hashlib.sha256((a.v1_overlay/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('Unsupported v1 overlay: '+name)
if a.output.exists():raise FileExistsError(a.output)
shutil.copytree(a.v1_overlay,a.output,ignore=shutil.ignore_patterns('__pycache__'))
extra=json.loads((root/'compat/round2-extra.json').read_text())
for name,digest in extra.items():
    source=a.cuda_source/name
    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
        raise ValueError('Unsupported decode source: '+name)
    shutil.copy2(source,a.output/name)
subprocess.run(['patch','-p1','-i',str((root/'compat/round2-engine.patch').resolve())],cwd=a.output,check=True)
manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.iterdir())
          if f.is_file() and f.name!='manifest.json'}
(a.output/'manifest.json').write_text(json.dumps({'v1_sha256':expected,'extra_sha256':extra,'round2_sha256':manifest},indent=2)+'\n')
print('Round-two overlay created. All experimental features default off.')
