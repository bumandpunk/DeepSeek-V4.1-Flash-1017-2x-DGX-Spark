"""Create a local source release with evidence and checksums; does not publish."""
import argparse,hashlib,json,subprocess,sys,tarfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','__pycache__','.venv','dist','build','private','runtime','state'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'dist');a=p.parse_args()
    subprocess.run([sys.executable,str(ROOT/'scripts/check_release.py')],check=True)
    proof=ROOT/'results/release-v0.2.0-alpha.1/summary.json'
    if not proof.exists() or not json.loads(proof.read_text()).get('release_gate_passed'):
        raise SystemExit('Clean deployment/recovery release gate is missing or has not passed')
    a.output.mkdir(parents=True,exist_ok=True)
    prefix='Spark1017-v0.2.0-alpha.1';archive=a.output/(prefix+'.zip')
    files=[]
    for f in sorted(ROOT.rglob('*')):
        rel=f.relative_to(ROOT)
        if not f.is_file() or any(x in EXCLUDED for x in rel.parts):continue
        if f.name.endswith(('.pyc','.key','.local.json')) or '.private.' in f.name or f.name.startswith('.env') or f.name=='.DS_Store':continue
        if f.is_relative_to(a.output.resolve()):continue
        if f.is_symlink():raise SystemExit('Unexpected symlink: '+str(rel))
        files.append(f)
    manifest={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for f in files:z.write(f,prefix+'/'+f.relative_to(ROOT).as_posix())
        z.writestr(prefix+'/RELEASE-FILES.sha256.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (a.output/'SHA256SUMS').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps({'archive':str(archive),'files':len(files),'sha256':digest}))

if __name__=='__main__':main()
