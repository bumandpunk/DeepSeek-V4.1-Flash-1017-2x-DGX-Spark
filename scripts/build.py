"""Build from pinned public source and an exact base, then distribute one image."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from urllib.parse import urlsplit
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cluster import validate

ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'.git','.github','docs','results','__pycache__','.venv','private','runtime','dist','build'}

def source_digest(root):
    hashes={}
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if not p.is_file() or any(x in EXCLUDE for x in rel.parts):continue
        if len(rel.parts)==1 and (p.name.startswith('README') or p.name in ('CHANGELOG.md','CONTRIBUTING.md')):continue
        if p.name.endswith(('.pyc','.key','.local.json')) or p.name.startswith('.env') or '.private.' in p.name or p.name=='.DS_Store':continue
        hashes[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),hashes

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'config/cluster.local.json')
    parser.add_argument('--base-image',help='Offline local tag; must match the pinned official image ID')
    parser.add_argument('--package-index',default='https://pypi.org/simple',help='HTTPS PyPI mirror; downloaded packages must match the official pinned hashes')
    parser.add_argument('--no-cache',action='store_true');parser.add_argument('--skip-distribute',action='store_true');a=parser.parse_args()
    index=urlsplit(a.package_index)
    if index.scheme!='https' or not index.hostname or index.username or index.password or index.query or index.fragment:
        raise ValueError('Use an HTTPS package index without embedded credentials, query or fragment')
    config=validate(json.loads(a.config.read_text()));base=json.loads((ROOT/'docker/base-image.json').read_text())
    image=a.base_image or base['registry']+'@'+base['manifest_digest']
    existing=subprocess.run(['docker','image','inspect',image],capture_output=True,text=True)
    if existing.returncode:
        if a.base_image:raise RuntimeError('Offline base image not present')
        subprocess.run(['docker','pull','--platform','linux/arm64',image],check=True)
        existing=subprocess.run(['docker','image','inspect',image],capture_output=True,text=True,check=True)
    meta=json.loads(existing.stdout)[0]
    if meta['Id']!=base['image_id'] or meta['Architecture']!='arm64':raise RuntimeError('Base image does not match the pinned official image ID')
    digest,files=source_digest(ROOT)
    args=['docker','build','--platform','linux/arm64','--network','host','--build-arg','BASE_IMAGE='+image,
        '--build-arg','SOURCE_SHA256='+digest,'--build-arg','PIP_INDEX_URL='+a.package_index,'-t',config['image']]
    if a.no_cache:args+=['--no-cache']
    for name in ('HTTP_PROXY','HTTPS_PROXY','NO_PROXY'):
        value=os.environ.get(name) or os.environ.get(name.lower())
        if value:args+=['--build-arg',name+'='+value]
    args+=[str(ROOT)];subprocess.run(args,check=True)
    built=json.loads(subprocess.check_output(['docker','image','inspect',config['image']],text=True))[0]
    assert built['Config']['Labels']['org.spark1017.source-sha256']==digest
    state=Path(config['nodes'][0]['state']);state.mkdir(parents=True,exist_ok=True,mode=0o700)
    record={'image_id':built['Id'],'base_image_id':base['image_id'],'base_manifest_digest':base['manifest_digest'],
        'source_sha256':digest,'source_files':files,'source_scope':'Declared runtime/build inputs; documentation, CI workflows and result evidence are excluded.',
        'no_application_build_cache':a.no_cache,'package_index':a.package_index}
    if not a.skip_distribute:
        target=config['nodes'][1]['ssh']
        sender=subprocess.Popen(['docker','save',config['image']],stdout=subprocess.PIPE)
        receiver=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',target,'docker load'],stdin=sender.stdout)
        sender.stdout.close();receive_code=receiver.wait();send_code=sender.wait()
        if receive_code or send_code:raise RuntimeError('Image distribution failed; preserve the local image and retry')
        worker=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes',target,
            'docker image inspect '+__import__('shlex').quote(config['image'])],text=True))[0]
        if worker['Id']!=built['Id']:raise RuntimeError('Distributed image ID differs')
        record['worker_image_id']=worker['Id']
    (state/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='source_files'},indent=2))

if __name__=='__main__':main()
