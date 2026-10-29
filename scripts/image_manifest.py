"""Record installed versions and runtime source hashes inside the built image."""
import hashlib,importlib.metadata,json,re,sys
from pathlib import Path
from urllib.parse import unquote,urlsplit
root=Path('/opt/tensorfold');project=Path('/opt/spark1017');sources={}
for name,folder in [('tensorfold',root/'src'),('spark1017',project/'src'),('spark1017-native',project/'native')]:
    for p in sorted(folder.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:sources[name+'/'+str(p.relative_to(folder))]=hashlib.sha256(p.read_bytes()).hexdigest()
versions={p.metadata['Name']:p.version for p in importlib.metadata.distributions()}
locks=json.loads((project/'docker/package-hashes.json').read_text());downloads=[]
report=json.loads(Path('/opt/spark1017-pip-report.json').read_text())
for item in report['install']:
    info=item['download_info'];url=info['url'];meta=item['metadata']
    if info.get('dir_info',{}).get('editable'):
        assert url in ('file:///opt/tensorfold','file:///opt/spark1017'),url
        continue
    name=re.sub(r'[-_.]+','-',meta['name']).lower();lock=locks[name]
    filename=unquote(urlsplit(url).path).rsplit('/',1)[-1]
    digest=info['archive_info']['hashes']['sha256']
    assert meta['version']==lock['version'] and lock['files'].get(filename)==digest,'Unverified package: '+name
    downloads.append({'name':name,'version':meta['version'],'filename':filename,'sha256':digest})
Path(sys.argv[1]).write_text(json.dumps({'versions':dict(sorted(versions.items())),'source_hashes':sources,
    'downloaded_packages':downloads,'official_pypi_hashes_verified':True},indent=2)+'\n')
