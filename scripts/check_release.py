"""Check distributable files and source syntax without GPU access or model execution."""
import ast,json,re,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from cluster import validate

def main():
    required=['README.md','README.zh-CN.md','LICENSE','NOTICE','CONTRIBUTING.md',
        'docs/operations.md','docs/releases/v0.2.0-alpha.1.md','docker/base-image.json',
        'config/cluster.example.json','.github/workflows/cpu-tests.yml']
    for name in required:assert (ROOT/name).is_file(),name
    meta=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']
    assert meta['version']=='0.2.0a1'
    validate(json.loads((ROOT/'config/cluster.example.json').read_text()))
    base=json.loads((ROOT/'docker/base-image.json').read_text())
    assert base['manifest_digest'] in (ROOT/'Dockerfile').read_text()
    files=[]
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if any(v in rel.parts for v in ('.git','__pycache__','.venv','dist','build')):continue
        if not p.is_file():continue
        assert not p.name.endswith(('.key','.local.json')) and '.private.' not in p.name,str(rel)
        assert not p.name.startswith('.env'),str(rel)
        if 'results' not in rel.parts and p.suffix=='.py':
            ast.parse(p.read_bytes(),filename=str(rel));files.append(str(rel))
    print(json.dumps({'release':'v0.2.0-alpha.1','python_sources':len(files),'required_files':len(required),'checks':'passed'}))

if __name__=='__main__':main()
