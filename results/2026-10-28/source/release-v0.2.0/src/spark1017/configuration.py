"""Reject inconsistent features or source across tensor-parallel ranks at boot."""
import hashlib
import json
import os
from pathlib import Path


def fingerprint(environ=None, source_dir=None):
    env = os.environ if environ is None else environ
    defaults = {'SPARK1017_STORAGE':'0', 'SPARK1017_RESIDENT_SCALES':'1',
                'SPARK1017_STORAGE_GIB':'4', 'SPARK1017_RESERVE_GIB':'12',
                'SPARK1017_HOT_MIB':'512', 'SPARK1017_COALESCE':'1',
                'SPARK1017_ADAPTIVE_DEPTH':'0', 'SPARK1017_LOOKUP':'0',
                'SPARK1017_CONTROL_FUSION':'0','SPARK1017_SEGMENTED_RDMA':'0',
                'SPARK1017_RDMA_POST':'0',
                'SPARK1017_PREFILL_SIDE':'0',
                'SPARK1017_OCCUPANCY':'0','SPARK1017_PERSISTENT':'0',
                'SPARK1017_CHAIN':'0','SPARK1017_CHAIN_DECODE':'0',
                'SPARK1017_DRAFT_HELPER':'','SPARK1017_DRAFT_KEY':'',
                'TF_DS_ROUND_MS':'', 'TF_DS_DRAFT_MS':'', 'TF_DS_DEPTH_POLICY':''}
    values = {key: env.get(key, default) for key, default in defaults.items()}
    cost_file = env.get('SPARK1017_COSTS')
    values['cost_sha256'] = hashlib.sha256(Path(cost_file).read_bytes()).hexdigest() if cost_file else None
    directory = Path(source_dir) if source_dir else Path(__file__).parent
    values['sources'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(directory.glob('*.py'))}
    native=directory.parents[1]/'native'
    values['native_sources']={str(p.relative_to(native)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(native.rglob('*'))
                             if p.suffix in ('.cu','.cuh','.cpp','.hpp','.inc') and p.is_file()}
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def verify_peers(engine, digest=None, *, engine_source=None):
    try:
        value = digest or fingerprint()
        if engine_source is not None:
            root=Path(engine_source)
            overlay={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                     for name in ('engine.py','model.py','multi.py','rounds.py')}
            value=hashlib.sha256(json.dumps([value,overlay],sort_keys=True).encode()).hexdigest()
        mine = {'digest': value}
    except Exception as error:
        mine = {'error': type(error).__name__ + ': ' + str(error)}
    if engine.world > 1:
        store = engine.nccl.store
        prefix = 'spark1017/features/v1/'
        store.set(prefix + str(engine.rank), json.dumps(mine))
        peers = [json.loads(store.get(prefix + str(rank))) for rank in range(engine.world)]
    else:
        peers = [mine]
    if any('error' in p for p in peers):
        raise RuntimeError('Spark1017 configuration preparation failed: ' + repr(peers))
    if len({p['digest'] for p in peers}) != 1:
        raise RuntimeError('Spark1017 source, features or hardware cost file differ across ranks')
    engine._spark1017_verified = True
    if engine.rank == 0:
        print('[spark1017] peer configuration verified ' + mine['digest'], flush=True)
