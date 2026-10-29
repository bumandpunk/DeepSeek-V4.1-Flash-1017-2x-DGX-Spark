"""Per-node operations. Executed locally or sent over authenticated SSH by cluster.py."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import tarfile
import time

LABEL='org.spark1017.deployment'
PROFILES={
    'baseline':{'SPARK1017_LOOKUP':'0','SPARK1017_RDMA_POST':'0'},
    'edit':{'SPARK1017_LOOKUP':'1','SPARK1017_RDMA_POST':'0'},
    'prefill-experimental':{'SPARK1017_LOOKUP':'0','SPARK1017_RDMA_POST':'all'},
}

def run(args,check=True,timeout=90):
    p=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
    if check and p.returncode:
        # Arguments contain paths only, never API key contents.
        raise RuntimeError(f'{args[0]} failed ({p.returncode}): '+p.stderr[-2000:])
    return p

def inspect(name):
    p=run(['docker','inspect',name],check=False)
    return json.loads(p.stdout)[0] if p.returncode==0 else None

def owned(c,deployment):
    return bool(c) and c['Config'].get('Labels',{}).get(LABEL)==deployment

def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])/2**20
    raise RuntimeError('MemAvailable unavailable')

def container_name(config,rank):return f"{config['deployment']}-r{rank}"

def prebuild_name(config,rank):return f"{config['deployment']}-prebuild-r{rank}"

def check_writable_paths(node):
    for key in ('cache','state'):
        q=Path(node[key]).resolve()
        if any(q==Path(node[k]).resolve() or q.is_relative_to(Path(node[k]).resolve()) or Path(node[k]).resolve().is_relative_to(q) for k in ('model','engram')):
            raise RuntimeError('Writable path overlaps immutable weights on this node')

def stop_prebuild(config,node):
    name=prebuild_name(config,node['rank']);c=inspect(name)
    if c is None:return False
    if not owned(c,config['deployment']):raise RuntimeError('Refusing to stop a foreign container: '+name)
    run(['docker','rm','-f',name])
    return True

def environment(config,node,profile,run_id):
    env={'MAX_JOBS':'1','TORCH_EXTENSIONS_DIR':'/cache/torch_extensions',
        'CUDA_CACHE_PATH':'/cache/nv','TRITON_CACHE_DIR':'/cache/triton',
        'TF_DS_ENGRAM':'/engram','TF_DS_RANK_CACHE':'/cache/rank',
        'TF_DS_SPILL_DIR':f'/cache/spill/{run_id}','TF_DS_SPILL_GIB':'16',
        'TF_DS_PREFILL_CHUNK':'2048','TF_DS_DEPTH_POLICY':'joint',
        'TF_DS_REPLAY':'1','TF_DS_REPLAY_FLOOR':'1','TF_DS_ENGRAM_DONTNEED':'1',
        'TF_DS_MEM_FLOOR_GIB':'4','TF_DS_MEM_FLOOR_INCLUDES_KERNEL':'1',
        'TF_DS_WORKSPACE_GIB':'1','TF_DS_HOST_MIN_GIB':str(config['guard_floor_gib']+1.0),
        'TF_DS_KEEP_MARKS':'4','TF_DS_KEEP_ENTRIES':'2',
        'NCCL_SOCKET_IFNAME':node['socket_interface'],
        'NCCL_IB_HCA':','.join(node['rdma_devices']),
        'TF_RDMA_DEVICES':','.join(node['rdma_devices']),
        'NCCL_IB_GID_INDEX':str(node['gid_index']),'TF_RDMA_GID_INDEX':str(node['gid_index']),
        'SPARK1017_STORAGE':'0','SPARK1017_ADAPTIVE_DEPTH':'0',
        'SPARK1017_PREFILL_SIDE':'0','SPARK1017_CONTROL_FUSION':'0',
        'SPARK1017_SEGMENTED_RDMA':'0','SPARK1017_OCCUPANCY':'0',
        'SPARK1017_PERSISTENT':'0','SPARK1017_CHAIN':'0','SPARK1017_CHAIN_DECODE':'0',
        'SPARK1017_DRAFT_HELPER':'','SPARK1017_PROFILE':'','SPARK1017_KERNEL_TRACE':''}
    env.update(PROFILES[profile]);return env

def docker_command(config,node,profile,run_id):
    args=['docker','run','-d','--name',container_name(config,node['rank']),
        '--label',LABEL+'='+config['deployment'],'--label','org.spark1017.run='+run_id,
        '--label','org.spark1017.profile='+profile,'--gpus','all','--network','host','--ipc','host',
        '--device','/dev/infiniband','--cap-add','IPC_LOCK','--ulimit','memlock=-1:-1',
        '--ulimit','stack=67108864:67108864','--log-driver','local','--log-opt','max-size=20m','--log-opt','max-file=3']
    for source,target,readonly in [(node['model'],'/model',True),(node['engram'],'/engram',True),
                                   (node['cache'],'/cache',False),(node['key_file'],'/run/secrets/api-key',True)]:
        args+=['--mount',f'type=bind,src={source},dst={target}'+(',readonly' if readonly else '')]
    for key,value in sorted(environment(config,node,profile,run_id).items()):args+=['-e',key+'='+value]
    args+=[config['image'],'serve','/model','--tp','2','--rank',str(node['rank']),
        '--master',config['nodes'][0]['fabric_ip'],'--host',config['api_host'],'--port',str(config['api_port']),
        '--name',config['model_name'],'--api-key-file','/run/secrets/api-key',
        '--context',str(config['context']),'--parallel',str(config['parallel']),
        '--mtp-drafts','5','--temperature','1.0','--top-p','1.0','--max-tokens','32768',
        '--no-thinking','--no-update-check']
    if config['vision']:args+=['--vision']
    return args

def doctor(config,node,idle=False):
    problems=[]
    if platform.machine()!='aarch64':problems.append('requires an aarch64 GB10 host')
    gpu=run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],check=False)
    if gpu.returncode or 'GB10' not in gpu.stdout:problems.append('GB10 GPU not detected')
    run(['docker','info'],timeout=20)
    paths={}
    for name in ('model','engram','key_file'):
        p=Path(node[name]);paths[name]=p.exists()
        if not p.exists():problems.append(f'missing {name}: {p}')
    model=Path(node['model']);index=model/'model.safetensors.index.json'
    if not (model/'config.json').is_file():problems.append('model config.json missing')
    if not (model/'tokenizer.json').is_file():problems.append('model tokenizer.json missing')
    if index.exists():
        for name in set(json.loads(index.read_text())['weight_map'].values()):
            q=model/name
            if not q.is_file() or q.stat().st_size<8:problems.append('missing/empty checkpoint shard: '+name)
            if q.is_symlink() and not q.resolve().is_relative_to(model.resolve()):problems.append('checkpoint symlink leaves the container mount: '+name)
    else:problems.append('model.safetensors.index.json missing')
    if Path(node['key_file']).is_file():
        if len(Path(node['key_file']).read_text().strip())<16:problems.append('API key must contain at least 16 characters')
        if Path(node['key_file']).stat().st_mode&0o077:problems.append('API key permissions must be 0600')
    for device in node['rdma_devices']:
        port=Path('/sys/class/infiniband')/device/'ports/1'
        if not (port/'state').exists() or 'ACTIVE' not in (port/'state').read_text():problems.append('inactive RDMA device: '+device)
        gid=port/'gids'/str(node['gid_index'])
        if not gid.exists() or gid.read_text().strip()=='0000:0000:0000:0000:0000:0000:0000:0000':problems.append('unusable GID: '+device)
    links=json.loads(run(['ip','-j','address','show','dev',node['socket_interface']]).stdout)
    if not any(a.get('local')==node['fabric_ip'] for x in links for a in x.get('addr_info',[])):problems.append('fabric address is not assigned to socket interface')
    if not Path('/dev/infiniband').exists():problems.append('/dev/infiniband unavailable')
    available=available_gib()
    if idle and available<config['minimum_start_gib']:problems.append(f'only {available:.2f} GiB available; need {config["minimum_start_gib"]}')
    image=run(['docker','image','inspect',config['image']],check=False)
    image_id=json.loads(image.stdout)[0]['Id'] if image.returncode==0 else None
    return {'rank':node['rank'],'gpu':gpu.stdout.strip(),'available_gib':available,
        'image_id':image_id,'paths':paths,'problems':problems,'ok':not problems}

def snapshot(config,node,snapshot_id):
    destination=Path(node['state'])/'snapshots'/snapshot_id
    destination.mkdir(parents=True,exist_ok=False,mode=0o700)
    files={};excluded=[]
    for name in node['previous_paths']:
        root=Path(name)
        if not root.exists():raise RuntimeError('snapshot path missing: '+name)
        for p in ([root] if root.is_file() else sorted(root.rglob('*'))):
            if not p.is_file():continue
            if any(x in p.parts for x in ('__pycache__','.git','logs','runs','.venv','node_modules')):continue
            if p.stat().st_size>16*2**20:raise RuntimeError('snapshot contains a large file; narrow previous_paths: '+str(p))
            files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (destination/'files.json').write_text(json.dumps(files,indent=2))
    with tarfile.open(destination/'configs.tar.gz','w:gz') as archive:
        for name in files:archive.add(name,arcname=name.lstrip('/'),recursive=False)
    old=[inspect(name) for name in node['previous_containers']]
    if any(c is None for c in old):raise RuntimeError('a configured previous container does not exist')
    (destination/'containers.private.json').write_text(json.dumps(old,indent=2))
    for c in old:run(['docker','tag',c['Image'],f"spark1017-backup/{config['deployment']}-r{node['rank']}-{c['Name'].lstrip('/')}:{snapshot_id}"])
    for p in destination.iterdir():p.chmod(0o600)
    checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()}
    (destination/'checksums.json').write_text(json.dumps(checks,indent=2))
    return {'rank':node['rank'],'snapshot':str(destination),'files':len(files),'containers':len(old)}

def check_snapshot(config,node,snapshot_id,containers=False):
    folder=Path(node['state'])/'snapshots'/snapshot_id
    for name,expected in json.loads((folder/'checksums.json').read_text()).items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('backup changed: '+name)
    files=json.loads((folder/'files.json').read_text())
    changed=[name for name,expected in files.items() if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected]
    if changed:raise RuntimeError('Original files changed; preserve and inspect before restoring: '+repr(changed))
    checks=[]
    if containers:
        for old in json.loads((folder/'containers.private.json').read_text()):
            now=inspect(old['Name'].lstrip('/'))
            fields=['Image']
            ok=now is not None and now['Image']==old['Image'] and now['State']['Running']==old['State']['Running']
            if now:
                for k in ('Cmd','Entrypoint'):ok=ok and now['Config'][k]==old['Config'][k]
                # Docker may reorder unique environment entries on recreation.
                # Compare effective values, retaining order only for duplicate keys.
                def environment_spec(values):
                    groups={}
                    for value in values or []:
                        key,separator,content=value.partition('=')
                        groups.setdefault(key,[]).append((separator,content))
                    return groups
                ok=ok and environment_spec(now['Config']['Env'])==environment_spec(old['Config']['Env'])
                mounts=lambda c:sorted((x['Source'],x['Destination'],x['RW']) for x in c['Mounts'])
                ok=ok and mounts(now)==mounts(old)
            checks.append({'name':old['Name'].lstrip('/'),'matches_snapshot':bool(ok)})
    return {'rank':node['rank'],'files_verified':len(files),'containers':checks,
        'ok':not changed and all(v['matches_snapshot'] for v in checks)}

def stop(config,node):
    prebuild_stopped=stop_prebuild(config,node)
    name=container_name(config,node['rank']);c=inspect(name)
    if c is None:return {'rank':node['rank'],'stopped':True,'already_absent':True,'prebuild_stopped':prebuild_stopped}
    if not owned(c,config['deployment']):raise RuntimeError('Refusing to stop a foreign container: '+name)
    folder=Path(node['state']);folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    logs=run(['docker','logs',name],check=False)
    (folder/'last-server.log').write_text(logs.stdout+logs.stderr)
    run(['docker','stop','-t','30',name],check=False,timeout=45)
    run(['docker','rm','-f',name])
    return {'rank':node['rank'],'stopped':True}

GUARD=r'''import json,pathlib,subprocess,time,sys
name,deployment,floor,hold=sys.argv[1:];floor=float(floor);hold=float(hold);since=None
while True:
 p=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
 if p.returncode:break
 c=json.loads(p.stdout)[0]
 if c['Config'].get('Labels',{}).get('org.spark1017.deployment')!=deployment or not c['State']['Running']:break
 available=next(int(s.split()[1])/2**20 for s in pathlib.Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
 since=(since or time.monotonic()) if available<float(floor) else None
 if since and time.monotonic()-since>=hold:
  print(json.dumps({'memory_guard_triggered':True,'available_gib':available}),flush=True)
  subprocess.run(['docker','stop','-t','5',name]);break
 time.sleep(1)
'''

def execute(request):
    os.umask(0o077)
    config=request['config'];node=config['nodes'][request['rank']];action=request['action']
    if action=='doctor':return doctor(config,node,request.get('idle',False))
    if action=='plan':return {'rank':node['rank'],'command':docker_command(config,node,request['profile'],request['run_id'])}
    if action=='status':
        c=inspect(container_name(config,node['rank']))
        return {'rank':node['rank'],'exists':bool(c),'owned':owned(c,config['deployment']),
            'running':bool(c and c['State']['Running']),'image_id':c['Image'] if c else None,
            'profile':c['Config'].get('Labels',{}).get('org.spark1017.profile') if c else None}
    if action=='stop':return stop(config,node)
    if action=='snapshot':return snapshot(config,node,request['snapshot_id'])
    if action=='check-snapshot':return check_snapshot(config,node,request['snapshot_id'],request.get('containers',False))
    if action=='prebuild':
        check_writable_paths(node)
        if available_gib()<12:raise RuntimeError('Stop inference before compiling extensions')
        name=prebuild_name(config,node['rank'])
        if inspect(name):raise RuntimeError('Prebuild container already exists; inspect it and stop explicitly: '+name)
        Path(node['cache']).mkdir(parents=True,exist_ok=True,mode=0o700)
        try:
            result=run(['docker','run','--rm','--name',name,'--label',LABEL+'='+config['deployment'],
                '--gpus','all','--memory','8g','--memory-swap','8g',
                '--mount',f'type=bind,src={node["cache"]},dst=/cache','-e','MAX_JOBS=1',
                '--entrypoint','python3',config['image'],'/opt/spark1017/tools/prebuild.py'],timeout=1800)
        finally:
            stop_prebuild(config,node)
        return {'rank':node['rank'],'passed':True,'log':result.stdout+result.stderr}
    if action=='start':
        check_writable_paths(node)
        name=container_name(config,node['rank']);existing=inspect(name)
        if existing:raise RuntimeError('Container name already exists; inspect/status then stop it explicitly: '+name)
        check=doctor(config,node,True)
        if not check['ok'] or not check['image_id']:raise RuntimeError('Preflight failed: '+json.dumps(check))
        for key in ('cache','state'):Path(node[key]).mkdir(parents=True,exist_ok=True,mode=0o700)
        run(['docker','run','--rm','--entrypoint','python3',config['image'],'-c',
            'from spark1017 import __version__;assert __version__=="0.2.0a1"'])
        result=run(docker_command(config,node,request['profile'],request['run_id']))
        guard=Path(node['state'])/'memory_guard.py';guard.write_text(GUARD)
        log=(Path(node['state'])/'memory-guard.log').open('a')
        process=subprocess.Popen(['python3',str(guard),name,config['deployment'],str(config['guard_floor_gib']),str(config['guard_hold_seconds'])],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return {'rank':node['rank'],'container_id':result.stdout.strip(),'guard_pid':process.pid,'image_id':check['image_id']}
    raise ValueError('Unknown node action: '+action)

def main(request):
    try:print(json.dumps({'ok':True,'result':execute(request)}))
    except Exception as error:
        print(json.dumps({'ok':False,'error':str(error)}));raise SystemExit(1)
