"""Explicit two-node lifecycle, private snapshots and recovery. Standard library only."""
import argparse,concurrent.futures,datetime,fcntl,hashlib,ipaddress,json,os,re,shlex,shutil,subprocess,sys,time
from pathlib import Path
from health import request,verify,wait_ready
from node import PROFILES

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=Path(__file__).resolve().parent

def validate(config):
    if not re.fullmatch(r'[a-z][a-z0-9-]{2,40}',config['deployment']):raise ValueError('Invalid deployment name')
    if len(config['nodes'])!=2 or [n['rank'] for n in config['nodes']]!=[0,1]:raise ValueError('Exactly ordered ranks 0 and 1 are required')
    if config['nodes'][0]['ssh']:raise ValueError('Run the controller on rank 0; its ssh field must be empty')
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.:-]+',config['nodes'][1]['ssh']):raise ValueError('Worker SSH must be user@host')
    if not isinstance(config['vision'],bool):raise ValueError('vision must be boolean')
    for key,lo,hi in [('api_port',1024,65535),('context',8192,524288),('parallel',1,2),('rollback_minutes',10,480)]:
        if not isinstance(config[key],int) or not lo<=config[key]<=hi:raise ValueError('Out of range: '+key)
    if config['minimum_start_gib']<100 or config['guard_floor_gib']<2.5 or not 1<=config['guard_hold_seconds']<=5:raise ValueError('Host memory protection cannot be disabled')
    if config['previous_service'] and not re.fullmatch(r'[a-zA-Z0-9_.@-]+\.service',config['previous_service']):raise ValueError('Invalid previous_service')
    if config['previous_service'] and not config.get('previous_health'):raise ValueError('previous_health is required to verify restoration of a previous service')
    ipaddress.ip_address(config['api_host'])
    for n in config['nodes']:
        ipaddress.ip_address(n['fabric_ip'])
        for key in ('model','engram','cache','state','key_file'):
            path=n[key]
            if not Path(path).is_absolute() or any(x in path for x in ('\n','\r',',','\x00')) or '..' in Path(path).parts:raise ValueError('Use an absolute, comma-free path for '+key)
        protected=[Path(n[k]).resolve() for k in ('model','engram')]
        for key in ('cache','state'):
            q=Path(n[key]).resolve()
            if q==Path('/') or any(q==v or q.is_relative_to(v) or v.is_relative_to(q) for v in protected):raise ValueError('Cache/state must be separate from model and Engram paths')
        if Path(n['cache'])==Path(n['state']):raise ValueError('cache and state must differ')
        if not re.fullmatch(r'[a-zA-Z0-9_.:-]+',n['socket_interface']):raise ValueError('Invalid interface')
        if not n['rdma_devices'] or not all(re.fullmatch(r'[a-zA-Z0-9_.-]+',s) for s in n['rdma_devices']):raise ValueError('Invalid RDMA devices')
        if not isinstance(n['gid_index'],int) or not 0<=n['gid_index']<=255:raise ValueError('Invalid GID index')
    return config

def node_call(config,rank,action,**kwargs):
    payload={'config':config,'rank':rank,'action':action,**kwargs}
    source=(SCRIPTS/'node.py').read_text()+'\nmain('+repr(payload)+')\n'
    target=config['nodes'][rank]['ssh']
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',target,'python3 -'] if target else ['python3','-']
    p=subprocess.run(cmd,input=source,text=True,capture_output=True,timeout=2100 if action=='prebuild' else 180)
    try:data=json.loads(p.stdout)
    except ValueError:raise RuntimeError(f'rank {rank} did not return a node result: '+p.stderr[-1000:])
    if p.returncode or not data.get('ok'):raise RuntimeError(f'rank {rank}: '+data.get('error','node operation failed'))
    return data['result']

def both(config,action,**kwargs):
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(node_call,config,rank,action,**kwargs) for rank in (0,1)]
        results=[];errors=[]
        for rank,job in enumerate(jobs):
            try:results.append(job.result())
            except Exception as error:errors.append(f'rank {rank}: {error}')
    if errors:raise RuntimeError('; '.join(errors))
    return results

def write_private(path,data):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.chmod(0o600);temp.replace(path)

def stamp():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def state_path(config):return Path(config['nodes'][0]['state'])/'lifecycle.json'
def timer_name(config):return config['deployment']+'-rollback'
def service_state(unit):
    if not unit:return 'absent'
    p=subprocess.run(['systemctl','--user','is-active',unit],text=True,capture_output=True)
    return p.stdout.strip()

def snapshot(config):
    path=state_path(config)
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('phase')!='restored':raise RuntimeError('An unfinished lifecycle exists; restore it before taking a new snapshot')
    unit=config['previous_service'];active=service_state(unit)
    if unit and active not in ('active','inactive'):raise RuntimeError('Previous service must be stable (active or inactive), got '+active)
    snapshot_id=stamp();records=both(config,'snapshot',snapshot_id=snapshot_id)
    state={'schema':1,'snapshot_id':snapshot_id,'phase':'snapshotted','previous_service':unit,
        'previous_was_active':active=='active','nodes':records,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_private(path,state)
    write_private(path.parent/'active-config.json',config)
    if unit:
        p=subprocess.run(['systemctl','--user','cat',unit],text=True,capture_output=True,check=True)
        (path.parent/'previous-unit.private.txt').write_text(p.stdout)
    return state

def arm_recovery(config):
    folder=state_path(config).parent/'recovery';folder.mkdir(exist_ok=True,mode=0o700)
    for name in ('cluster.py','node.py','health.py'):shutil.copy2(SCRIPTS/name,folder/name)
    subprocess.run(['systemd-run','--user','--collect','--unit='+timer_name(config),
        '--on-active='+str(config['rollback_minutes'])+'m','--timer-property=AccuracySec=1s',
        '--property=Type=oneshot','--property=TimeoutStartSec=2400',
        '--property=Restart=on-failure','--property=RestartSec=30','--property=StartLimitIntervalSec=0',
        sys.executable,str(folder/'cluster.py'),'restore',
        '--config',str(state_path(config).parent/'active-config.json')],check=True,capture_output=True,text=True)

def disarm(config):
    p=subprocess.run(['systemctl','--user','stop',timer_name(config)+'.timer'],capture_output=True,text=True)
    if p.returncode and 'not loaded' not in p.stderr:raise RuntimeError('Cannot stop recovery timer: '+p.stderr)

def restore(config):
    path=state_path(config)
    if not path.exists():raise RuntimeError('No snapshot; use stop to stop only the experimental containers')
    state=json.loads(path.read_text())
    both(config,'stop')
    both(config,'check-snapshot',snapshot_id=state['snapshot_id'])
    if state['previous_was_active']:
        subprocess.run(['systemctl','--user','start',state['previous_service']],check=True)
        deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            checked=both(config,'check-snapshot',snapshot_id=state['snapshot_id'],containers=True)
            if all(x['ok'] for x in checked):break
            if service_state(state['previous_service'])=='failed':raise RuntimeError('Previous service failed to restart')
            time.sleep(5)
        else:raise TimeoutError('Previous container specifications did not return to snapshot')
    else:checked=both(config,'check-snapshot',snapshot_id=state['snapshot_id'],containers=True)
    if not all(x['ok'] for x in checked):raise RuntimeError('Restored state differs from snapshot')
    health=None
    if state['previous_was_active']:
        previous=config['previous_health'];key=Path(previous['key_file']).read_text().strip()
        wait_ready(previous['base'],key,previous['model'],1800)
        health=verify(previous['base'],key,previous['model'])
    state.update(phase='restored',recovery_armed=False,restored_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),restore_checks=checked,restore_api=health)
    write_private(path,state);disarm(config)
    return {'restored':True,'checks':checked,'previous_api':health}

def base_url(config):
    host='127.0.0.1' if config['api_host']=='0.0.0.0' else config['api_host']
    return f"http://{host}:{config['api_port']}"

def start(config,profile,takeover=False):
    statuses=both(config,'status')
    if any(s['exists'] for s in statuses):raise RuntimeError('A target container already exists; status and stop before changing profiles')
    # Validate paths/fabric/images before stopping an existing service.
    checks=both(config,'doctor')
    if any(not c['ok'] or not c['image_id'] for c in checks):raise RuntimeError('Preflight failed: '+json.dumps(checks))
    if checks[0]['image_id']!=checks[1]['image_id']:raise RuntimeError('Image IDs differ between ranks')
    if config['previous_service'] and service_state(config['previous_service'])=='active' and not takeover:raise RuntimeError('Use --takeover to snapshot and stop the configured previous service')
    path=state_path(config)
    if not path.exists() or json.loads(path.read_text()).get('phase')=='restored':snapshot(config)
    state=json.loads(path.read_text())
    if state['phase'] not in ('snapshotted','stopped'):raise RuntimeError('Finish or restore the previous lifecycle first')
    if not state.get('recovery_armed'):arm_recovery(config)
    run_id=stamp();state.update(phase='starting',recovery_armed=True,profile=profile,run_id=run_id);write_private(path,state)
    try:
        if state['previous_was_active'] and service_state(state['previous_service'])=='active':
            if not takeover:raise RuntimeError('Explicit --takeover required')
            subprocess.run(['systemctl','--user','stop',state['previous_service']],check=True,timeout=240)
        checks=both(config,'doctor',idle=True)
        if any(not c['ok'] for c in checks):raise RuntimeError('Idle preflight failed: '+json.dumps(checks))
        first=node_call(config,0,'start',profile=profile,run_id=run_id)
        # Rank 0 owns the TCP store. Rank 1 retries while it starts.
        second=node_call(config,1,'start',profile=profile,run_id=run_id)
        key=Path(config['nodes'][0]['key_file']).read_text().strip()
        wait_ready(base_url(config),key,config['model_name'],1800)
        result=verify(base_url(config),key,config['model_name'])
        state.update(phase='running',containers=[first,second],api_checks=result);write_private(path,state)
        return {'started':True,'profile':profile,'api':result,'rollback_minutes':config['rollback_minutes'],'run_id':run_id}
    except BaseException:
        # Every started rank belongs to this deployment. Recovery is independently armed too.
        try:restore(config)
        except Exception as error:print('Automatic recovery incomplete; timer remains armed: '+str(error),file=sys.stderr)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['doctor','plan','prebuild','snapshot','start','stop','status','verify','restore','confirm'])
    p.add_argument('--config',type=Path,default=ROOT/'config/cluster.local.json')
    p.add_argument('--profile',choices=PROFILES,default='baseline');p.add_argument('--takeover',action='store_true')
    p.add_argument('--idle',action='store_true');a=p.parse_args()
    config=validate(json.loads(a.config.read_text()));os.umask(0o077)
    state=Path(config['nodes'][0]['state']);state.mkdir(parents=True,exist_ok=True,mode=0o700)
    # Serialize lifecycle mutations, including the independent timer.
    with (state/'lifecycle.lock').open('a') as lock:
        if a.action not in ('doctor','plan','status'):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.action=='doctor':result=both(config,'doctor',idle=a.idle)
        elif a.action=='plan':result=both(config,'plan',profile=a.profile,run_id='dry-run')
        elif a.action=='status':result=both(config,'status')
        elif a.action=='prebuild':result=both(config,'prebuild')
        elif a.action=='snapshot':result=snapshot(config)
        elif a.action=='start':result=start(config,a.profile,a.takeover)
        elif a.action=='restore':result=restore(config)
        elif a.action=='verify':result=verify(base_url(config),Path(config['nodes'][0]['key_file']).read_text().strip(),config['model_name'])
        elif a.action=='confirm':
            verify(base_url(config),Path(config['nodes'][0]['key_file']).read_text().strip(),config['model_name']);disarm(config)
            s=json.loads(state_path(config).read_text());s['recovery_armed']=False;write_private(state_path(config),s);result={'rollback_disarmed':True}
        elif a.action=='stop':
            result=both(config,'stop')
            if state_path(config).exists():
                s=json.loads(state_path(config).read_text())
                if s['phase']!='restored':s['phase']='stopped';write_private(state_path(config),s)
        print(json.dumps(result,indent=2))
        if a.action=='doctor' and any(not x['ok'] for x in result):return 1
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (RuntimeError,ValueError,TimeoutError,OSError) as error:print(str(error),file=sys.stderr);raise SystemExit(1)
