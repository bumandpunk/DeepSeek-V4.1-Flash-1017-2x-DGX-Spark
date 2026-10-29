"""Recompute release evidence; no inference and no generated-code execution."""
import collections,hashlib,json,pathlib

HERE=pathlib.Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def speed_signature(folder):
    records=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
    for record in records:
        assert record['request_sha256']==hashlib.sha256(json.dumps(record['request'],sort_keys=True).encode()).hexdigest()
        assert record['response']['tokens'] and record['response']['ttft_s'] is not None
    return collections.Counter((r['request_sha256'],r['response']['text'],r['response']['tokens']) for r in records)

def recompute():
    bench=HERE/'bench';baseline=speed_signature(bench/'baseline-speed')
    assert sum(baseline.values())==58
    for profile in ('baseline','edit','prefill-experimental'):
        api=read(bench/(profile+'-api')/'summary.json')
        assert len(api)==3 and all(v['passed'] for v in api)
    for profile in ('edit','prefill-experimental'):assert baseline==speed_signature(bench/(profile+'-speed')),profile
    full={p:read(bench/(p+'-fullfile')/'summary.json') for p in ('baseline','edit')}
    assert len(full['baseline'])==len(full['edit'])==6
    assertions={}
    for profile,rows in full.items():
        assertions[profile]=sum(v['grade']['passed'] for v in rows)
        assert all(v['grade']['passed']==v['grade']['total'] for v in rows)
        assert all(0<v['tokens']<16384 for v in rows)
        assert assertions[profile]==836
    for left,right in zip(full['baseline'],full['edit']):
        for key in ('name','rep','request_sha256','text_sha256','tokens','grade'):assert left[key]==right[key],key
    cases={p:read(bench/(p+'-canitedit')/'summary.json') for p in ('baseline','edit')}
    assert len(cases['baseline'])==len(cases['edit'])==24
    for left,right in zip(cases['baseline'],cases['edit']):
        for key in ('id','rep','request_sha256','text_sha256','solution_sha256','passed','tokens'):assert left[key]==right[key],key
        assert not left['possibly_truncated'] and not right['possibly_truncated']
    sc={};parents={};steps=[];statuses={};exhausted={}
    for profile in ('baseline','edit'):
        root=bench/(profile+'-scicode');grades={p.parent.name:read(p) for p in root.glob('*/grade.json')}
        assert len(grades)==50
        assert all(v['status'] in ('passed','failed','timeout') for v in grades.values())
        statuses[profile]=dict(collections.Counter(v['status'] for v in grades.values()))
        exhausted[profile]=sum(read(p)['choices'][0]['finish_reason']=='length' for p in root.glob('*/response.json'))
        sc[profile]=sum(v['status']=='passed' for v in grades.values())
        parent_ids={sid.split('.')[0] for sid in grades}
        parents[profile]=sum(all(v['status']=='passed' for sid,v in grades.items() if sid.split('.')[0]==pid) for pid in parent_ids)
        assert len(parent_ids)==15
        steps.append(grades)
    assert steps[0].keys()==steps[1].keys()
    for sid in steps[0]:
        left=bench/'baseline-scicode'/sid;right=bench/'edit-scicode'/sid
        for name in ('request.json','step_code.py','solution.py'):assert left.joinpath(name).read_bytes()==right.joinpath(name).read_bytes(),(sid,name)
        assert read(left/'response.json')['choices']==read(right/'response.json')['choices'],sid
        assert steps[0][sid]['status']==steps[1][sid]['status'],sid
    gold=[read(p) for p in (bench/'reference-scicode').glob('*/grade.json')]
    assert len(gold)==50 and sum(v['status']=='passed' for v in gold)==48
    previous=HERE/'pre-workspace-fix-quality'
    provenance=read(previous/'provenance.json')
    assert digest(previous/'model-before.py')==read(previous/'installed-image.json')['source_hashes']['tensorfold/tensorfold/families/deepseek_v41/cuda/model.py']
    assert provenance['context']==524288 and provenance['a0_input_metadata_equal']
    assert read(previous/'build.json')['image_id']==provenance['image_id']
    for name,h in provenance['unchanged_harnesses'].items():assert digest(PROJECT/name)==h
    old_full=read(previous/'baseline-fullfile/summary.json')
    assert len(old_full)==6
    for left,right in zip(old_full,full['baseline']):
        for key in ('name','rep','request_sha256','text_sha256','tokens','grade'):assert left[key]==right[key],('pre-fix',key)
    old_can=read(previous/'baseline-canitedit/summary.json')
    assert len(old_can)==24
    for left,right in zip(old_can,cases['baseline']):
        for key in ('id','rep','request_sha256','text_sha256','solution_sha256','passed','tokens'):assert left[key]==right[key],('pre-fix',key)
    for sid in steps[0]:
        left=previous/'baseline-scicode'/sid;right=bench/'baseline-scicode'/sid
        for name in ('request.json','step_code.py','solution.py'):assert (left/name).read_bytes()==(right/name).read_bytes(),('pre-fix',sid,name)
        assert read(left/'response.json')['choices']==read(right/'response.json')['choices'],('pre-fix',sid)
        assert read(left/'grade.json')['status']==read(right/'grade.json')['status'],('pre-fix',sid)
    canitedit={p:sum(v['passed'] for v in rows) for p,rows in cases.items()}
    references={p:read(bench/('reference-'+p)/'complete.json') for p in ('gold','before')}
    assert references['gold']['passed']==12 and references['before']['passed']==0
    return {'fixed_requests_per_profile':58,'profiles':3,'fullfile_generations_per_profile':6,'fullfile_assertions':assertions,
        'canitedit_generations_per_profile':24,'canitedit_passed':canitedit,'scicode_steps_per_profile':50,
        'scicode_passed':sc,'scicode_parents_passed':parents,'scicode_reference_passed':48,
        'scicode_statuses':statuses,'scicode_completion_budget_exhausted':exhausted,
        'paired_complete_outputs_and_verdicts_equal':True,'pre_workspace_fix_quality_equal':True}

def main():
    summary=read(HERE/'summary.json');assert summary['release_gate_passed']
    for path,expected in read(HERE/'files.sha256.json').items():assert digest(HERE/path)==expected,path
    build=read(HERE/'build.json')
    assert build['image_id']==build['worker_image_id'] and build['no_application_build_cache']
    assert hashlib.sha256(json.dumps(build['source_files'],sort_keys=True).encode()).hexdigest()==build['source_sha256']
    installed=read(HERE/'installed-image.json');assert installed['official_pypi_hashes_verified']
    modules=PROJECT/'src/spark1017'
    expected_modules={'spark1017/spark1017/'+p.relative_to(modules).as_posix():digest(p) for p in modules.rglob('*.py')}
    actual_modules={k:v for k,v in installed['source_hashes'].items() if k.startswith('spark1017/spark1017/')}
    assert expected_modules==actual_modules,'Installed Spark1017 sources differ from release source'
    locks=read(PROJECT/'docker/package-hashes.json')
    for package in installed['downloaded_packages']:
        lock=locks[package['name']]
        assert package['version']==lock['version'] and lock['files'][package['filename']]==package['sha256']
    for path,expected in build['source_files'].items():assert digest(PROJECT/path)==expected,'Build source differs: '+path
    for path,expected in read(HERE/'client-harnesses.json')['files'].items():assert digest(PROJECT/path)==expected,'Client harness differs: '+path
    profiles=read(HERE/'profiles.json')
    policy=read(HERE/'runtime-policy.json');assert policy['guard_floor_gib']>=2.5
    expected={'baseline':('0','0'),'edit':('1','0'),'prefill-experimental':('0','all')}
    assert set(profiles)==set(expected)
    for profile,ranks in profiles.items():
        assert [r['rank'] for r in ranks]==[0,1]
        assert read(HERE/'bench'/(profile+'-speed')/'manifest.json')['context_capacity']==policy['context']
        for rank in ranks:
            assert rank['image_id']==build['image_id']
            assert rank['labels']['org.spark1017.source-sha256']==build['source_sha256']
            env=rank['environment']
            command=rank['command'];assert int(command[command.index('--context')+1])==policy['context']
            assert float(env['TF_DS_HOST_MIN_GIB'])>=policy['guard_floor_gib']+0.5
            assert (env['SPARK1017_LOOKUP'],env['SPARK1017_RDMA_POST'])==expected[profile]
            mounts={m['destination']:m for m in rank['mounts']}
            assert all(mounts[p]['read_only'] for p in ('/model','/engram','/run/secrets/api-key'))
            assert not any(p.startswith('/opt/') for p in mounts)
    engram=read(HERE/'engram-cuda.json');assert engram['image_id']==build['image_id']
    assert set(engram['nodes'])=={'head','worker'}
    for node in engram['nodes'].values():
        assert node['passed'] and node['exact_cases']==9 and node['peak_saved_bytes']>=450*2**20
    prebuild=read(HERE/'prebuild.json')
    assert len(prebuild)==2 and all(r['passed'] for r in prebuild)
    inputs={role:read(HERE/('input-files-'+role+'.json')) for role in ('head','worker')}
    for row in inputs.values():assert row['fresh_rank_cache_absent'] and row['fresh_native_cache_empty']
    for kind in ('model','engram'):
        # Export audit timestamps differ by node; they are not model tensors.
        left={k:v for k,v in inputs['head'][kind].items() if k!='EXPORT_VERIFIED.json'}
        right={k:v for k,v in inputs['worker'][kind].items() if k!='EXPORT_VERIFIED.json'}
        assert left==right,kind
    events={v['event']:v for v in read(HERE/'timeline.json')}
    assert events['final_build_record_verified']['result']['image_id']==build['image_id']
    assert events['injected_startup_failure_recovered']['result']['api']['passed']==5
    assert events['original_restored_by_frozen_recovery_service']['result']['api']['passed']==5
    restore=read(HERE/'restoration.json');assert restore['passed']
    for profile,checks in read(HERE/'profile-checks.json').items():
        assert checks['profile']==profile and checks['phase']=='running' and checks['api_checks']['passed']==5
    ci=read(HERE/'ci-environment.json')
    assert ci['ci_exit_status']=='0' and ci['clean_venv']['ci_exit_status']=='0'
    assert ci['build_source_sha256']==build['source_sha256']
    quality=recompute();assert quality==summary['quality']
    print(json.dumps({'release':'v0.2.0-alpha.1','record_consistency':'passed','quality':quality,
        'scope':'Recorded clean-deployment, quality and restore evidence; no GPU rerun.'},indent=2))

if __name__=='__main__':main()
