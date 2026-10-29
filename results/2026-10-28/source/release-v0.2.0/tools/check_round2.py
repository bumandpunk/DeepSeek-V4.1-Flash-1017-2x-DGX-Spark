"""Validate released evidence and source digests without running generated code."""
import collections,hashlib,json,re
from pathlib import Path
from summarize_round2 import summarize_speed, paired, scicode, rows

r=Path(__file__).resolve().parents[1];d=r/'results/2026-10-28'
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=read(d/'summary.json')
for label,summary in s['speed'].items():
    assert summarize_speed(d/'raw/general'/label)==summary,label
for key,summary in s['pairs'].items():
    base,candidate=key.split('__')
    assert paired(d/'raw/general'/base/'requests.jsonl',d/'raw/general'/candidate/'requests.jsonl')==summary,key
for label in ('e320-base-v2','e320-return-baseline','e320-final-v1'):
    assert s['speed'][label]['complete']
    assert s['speed'][label]['requests']==58
    assert s['speed'][label]['manifest']['repetitions']==9
    assert s['speed'][label]['manifest']['prefix_seed_scheme']=='20261027 + (length << 4) + occurrence; wrapper v2'
for key in ('e320-base-v2__e320-final-v1','e320-return-baseline__e320-final-v1'):
    assert s['pairs'][key]['same_request_multiset']
    assert s['pairs'][key]['all_texts_and_token_counts_equal']
sc=s['quality']['scicode'];assert sc['candidate']['total']==50
assert sc['all_requests_equal'] and sc['all_step_code_equal'] and sc['all_grades_equal']
assert scicode(rows(d/'raw/quality/e320-final-scicode.jsonl'))==sc['candidate']
baseline={row['step']:row for row in rows(r/'results/2026-10-27/raw/quality/baseline_e320_2048.jsonl')}
candidate={row['step']:row for row in rows(d/'raw/quality/e320-final-scicode.jsonl')}
assert baseline.keys()==candidate.keys() and len(candidate)==50
for step,row in candidate.items():
    assert row['request']==baseline[step]['request'],step
    assert row['step_code']==baseline[step]['step_code'],step
    assert row['grade']['status']==baseline[step]['grade']['status'],step
can=s['quality']['canitedit'];assert can['total']==24
assert all(row['request_equal'] and row['text_equal'] and row['solution_equal'] and row['grade_equal'] for row in can['pairs'])
baseline_can=r/'results/2026-10-27/raw/canitedit/canitedit-e320-baseline'
for row in read(d/'raw/canitedit/e320-final-canitedit/summary.json'):
    case=d/'raw/canitedit/e320-final-canitedit'/f"{row['id']}-{row['rep']}"
    base=baseline_can/case.name
    text=read(case/'record.json')['response']['text']
    assert hashlib.sha256(text.encode()).hexdigest()==row['text_sha256']
    assert digest(case/'solution.py')==row['solution_sha256']
    assert read(case/'grade.json')['passed']==row['passed']
    assert read(case/'record.json')['request']==read(base/'record.json')['request']
    assert text==read(base/'record.json')['response']['text']
    assert (case/'solution.py').read_bytes()==(base/'solution.py').read_bytes()
    assert read(case/'grade.json')['passed']==read(base/'grade.json')['passed']
for label in ('a0-text-baseline','a0-text-fused','a0-mia-native'):
    assert s['speed'][label]['complete'] and s['speed'][label]['requests']==58
for key in ('a0-text-baseline__a0-text-fused','a0-text-baseline__a0-mia-native','a0-mia-native__a0-text-fused'):
    assert s['pairs'][key]['all_texts_and_token_counts_equal']
ret=s['pairs']['a0-return-decode__a0-text-fused']
assert ret['baseline_requests']==30 and ret['identical_response_multiset_count']==30
assert ret['complete_baseline_output_multiset_is_subset']
assert read(d/'profiles/tokenization-parity.json')['all_prompt_hashes_and_token_ids_equal']
assert read(d/'profiles/native-binary-parity.json')['equal_across_ranks']
for label,row in read(d/'cold-prefix-validation.json')['runs'].items():
    log=d/row['server_log'];assert digest(log)==row['server_log_sha256']
    events=[{'prompt':int(a),'cached':int(b)} for a,b in re.findall(r'\bprompt=(\d+) cached=(\d+)',log.read_text())
        if int(a) in (8192,32768,131072)]
    assert events==row['records'] and all(v['cached']==0 for v in events),label
    assert collections.Counter(v['prompt'] for v in events)=={8192:3,32768:3,131072:4}
for path,expected in read(d/'release-source-manifest.json').items():assert digest(r/path)==expected,path
snap=d/'source/e320-final-v1';provenance=read(snap/'provenance.json')
for path,expected in provenance['sources'].items():assert digest(snap/path)==expected,path
for rank in (0,1):
    micro=read(d/'experiments'/f'transport-post-pipeline-v3-r{rank}.json')
    assert micro['skew_stress_transactions']==100
    assert all(row['bit_exact'] and row['graph_exact'] for row in micro['cases'])
restored=read(d/'deployment-restore.json')
assert restored['api_contract_passed']==5 and not restored['head']['changed_files']
assert restored['lan_generation_status']==200 and restored['web_status']==200
cleanup=read(d/'experiments/helper-network-restore.json')
assert all(cleanup[k] for k in ('task_firewall_rule_removed','other_ipv4_entries_match_snapshot','temporary_proxy_removed','helper_stopped'))
print(json.dumps({'record_consistency':'passed','e320_speed_requests':174,'a0_speed_requests':174,
    'a0_return_short_requests':30,'scicode_paired_steps':50,
    'canitedit_paired_generations':24,'runtime_source_files':len(provenance['sources']),
    'release_source_files':len(read(d/'release-source-manifest.json')),
    'scope':'Artifact consistency only; no inference or generated code execution.'},indent=2))
