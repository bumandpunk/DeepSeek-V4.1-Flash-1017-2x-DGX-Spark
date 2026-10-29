"""Recompute the second-round tables from immutable request and grader records."""
import argparse, collections, hashlib, json, math, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def summarize_speed(folder):
    manifest=read(folder/'manifest.json')
    raw=rows(folder/'requests.jsonl')
    for row in raw:
        assert digest(json.dumps(row['request'],sort_keys=True))==row['request_sha256']
        answer=row['response']
        if answer.get('decode_tps') is not None:
            assert math.isclose(answer['decode_tps'],(answer['tokens']-1)/answer['decode_s'],rel_tol=1e-12)
    complete=(folder/'summary.json').exists()
    summary=read(folder/'summary.json') if complete else {
        key:read(folder/(key+'.json')) for key in ('decode','concurrent','prefill','depth') if (folder/(key+'.json')).exists()}
    recovered=[];counts=collections.Counter()
    for row in raw:
        body=row['request']['body'];answer=row['response'];length=answer.get('prompt_tokens')
        if body.get('max_tokens')==1 and length in (8192,32768,131072) and answer.get('tokens')==1 and answer.get('ttft_s'):
            recovered.append({'length':length,'rep':counts[length],'prompt_tokens':length,
                'ttft_s':answer['ttft_s'],'prefill_tps':length/answer['ttft_s']});counts[length]+=1
    if 'prefill' in summary:assert summary['prefill']==recovered,folder.name
    summary['prefill']=recovered
    result={'complete':complete,'requests':len(raw),'manifest':manifest,'decode':{},'prefill':{},'concurrent':[],'depth':[]}
    offset=0
    for row in summary.get('decode',[]):
        run=raw[offset+1:offset+1+row['reps']];offset+=row['reps']+1
        assert len(run)==row['reps'] and len({v['request_sha256'] for v in run})==1
        calculated={'decode_tps_median':statistics.median(v['response']['decode_tps'] for v in run),
            'decode_tps_all':[round(v['response']['decode_tps'],3) for v in run],
            'ttft_s_median':statistics.median(v['response']['ttft_s'] for v in run),
            'tokens_all':[v['response']['tokens'] for v in run],
            'identical_texts':len({v['response']['text'] for v in run})==1}
        assert calculated=={k:row[k] for k in calculated},(folder.name,row['prompt'])
        result['decode'][row['prompt']]=calculated
        result['decode'][row['prompt']]['whole_request_seconds_median']=statistics.median(v['response']['end']-v['response']['start'] for v in run)
    for length in sorted({row['length'] for row in summary.get('prefill',[])}):
        group=[row for row in summary['prefill'] if row['length']==length]
        result['prefill'][str(length)]={'tps_median':statistics.median(row['prefill_tps'] for row in group),
            'ttft_s_median':statistics.median(row['ttft_s'] for row in group),'raw':group}
    result['concurrent']=summary.get('concurrent',[])
    result['depth']=summary.get('depth',[])
    return result

def paired(base, candidate):
    def bank(path):
        result=collections.defaultdict(collections.Counter);requests=collections.Counter();failed=0
        for row in rows(path):
            requests[row['request_sha256']]+=1
            if row['response'].get('tokens') is None or row['response'].get('ttft_s') is None:
                failed+=1;continue
            result[row['request_sha256']][(digest(row['response']['text']),row['response']['tokens'])]+=1
        return result,requests,failed
    a,ar,af=bank(base);b,br,bf=bank(candidate)
    matched=sum(sum((a[k]&b[k]).values()) for k in a.keys() & b.keys())
    total_a=sum(ar.values());total_b=sum(br.values())
    return {'baseline_requests':total_a,'candidate_requests':total_b,'identical_response_multiset_count':matched,
        'baseline_incomplete_requests':af,'candidate_incomplete_requests':bf,
        'same_request_multiset':ar==br,'all_texts_and_token_counts_equal':af==bf==0 and a==b,
        'complete_baseline_output_multiset_is_subset':af==0 and bool(a) and all(not (values-b.get(key,collections.Counter())) for key,values in a.items())}

def scicode(records):
    parents=collections.defaultdict(list)
    for row in records:parents[row['step'].split('.')[0]].append(row['grade']['status']=='passed')
    return {'total':len(records),'passed':sum(row['grade']['status']=='passed' for row in records),
        'statuses':dict(collections.Counter(row['grade']['status'] for row in records)),
        'parent_total':len(parents),'parent_all_passed':sum(all(v) for v in parents.values())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results/2026-10-28');a=p.parse_args()
    result={'speed':{},'pairs':{},'quality':{}}
    for folder in sorted((a.results/'raw/general').iterdir()):
        if (folder/'requests.jsonl').exists():result['speed'][folder.name]=summarize_speed(folder)
    for base,candidate in [('e320-base-v2','e320-final-v1'),('e320-return-baseline','e320-final-v1'),
                            ('a0-text-baseline','a0-text-fused'),('a0-text-baseline','a0-mia-native'),('a0-mia-native','a0-text-fused'),
                            ('a0-return-decode','a0-text-fused')]:
        if base in result['speed'] and candidate in result['speed']:
            result['pairs'][base+'__'+candidate]=paired(a.results/'raw/general'/base/'requests.jsonl',a.results/'raw/general'/candidate/'requests.jsonl')
    baseline=rows(ROOT/'results/2026-10-27/raw/quality/baseline_e320_2048.jsonl')
    final=a.results/'raw/quality/e320-final-scicode.jsonl'
    if final.exists():
        candidate=rows(final);by_id={row['step']:row for row in baseline}
        comparisons=[{'step':r['step'],'request_equal':r['request']==by_id[r['step']]['request'],
            'step_code_equal':r['step_code']==by_id[r['step']]['step_code'],
            'grade_equal':r['grade']['status']==by_id[r['step']]['grade']['status']} for r in candidate]
        result['quality']['scicode']={'baseline':scicode(baseline),'candidate':scicode(candidate),'pairs':comparisons,
            'all_requests_equal':all(r['request_equal'] for r in comparisons),
            'all_step_code_equal':all(r['step_code_equal'] for r in comparisons),
            'all_grades_equal':all(r['grade_equal'] for r in comparisons)}
    base_folder=ROOT/'results/2026-10-27/raw/canitedit/canitedit-e320-baseline'
    folder=a.results/'raw/canitedit/e320-final-canitedit'
    if (folder/'complete.json').exists():
        b=read(base_folder/'summary.json');c=read(folder/'summary.json');by_id={(r['id'],r['rep']):r for r in b}
        comparisons=[{'id':r['id'],'rep':r['rep'],
            'request_equal':read(folder/f"{r['id']}-{r['rep']}"/'record.json')['request']==read(base_folder/f"{r['id']}-{r['rep']}"/'record.json')['request'],
            'text_equal':r['text_sha256']==by_id[r['id'],r['rep']]['text_sha256'],
            'solution_equal':r['solution_sha256']==by_id[r['id'],r['rep']]['solution_sha256'],
            'grade_equal':r['passed']==by_id[r['id'],r['rep']]['passed']} for r in c]
        result['quality']['canitedit']={'baseline_passed':sum(r['passed'] for r in b),'candidate_passed':sum(r['passed'] for r in c),
            'total':len(c),'repetitions':2,'pairs':comparisons}
    (a.results/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'speed_arms':len(result['speed']),'paired_comparisons':len(result['pairs']),'quality':list(result['quality'])}))

if __name__=='__main__':main()
