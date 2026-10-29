"""Check that release tables agree with bundled raw records; execute no model code."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', nargs='?', type=Path, default=ROOT/'results/2026-10-27')
args = parser.parse_args()
results = args.results.resolve()


def read(path):
    return json.loads(path.read_text())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


source = read(results/'extension-source-manifest.json')
source_root = results/'source' if (results/'source').is_dir() else ROOT
for name, info in source.items():
    expected = info if isinstance(info, str) else info['sha256']
    assert hashlib.sha256((source_root/name).read_bytes()).hexdigest() == expected, name

edit_count = 0
for label in ['edit-public-mia-a0', 'edit-a0-lookup', 'edit-e320-baseline', 'edit-e320-lookup']:
    folder = results/'raw/edit'/label
    rows = read(folder/'summary.json')
    assert len(rows) == 6
    for row in rows:
        record = read(folder/(row['name']+'-'+str(row['rep'])+'.json'))
        response = record['response']
        assert digest(response['text']) == row['text_sha256']
        assert response['tokens'] == row['tokens']
        assert row['grade']['passed'] == row['grade']['total']
    assert sum(row['grade']['total'] for row in rows) == 836
    edit_count += len(rows)

quality = read(results/'quality-comparison.json')
for label in ['baseline_e320_2048', 'lookup_e320_2048']:
    rows = jsonl(results/'raw/quality'/(label+'.jsonl'))
    assert len(rows) == quality['runs'][label]['total']
    assert sum(x['grade']['status']=='passed' for x in rows) == quality['runs'][label]['passed']
    for row in rows:
        step = quality['runs'][label]['steps'][row['step']]
        assert digest(row['step_code']) == step['step_code_sha256']
        assert row['grade']['status'] == step['status']

can = read(results/'canitedit-comparison.json')
for label, key in [('canitedit-e320-baseline','baseline_passed'), ('canitedit-e320-lookup','lookup_passed')]:
    folder = results/'raw/canitedit'/label
    rows = read(folder/'summary.json')
    assert len(rows) == can['total_generations_per_arm']
    assert sum(x['passed'] for x in rows) == can[key]
    for row in rows:
        case = folder/(str(row['id'])+'-'+str(row['rep']))
        response = read(case/'record.json')['response']
        assert digest(response['text']) == row['text_sha256']
        assert digest((case/'solution.py').read_text()) == row['solution_sha256']
        assert read(case/'grade.json')['passed'] == row['passed']

fixed = read(results/'matched-request-checks.json')
arms = []
for label in ['e320-fixed-baseline','spark-fixed-lookup']:
    rows = jsonl(results/'raw/general'/label/'requests.jsonl')
    grouped = collections.defaultdict(collections.Counter)
    for row in rows:
        request_sha = digest(json.dumps(row['request'], sort_keys=True))
        assert request_sha == row['request_sha256']
        grouped[request_sha][(digest(row['response']['text']),row['response']['tokens'])] += 1
    assert len(rows) == fixed['requests_per_arm']
    arms.append(grouped)
assert arms[0].keys() == arms[1].keys()
assert (arms[0] == arms[1]) == fixed['all_outputs_equal']

print(json.dumps({'source_files':len(source), 'complete_edit_generations':edit_count,
                  'scicode_steps':100, 'canitedit_generations':48,
                  'fixed_suite_requests_per_arm':fixed['requests_per_arm'],
                  'record_consistency':'passed',
                  'scope':'Checks recorded artifacts only; does not rerun inference or execute generated code.'},indent=2))
