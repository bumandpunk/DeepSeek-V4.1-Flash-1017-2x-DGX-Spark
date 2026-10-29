"""Materialize recorded responses for isolated regrading; never execute code."""
import argparse
import json
import pathlib
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--records', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
args = parser.parse_args()
records = [json.loads(line) for line in args.records.read_text().splitlines()]
ids = [row['step'] for row in records]
if len(set(ids)) != len(ids) or not all(re.fullmatch(r'\d+\.\d+', sid) for sid in ids):
    parser.error('duplicate or invalid step identifiers')
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
for row in records:
    folder = out / row['step']
    folder.mkdir()
    for name in ['request', 'response']:
        (folder / (name + '.json')).write_text(json.dumps(row[name], ensure_ascii=False, indent=2))
    (folder / 'prompt.txt').write_text(row['request']['messages'][0]['content'])
    (folder / 'step_code.py').write_text(row['step_code'])
    (folder / 'solution.py').write_text(row['solution'])
    (folder / 'recorded-grade.json').write_text(json.dumps(row['grade'], ensure_ascii=False, indent=2))
print(f'Wrote {len(records)} recorded cases. Run the isolated grade command; do not execute generated files on the host.')
