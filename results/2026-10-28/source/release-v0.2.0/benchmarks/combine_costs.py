"""Make a frozen cost table stratified by number of live streams."""
import argparse
import hashlib
import json
from pathlib import Path
from fit_costs import fit

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, action='append', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows, sources = [], []
for file in a.input:
    raw = file.read_bytes()
    sources.append({'name': file.name, 'sha256': hashlib.sha256(raw).hexdigest()})
    rows += [json.loads(line) for line in raw.splitlines()]
curves = {}
for live in sorted({r['live'] for r in rows}):
    subset = [r for r in rows if r['live'] == live]
    if len(subset) >= 100:
        curves[str(live)] = fit(subset, max_rows=live * 6)
if not {'1', '2'}.issubset(curves):
    raise ValueError('need independent single- and two-stream measurements')
result = {'schema': 2, 'timer': 'whole_round', 'draft_ms': 0.0,
          'round_ms': curves['2']['round_ms'], 'by_live': curves,
          'calibration_sources': sources}
with a.output.open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
