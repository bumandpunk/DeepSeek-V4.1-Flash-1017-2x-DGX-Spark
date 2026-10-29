"""Freeze a monotone row-cost curve from disjoint calibration requests.

The observed timer covers the complete verification round, including drafting.
Missing row sizes are explicitly reported and interpolated (never called measured).
Run this offline and copy the identical JSON to both tensor-parallel ranks.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def fit(rows, max_rows=12, min_samples=20):
    grouped = {}
    for row in rows:
        n, ms = row['rows'], 1000 * row['seconds']
        if type(n) is int and 1 <= n <= max_rows and 0 < ms < 1000:
            grouped.setdefault(n, []).append(ms)
    measured = {n: statistics.median(v) for n, v in grouped.items() if len(v) >= min_samples}
    if len(measured) < 2:
        raise ValueError('need at least two row sizes with enough samples')
    # Weighted pool-adjacent-violators. Noise may make a larger batch look faster;
    # this estimate enforces a nondecreasing expected service time.
    blocks = []
    for n, median in sorted(measured.items()):
        count = len(grouped[n])
        blocks.append(([n], median * count, count))
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
            b, a = blocks.pop(), blocks.pop()
            blocks.append((a[0] + b[0], a[1] + b[1], a[2] + b[2]))
    smoothed = {n: total / count for ns, total, count in blocks for n in ns}
    keys = sorted(smoothed)
    def value(n):
        if n in smoothed:
            return smoothed[n]
        lo = max((k for k in keys if k < n), default=keys[0])
        hi = min((k for k in keys if k > n), default=keys[-1])
        if lo == hi:
            # Boundary extrapolation uses the nearest observed slope, bounded
            # below by 1 ms; no division by zero for the two-point minimum.
            lo, hi = keys[:2] if n < keys[0] else keys[-2:]
        return max(1.0, smoothed[lo] + (n - lo) * (smoothed[hi] - smoothed[lo]) / (hi - lo))
    return {'schema': 1, 'timer': 'whole_round', 'draft_ms': 0.0,
            'round_ms': [round(value(n), 6) for n in range(1, max_rows + 1)],
            'min_samples': min_samples,
            'observed': {str(n): {'count': len(v), 'median_ms': statistics.median(v)} for n, v in sorted(grouped.items())},
            'unmeasured_rows': [n for n in range(1, max_rows + 1) if n not in measured]}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--max-rows', type=int, default=12)
    a = p.parse_args()
    raw = a.input.read_bytes()
    result = fit([json.loads(line) for line in raw.splitlines()], a.max_rows)
    result['calibration_sha256'] = hashlib.sha256(raw).hexdigest()
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
