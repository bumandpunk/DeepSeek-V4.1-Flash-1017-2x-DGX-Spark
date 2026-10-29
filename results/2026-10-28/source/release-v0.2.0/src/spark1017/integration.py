"""Small integration hooks; keep runtime costs and acceptance state separate."""
import json
import os
import time
from pathlib import Path
from .depth import AdaptiveDepth

def choose(decoder, live, drafting, confs, limits, fallback_costs, draft_cost):
    policy = getattr(decoder, '_spark1017_depth', None)
    if policy is None:
        path = os.environ.get('SPARK1017_COSTS')
        frozen = json.loads(Path(path).read_text()) if path else {}
        costs = frozen.get('round_ms', list(fallback_costs))
        # A measured end-to-end round already includes drafting. Do not add it
        # twice; forward-only curves retain the separately supplied draft cost.
        policy = AdaptiveDepth(costs, frozen.get('draft_ms', draft_cost))
        policy.costs_by_live = frozen.get('by_live', {})
        decoder._spark1017_depth = policy
    # The same row count can involve different expert diversity with one or
    # several independent streams. Select a frozen, concurrency-specific curve.
    if str(len(live)) in policy.costs_by_live:
        policy.costs = policy.costs_by_live[str(len(live))]['round_ms']
    ks = policy.choose(confs, limits, len(live), min(decoder.max_rows, len(policy.costs)))
    return {s.sid: k for s, k in zip(drafting, ks)}

def observe(decoder, confs, verified, accepted):
    policy = getattr(decoder, '_spark1017_depth', None)
    if policy is not None:
        policy.observe(confs, verified, accepted)

def record_round(decoder, rows, live, seconds):
    """Calibration only; no local timer enters the distributed runtime policy."""
    path = os.environ.get('SPARK1017_PROFILE')
    if not path or decoder.e.rank != 0:
        return
    with open(path, 'a') as f:
        f.write(json.dumps({'rows': rows, 'live': live, 'seconds': seconds}) + '\n')
