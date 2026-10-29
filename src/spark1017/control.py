"""One ordered rank collective for plan agreement and memory admission.

The ordinary path performs these checks in two consecutive collectives. Pack
the identical 64-bit plan digest with each rank's local capacity vote instead.
No check is sampled or delayed. A failed vote retains the original collective
eviction/recheck path; a plan mismatch fails before any model collective.
"""
import hashlib
import json


def packet(plan, fits):
    digest = int.from_bytes(hashlib.sha256(json.dumps(plan).encode()).digest()[:8], 'big')
    return [(digest >> bit) & 65535 for bit in (0, 16, 32, 48)] + [int(fits), 0, 0, 0]


def consensus(rows, world):
    if len(rows) != world or not rows or any(len(row) != 8 for row in rows):
        raise ValueError('rank control shape mismatch')
    if any(row[:4] != rows[0][:4] for row in rows):
        raise ValueError('rank control plan mismatch')
    if any(row[4] not in (0, 1) or row[5:] != [0, 0, 0] for row in rows):
        raise ValueError('rank control vote mismatch')
    return all(row[4] == 1 for row in rows)


def fits_budget(available, allocated, needed, ceiling, host_floor):
    return ceiling - allocated >= needed and available - host_floor >= needed


def agree_and_budget(decoder, what, plan, needed, error_class):
    import torch
    engine = decoder.e
    fits = not hasattr(engine, 'memory_ceiling_bytes')
    if not fits:
        try:
            with open('/proc/meminfo') as file:
                available = next(int(s.split()[1])*1024 for s in file if s.startswith('MemAvailable:'))
            from .memory_budget import refresh_limit
            refresh_limit(engine, torch.cuda, available)
            allocated = torch.cuda.memory_allocated()
            decoder._memory_snapshot = dict(available=available, allocated=allocated,
                reserved=torch.cuda.memory_reserved(), needed=needed,
                ceiling=engine.memory_ceiling_bytes, host_floor=engine.memory_available_floor_bytes)
            fits = fits_budget(available, allocated, needed,
                              engine.memory_ceiling_bytes, engine.memory_available_floor_bytes)
        except (OSError, StopIteration, ValueError):
            fits = False
    mine = packet(plan, fits)
    rows = (decoder.m.comm.gather(torch.tensor(mine, dtype=torch.float32, device='cuda')).tolist()
            if engine.world > 1 else [mine])
    try:
        return consensus(rows, engine.world)
    except ValueError as error:
        raise error_class(f'the ranks planned different {what}s: {error}') from error
