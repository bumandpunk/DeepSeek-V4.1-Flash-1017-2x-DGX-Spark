"""Refresh the CUDA allocator cap at existing collective memory boundaries.

GB10 host RAM is shared with CUDA. A cap computed only at startup becomes stale
as initialization memory is reclaimed or host pressure changes. Count current
reserved CUDA bytes once, and allow growth only from MemAvailable above the
unchanged host floor. Graph-private slack is never counted as free host memory.
"""

QUANTUM = 32 * 2**20


def allocator_limit(available, reserved, host_floor, total, quantum=QUANTUM):
    values = (available, reserved, host_floor, total, quantum)
    if any(v < 0 for v in values) or not total or not quantum or reserved > total:
        raise ValueError('Invalid allocator budget observation')
    budget = min(int(total), int(reserved) + max(0, int(available) - int(host_floor)))
    # Round down to limit updates on decode rounds. Never revoke held blocks.
    return max(int(reserved), budget // int(quantum) * int(quantum))


def refresh_limit(engine, cuda, available):
    total = getattr(engine, '_spark1017_total_memory', None)
    if total is None:
        total = int(cuda.mem_get_info()[1])
        engine._spark1017_total_memory = total
    reserved = int(cuda.memory_reserved())
    limit = allocator_limit(available, reserved, engine.memory_available_floor_bytes, total)
    previous = engine.memory_ceiling_bytes
    if limit != previous:
        cuda.set_per_process_memory_fraction(limit / total)
        engine.memory_ceiling_bytes = limit
        if abs(limit - previous) >= 128 * 2**20:
            print(f'[spark1017] rank {engine.rank}: allocator cap {previous / 2**30:.3f} -> '
                  f'{limit / 2**30:.3f} GiB; host available {available / 2**30:.3f}, '
                  f'protected floor {engine.memory_available_floor_bytes / 2**30:.3f}', flush=True)
    return limit
