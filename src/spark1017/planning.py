"""Budget anonymous table residency from measured host headroom after warm-up."""
from dataclasses import dataclass, asdict

GIB = 1 << 30

@dataclass(frozen=True)
class StoragePlan:
    available_before: int
    reserve: int
    quota: int
    resident_scale_bytes: int
    hot_row_budget: int

    def as_dict(self):
        return asdict(self)

def plan_storage(available: int, scale_bytes: int, *, quota: int = 4 * GIB,
                 reserve: int = 12 * GIB, hot_limit: int = GIB // 2) -> StoragePlan:
    if any(type(n) is not int or n < 0 for n in (available, scale_bytes, quota, reserve, hot_limit)):
        raise ValueError("storage budgets must be nonnegative integer bytes")
    budget = min(quota, max(0, available - reserve))
    # A partial scale table loses predictable coverage. Keep complete local hash
    # partitions, or use only the bounded demand cache when they do not fit.
    resident = scale_bytes if scale_bytes <= budget else 0
    hot = min(hot_limit, budget - resident)
    return StoragePlan(available, reserve, quota, resident, hot)

def mem_available(path="/proc/meminfo") -> int:
    with open(path) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) != 3 or fields[2] != "kB":
                    raise ValueError("unexpected MemAvailable units")
                return int(fields[1]) * 1024
    raise RuntimeError("MemAvailable is required for a unified-memory budget")
