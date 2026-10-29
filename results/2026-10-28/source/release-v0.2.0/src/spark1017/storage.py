"""TensorFold adapter: run after warm-up, before requests or scheduler admission."""
import json
import os
from .planning import GIB, mem_available, plan_storage

def initialize(engine):
    from .configuration import verify_peers
    if not getattr(engine, '_spark1017_verified', False):
        verify_peers(engine)
    if os.environ.get("SPARK1017_STORAGE", "0") != "1":
        return
    eng = engine.model.engram
    if eng is None or eng.dfiles is None:
        raise RuntimeError("Spark1017 storage requires initialized local Engram AIO")
    ranges = []
    col_lo, col_hi = eng.cols
    for li, layer in enumerate(eng.cfg.engram_layers):
        name = f"layers.{layer}.engram.embed.scale"
        fd, base, row_bytes = eng.files[name]
        direct_fd = eng.dfiles[name][0]
        lo = int(eng.np_offsets[li, col_lo])
        last = col_hi - 1
        hi = int(eng.np_offsets[li, last] + eng.np_primes[li, last])
        assert 0 <= lo < hi <= eng.cfg.engram_rows[li]
        ranges.append((fd, base + lo * row_bytes, (hi - lo) * row_bytes, direct_fd))
    scales = sum(r[2] for r in ranges)
    use_scales = os.environ.get("SPARK1017_RESIDENT_SCALES", "1") == "1"
    plan = plan_storage(mem_available(), scales if use_scales else 0,
                        quota=int(float(os.environ.get("SPARK1017_STORAGE_GIB", "4")) * GIB),
                        reserve=max(12 * GIB, int(float(os.environ.get("SPARK1017_RESERVE_GIB", "12")) * GIB)),
                        hot_limit=int(float(os.environ.get("SPARK1017_HOT_MIB", "512")) * (1 << 20)))
    if plan.resident_scale_bytes:
        for fd, offset, size, direct_fd in ranges:
            eng.io.cache_resident_add(fd, offset, size, direct_fd)
            # Only the pages just copied into anonymous residency. Weight pages
            # and other tasks' files are not subject to a global drop_caches.
            os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
    allocated = eng.io.cache_configure(plan.hot_row_budget,
                        os.environ.get("SPARK1017_COALESCE", "1") == "1")
    after = mem_available()
    if after < plan.reserve:
        raise RuntimeError("concurrent host allocation consumed the reserved workspace; refusing to serve")
    record = {"event": "storage_plan", "rank": engine.rank, **plan.as_dict(),
              "hot_allocated": allocated, "available_after": after, "stats": eng.io.cache_stats()}
    print("[spark1017] " + json.dumps(record, sort_keys=True), flush=True)
    # CUDA and host allocations share physical memory; recalculate the existing
    # allocator ceiling after reserving these host bytes.
    engine._memory_ceiling()
