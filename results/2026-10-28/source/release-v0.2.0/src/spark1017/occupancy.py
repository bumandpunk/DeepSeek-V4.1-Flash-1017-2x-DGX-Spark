"""Device-only expert occupancy partitioning for exact EXL3 prompt kernels.

Each expert belongs to one row-count bucket. Lists refer to expert *places*,
not IDs; row ordering and all per-row arithmetic remain upstream's.
"""
from __future__ import annotations

import torch
import triton as tr
import triton.language as tl

SENTINEL = 0x7fffffff


@tr.jit
def _prefix(counts, ids, used, ends, U: tl.constexpr, B: tl.constexpr,
            ROWS: tl.constexpr, MMA: tl.constexpr):
    bucket = tl.program_id(0)
    u = tl.arange(0, B)
    n = tl.load(used)
    eid = tl.load(ids + u, (u < U) & (u < n), 0)
    rows = tl.load(counts + eid, (u < U) & (u < n), 0)
    lo = tl.where(bucket == 0, 0, tl.where(bucket == 1, 16, 32))
    hi = tl.where(bucket == 0, 16, tl.where(bucket == 1, 32, ROWS))
    tile = tl.where(bucket == 0, 16, tl.where((bucket == 2) & MMA, 64, 32))
    groups = tl.where((rows > lo) & (rows <= hi), tl.cdiv(rows, tile), 0)
    tl.store(ends + bucket * B + u, tl.cumsum(groups, 0))


@tr.jit
def _emit(ends, work, U: tl.constexpr, B: tl.constexpr,
          CAP: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    bucket = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(ends + bucket * B + U - 1)
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), U, tl.int32)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        end = tl.load(ends + bucket * B + mid, mid < U, 0x7fffffff)
        after = end <= j
        lo = tl.where(after, mid + 1, lo)
        hi = tl.where(after, hi, mid)
    before = tl.load(ends + bucket * B + lo - 1, (lo > 0) & (lo <= U), 0)
    tl.store(work + (bucket * CAP + j) * 2,
             tl.where(j < total, lo, 0x7fffffff), j < CAP)
    tl.store(work + (bucket * CAP + j) * 2 + 1,
             tl.where(j < total, j - before, 0), j < CAP)


def lists(counts, ids, used, rows: int, slots: int, *, mma: bool):
    """Partition 1..16 / 17..32 / >32 members, with no device-to-host reads.

    This returns GU lists with tile sizes 16, 32, 64 (MMA) or 32;
    down lists reuse 16 and 32 and use a separate >32 list with tile 32.
    Bounded sentinel tails make CUDA graph-compatible launch sizes explicit.
    """
    u = ids.numel()
    b = tr.next_power_of_2(u)
    # Sum ceil(count/tile) <= floor((P + U*(tile-1))/tile).
    sizes = [u, u, (rows * slots + u * ((64 if mma else 32)-1)) // (64 if mma else 32),
             (rows * slots + u * 31) // 32]
    cap = max(sizes)
    # Per-call buffers avoid unsafe sharing across concurrent request streams.
    ends = torch.empty((4, b), dtype=torch.int32, device=counts.device)
    work = torch.empty((4, cap, 2), dtype=torch.int32, device=counts.device)
    _prefix[(4,)](counts, ids, used, ends, u, b, rows, mma)
    _emit[(4, tr.cdiv(cap, 256))](ends, work, u, b, cap, b.bit_length(), 256)
    return [work[i, :n] for i, n in enumerate(sizes)]


def reference(counts: list[int], ids: list[int], used: int, rows: int, mma: bool):
    specs = [(0, 16, 16), (16, 32, 32), (32, rows, 64 if mma else 32), (32, rows, 32)]
    return [[(place, group) for place, eid in enumerate(ids[:used])
             if lo < counts[eid] <= hi
             for group in range((counts[eid] + tile - 1) // tile)]
            for lo, hi, tile in specs]
