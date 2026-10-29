"""Build separate expert overlay from the pinned TensorFold revision."""
import argparse
import hashlib
import json
from pathlib import Path


def replace(s, old, new):
    assert s.count(old) == 1, 'unsupported source anchor: ' + old[:80]
    return s.replace(old, new, 1)


def build(source, output):
    s = source.read_text()
    original = hashlib.sha256(s.encode()).hexdigest()
    s = replace(s, '    chunk = group and R >= EXACT_ROWS',
                '    chunk = group and (R >= EXACT_ROWS or os.environ.get("SPARK1017_CHAIN_DECODE", "0") == "1")')
    s = replace(s, '    if (mode == "fused" and group', '    if (mode in ("fused", "persistent") and group')
    s = replace(s, '        return _routed_fused(ext, x, pick, wts, ex, s, out, R, ids, members, limit, act_mode, add, before_down)', '''        if ((mode == "persistent" or os.environ.get("SPARK1017_PERSISTENT", "0") == "1")
                and ex.cb == 2 and max(ex.k2_gu[1], ex.k2_d[1]) <= 10 and before_down is None):
            from spark1017.persistent import routed as persistent_routed
            return persistent_routed(ext, x, pick, wts, ex, s, out, R, ids, members, limit, act_mode, add)
        return _routed_fused(ext, x, pick, wts, ex, s, out, R, ids, members, limit, act_mode, add, before_down)''')
    s = replace(s, '    work_gu = work_d = s.no_work', '''    chain = (os.environ.get("SPARK1017_CHAIN", "0") == "1" and fast and GROUP_LIST
             and not (mma2 or mma3) and ex.cb == 2 and I % 128 == 0 and D % 128 == 0
             and max(ex.k2_gu[1], ex.k2_d[1]) <= 10)
    work_gu = work_d = s.no_work''')
    s = replace(s, '        rows_gu = 64 if mma else', '        rows_gu = 32 if chain else 64 if mma else')
    s = replace(s, '        rows_d = 64 if mma_d else', '        rows_d = 32 if chain else 64 if mma_d else')
    s = replace(s, '    elif mma:\n', '''    elif chain:
        from spark1017.chain import run as chain_run
        chain_run(s.xg,s.xu,ex.gate_ptr,ex.up_ptr,ex.gate_k2,ex.up_k2,ids,s.count,members,s.z,
                  2,D,I,P,sk,slots,ex.cb,w,ex.k2_gu[0],ex.k2_gu[1],1,work_gu)
    elif mma:
''')
    s = replace(s, '    elif mma_d:\n', '''    elif chain:
        from spark1017.chain import run as chain_run
        chain_run(s.xd,s.xd,ex.down_ptr,ex.down_ptr,ex.down_k2,ex.down_k2,ids,s.count,members,s.z,
                  1,I,D,P,sk,slots,ex.cb,w,ex.k2_d[0],ex.k2_d[1],int(sk>1),work_d)
        dsk=1
    elif mma_d:
''')
    s = replace(s, '    if getattr(s, "prompt", False):', '''    buckets = None
    if (os.environ.get("SPARK1017_OCCUPANCY", "0") == "1" and fast and GROUP_LIST
            and not (mma2 or mma3) and not mma_d and sk == 4 and w == 4
            and s.cfg_d[:3] == (8, 4, 1)):
        from spark1017.occupancy import lists as occupancy_lists
        buckets = occupancy_lists(s.counts, ids, s.count, R, slots, mma=mma)
    if getattr(s, "prompt", False):''')
    s = replace(s, '    elif mma:\n', '''    elif buckets is not None:
        for bi, (tile, work) in enumerate(zip((1, 2, 2), buckets[:3])):
            if bi == 2 and mma:
                ext.grouped_mma(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids,
                    s.count, members, s.z, 2, D, I, P, sk, slots, ex.cb, w,
                    ex.k2_gu[0], ex.k2_gu[1], 1 | _mma_steps() << 1, work)
            else:
                ext.grouped_rows(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids,
                    s.count, members, s.z, 2, D, I, P, sk, slots, ex.cb, 8, w, 1, tile,
                    ex.k2_gu[0], ex.k2_gu[1], 1, work)
    elif mma:
''')
    s = replace(s, '    elif mma_d:\n', '''    elif buckets is not None:
        for tile, work in zip((1, 2, 2), (buckets[0], buckets[1], buckets[3])):
            ext.grouped_rows(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids,
                s.count, members, s.z, 1, I, D, P, sk, slots, ex.cb, 8, w, 1, tile,
                ex.k2_d[0], ex.k2_d[1], 0, work)
    elif mma_d:
''')
    # Avoid building the uniform lists when the bucket route will replace them.
    s = replace(s, '    if fast and GROUP_LIST and not (mma2 or mma3):',
                '    if fast and GROUP_LIST and not (mma2 or mma3) and os.environ.get("SPARK1017_OCCUPANCY", "0") != "1":')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(s)
    print(json.dumps({'input_sha256': original, 'output_sha256': hashlib.sha256(s.encode()).hexdigest()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build(a.source, a.output)
