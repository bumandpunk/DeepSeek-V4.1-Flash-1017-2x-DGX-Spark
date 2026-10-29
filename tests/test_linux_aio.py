"""Integration test on Linux/PyTorch, CPU only. No model weights required.

Exercise the actual extension against independent os.pread results: unaligned
table bases, duplicate rows, simultaneous batches, eviction, and short reads.
"""
import json
import os
import pathlib
import tempfile
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.cpp_extension import load

root = pathlib.Path(__file__).resolve().parents[1]
io = load(name='spark1017_engram_io_v1', sources=[str(root/'native/engram_io.cpp')],
          extra_cflags=['-O3'], verbose=False)
assert io.aio_init(1024)
torch.set_num_threads(1)
with tempfile.TemporaryDirectory() as tmp:
    file = pathlib.Path(tmp)/'immutable-tables.bin'
    generator = torch.Generator().manual_seed(1017)
    file.write_bytes(torch.randint(0, 256, (8 << 20,), dtype=torch.uint8, generator=generator).numpy().tobytes())
    fd = os.open(file, os.O_RDONLY); direct = os.open(file, os.O_RDONLY | os.O_DIRECT)
    wbase, sbase, rw, rs = 73, (6 << 20) + 11, 256, 8
    io.cache_resident_add(fd, sbase, 20000 * rs, direct)
    io.cache_configure(16384, True)  # deliberately tiny: eviction during tests

    def request(ids, deferred=False):
        idx = torch.tensor(ids, dtype=torch.int64)
        w = torch.empty((len(ids), rw), dtype=torch.uint8)
        s = torch.empty((len(ids), rs), dtype=torch.uint8)
        bid = io.aio_start(direct,wbase,rw,direct,sbase,rs,idx,w,s,1)
        def finish():
            io.aio_wait(bid)
            wantw=b''.join(os.pread(fd,rw,wbase+i*rw) for i in ids)
            wants=b''.join(os.pread(fd,rs,sbase+i*rs) for i in ids)
            assert w.numpy().tobytes()==wantw
            assert s.numpy().tobytes()==wants
        if deferred:return finish
        finish()

    request([0, 1, 15, 16, 17, 0, 15, 100, 4097])
    request([0, 1, 15, 16, 17, 0, 15, 100, 4097])
    for batch in range(50):
        ids=torch.randint(0,16000,(100,),generator=generator).tolist()
        one=request(ids+ids[:25],True)
        two=request(list(reversed(ids)),True)
        io.aio_touch(direct,wbase,rw,direct,sbase,rs,torch.tensor(ids[:1]))
        two();one()
    # The prompt and direct-I/O descriptors share resident bytes without fd aliasing.
    def prompt(lane):
        ids=[0,1,2,17,4095,4096,4097,19999]
        idx=torch.tensor(ids);w=torch.empty((len(ids),rw),dtype=torch.uint8);s=torch.empty((len(ids),rs),dtype=torch.uint8)
        for _ in range(20):
            io.gather_rows2_at(lane,fd,wbase,rw,fd,sbase,rs,idx,w,s,8)
            assert w.numpy().tobytes()==b''.join(os.pread(fd,rw,wbase+i*rw) for i in ids)
            assert s.numpy().tobytes()==b''.join(os.pread(fd,rs,sbase+i*rs) for i in ids)
    with ThreadPoolExecutor(2) as pool:list(pool.map(prompt,[0,1]))
    bad=torch.tensor([50000]);w=torch.empty((1,rw),dtype=torch.uint8);s=torch.empty((1,rs),dtype=torch.uint8)
    failed=False
    try:io.aio_wait(io.aio_start(direct,wbase,rw,direct,sbase,rs,bad,w,s,1))
    except RuntimeError:failed=True
    assert failed,'failed disk reads must not enter the cache'
    request([0,1,2])  # an errored batch must not poison later reads
    stats=io.cache_stats()
    assert stats['resident_hits']>0 and stats['hot_hits']>0 and stats['coalesced']>0
    assert stats['disk_reads_submitted']<stats['logical_reads']
    print(json.dumps({'status':'passed','stats':stats}),flush=True)
    # Descriptors stay alive until all asynchronous touch submissions are reaped.
