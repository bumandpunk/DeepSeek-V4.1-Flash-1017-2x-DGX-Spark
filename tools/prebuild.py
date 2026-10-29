"""Compile installed runtime sources before loading any model weights.

Run inside the same prepared image, source mounts and cache mounts as serving,
on each idle rank. This command does not stop or inspect other applications.
"""
import os
os.environ.setdefault('MAX_JOBS','1')
from tensorfold.cuda import rdma
from tensorfold.cuda.exl3 import linear,experts,prefetch
from tensorfold.families.deepseek_v41.cuda.model import _engram_io
from spark1017 import transport
for module in (rdma,linear,experts,prefetch,transport):
    module._ext()
_engram_io()
print('Required runtime extensions built; model weights were not loaded.')
