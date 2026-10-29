# Two-rank Engram workspace diagnostic

A layered diagnostic image combined guarded allocator refresh and the FP32
sum/BF16 output operation. It passed five health checks, three structured API
checks, the full 58-request speed suite and six complete-file generations with
836 functional assertions. The original deployment was restored afterwards.
The CUDA test on GB10 passed eight exactness/ownership/fallback cases and reduced
representative extra peak allocation from 600 MiB to 500 MiB.

This diagnostic reused the previous base application image and cache namespace.
It is excluded from final speed and quality counts. A new no-cache application
build with empty native/rank caches is required for the final release gate.

A subsequent fresh build passed the full baseline but failed in the edit profile
while allocating the full 400 MiB gather; see [the retained failure](../failed-full-gather/README.md).
This earlier single-profile diagnostic is not release-readiness proof.
