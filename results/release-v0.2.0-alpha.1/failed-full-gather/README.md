# Full-gather candidate rejected

This no-cache application build and fresh native/rank caches passed the complete
baseline: 58 speed requests, 836 full-file assertions, 14/24 CanItEdit generations
and 37/50 SciCode steps (9/15 parents). All 50 SciCode requests, complete messages,
code and verdicts matched the earlier A0 quality reference. Startup-failure
recovery was also verified.

The edit profile then failed on its first 8192-token request after eight API
checks and 48 ordinary/concurrent requests. Rank 0 could not allocate a 400 MiB
FP32 gather buffer under its allocator cap; rank 1 subsequently lost the
collective. The external memory guard did not trip. The original deployment was
restored and its five API checks passed.

The direct two-rank sum/BF16 output saved 100 MiB at the representative size, but
still required a whole 400 MiB gather. Its successful baseline and CUDA test did
not establish stability for all profiles. This candidate is rejected and its
measurements are excluded from final release comparison counts. The next
implementation gathers deterministic row tiles into the same BF16 output,
without changing the FP32 projection, rank sum order or memory protection.
