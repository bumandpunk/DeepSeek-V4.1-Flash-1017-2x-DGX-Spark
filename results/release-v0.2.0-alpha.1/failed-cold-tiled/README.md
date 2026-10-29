# Cold tiled-image long-input failure

The final image built from public source and empty native/rank caches started
successfully, passed API checks and completed 55 of the planned 58 fixed-suite
requests. The 56th recorded request, the second repeated 128K prefill, failed.

The exception was a 64 MiB allocation in the upstream FP4 quantization path
(`fp4_qd` → `_to_e2m1` → `torch.where`). It exceeded the configured CUDA
allocator cap. The external host-memory guard did not trigger. The successful
prefix had identical request/output/token-count records to the earlier complete
baseline. This incomplete run is not a passing speed or quality result.

Engram's tiled reduction had passed nine exactness/workspace checks on both
GB10s and the separate warm-cache three-profile diagnostic. Those results do
not establish repeated 128K cold-deployment stability. Full SciCode and
CanItEdit generation was not reached in this run.

Original-service restoration is recorded separately. The v0.2.0-alpha.1 source
preview retains this as a known issue. A completed clean-deployment quality
certificate and validated release bundle remain unavailable for this run.

[Failure summary](failure.json) · [Build](build.json) · [Timeline](timeline.json) · [Requests](bench/baseline-speed/requests.jsonl)
