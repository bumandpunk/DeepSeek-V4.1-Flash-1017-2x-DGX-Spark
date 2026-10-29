# Known issues

## Repeated 128K input after a cold deployment

The clean application image starts with empty native and rank caches and passes
API checks. In the recorded sequence, ordinary generation, concurrency, all 8K
and 32K requests, and the first 128K request completed. The second 128K prefill
failed while allocating a 64 MiB temporary tensor in FP4 quantization.

The CUDA allocator cap rejected the allocation; the external host-memory guard
did not trigger. Original-service restoration passed afterwards.

A separate run with existing rank caches passed all three profiles with the
same installed runtime source hashes and Python versions. Cache state and
available workspace still need investigation. Repeated 128K cold-deployment
stability is unresolved in v0.2.0-alpha.1.

[Failure and raw records](../results/release-v0.2.0-alpha.1/failed-cold-tiled/README.md)
· [Runtime equivalence](../results/release-v0.2.0-alpha.1/tiled-workspace-probe/runtime-equivalence.json)
· [Restoration](../results/release-v0.2.0-alpha.1/restoration.json)

The full SciCode/CanItEdit pass was not reached in this cold run. Earlier quality
results and the successful six-generation edit diagnostic remain separate.
