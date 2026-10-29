# Tiled Engram workspace diagnostic

A layered image with deterministic Engram row-tiled gathers passed all three
profiles: five API health checks, three structured-output checks and 58 speed
requests per profile. All 174 request/output/token-count records matched the
complete 58-request pre-tiling baseline. The edit profile additionally passed
six full-file generations with 836 functional assertions. Original service
restoration was verified afterwards.

The GB10 microtest passed nine bit-exactness, tile-boundary, ownership and
fallback cases. Extra peak allocation at the representative size was 132 MiB
rather than the upstream operation's 600 MiB. Gather payloads were below 32 MiB
at the supported projection width.

This layered image/cache diagnostic is excluded from final benchmark counts.
The final release still requires a new no-cache application build, empty native
and rank caches, both-node CUDA checks, complete paired quality runs and recovery.
