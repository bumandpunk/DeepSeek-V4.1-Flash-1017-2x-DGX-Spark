# Changelog

## v0.2.0-alpha.1 — 2026-10-29

Known issue: the final cold deployment failed at its second repeated 128K prefill after 55 completed requests. Full SciCode/CanItEdit was not reached. Original services were restored; this version is a source alpha.

- Bound two-rank Engram gather storage with deterministic row tiles and direct FP32-sum/BF16 output slices. Add a CUDA bitwise, tile-boundary, ownership and peak-allocation gate. Retain the rejected whole-gather candidate; other rank counts use the upstream reduction.

- Add an arm64 serving image built from the verified official NGC base, a pinned public TensorFold commit and published patches. Constrain the NGC numerical stack, bound retries for public-source download interruptions, and record the installed packages and source hashes.
- Support a per-build HTTPS package mirror and verify downloaded wheels against pinned official PyPI hashes before accepting the image.
- Add a two-node controller with preflight checks, empty-cache extension compilation, authenticated readiness, explicit baseline/edit/experimental-prefill profiles, and matching image checks across ranks.
- Add private original-deployment snapshots, image retention, independent timed recovery, startup-failure recovery, original-file/spec/API verification, and host memory guards.
- Fix original-service verification to accept Docker environment reordering while still rejecting changed values and duplicate-key ordering changes; cover both cases with lifecycle regression tests.
- Give precompilation containers explicit ownership, clean them after timeouts and during stop/restore, and check writable-path aliases before compiling.
- Align engine cache admission with the external host-memory guard, keeping a 0.5 GiB margin after a real 32K guard trip exposed conflicting thresholds. Preserve the failed run and recovery evidence.
- Set the release default to a 256K shared context window after a 512K cold start could not admit a minimal request safely. Keep the 128K input plus 512-token regression unchanged and record the configured window in benchmark metadata.
- Refresh the CUDA allocator limit at the existing collective memory boundaries using current host headroom, instead of retaining a stale boot-time cap. Keep the host floor and eviction/admission checks unchanged; add tests for growth, pressure, graph-private slack and physical bounds.
- Add CPU and lifecycle failure tests, a Python 3.11/3.12 CI matrix, release-evidence recomputation, and a manual source-archive workflow. No workflow publishes a GitHub release automatically.
- Add English/Chinese quick starts, operating and recovery documentation, contribution guidance, issue/PR templates, and this developer-preview release. Validation evidence is recorded separately from historical performance claims.

## 0.2.0 source snapshot (unreleased) — 2026-10-28

- Add a bounded, two-segment RDMA receive pipeline with per-segment GPU sequence tickets and an HC residual-update kernel. Preserve upstream arithmetic order and BF16 rounding boundaries.
- Integrate prompt/decode switches and preserve the decode HC side-stream schedule. Verify source and feature agreement before weight loading and graph capture.
- Add independently disabled control-fusion, expert work-list, persistent-queue, fragment-chain and remote-DSpark experiments; retain negative results.
- Add paired ordinary-generation, concurrency and long-prefix runs, a return-baseline check, a same-A0 native-engine comparison and frozen programming regressions. See the recorded result tables for gains and non-gains.
- Preserve the exact measured E320 source snapshot; disclose the release's attribution-comment and package-version differences. Correct current TensorFold/Mia licensing attribution while retaining historical third-party notices.
- Add source prebuilding, result recomputation and artifact-consistency tools. That snapshot used an editable source checkout; it did not verify a standalone wheel or newly built serving image.

The selected second-round route keeps first-round lookup, adaptive depth and storage disabled. Combining those opt-in routes has not been revalidated in this round. The new transport is experimental; fault-injected RDMA recovery and a broader long-context coding suite remain unverified.

## 0.1.0 — 2026-10-27

Repeated-code lookup proposals, exact Engram storage experiments and deterministic draft scheduling. The first-round gains concern source-preserving editing tasks; those historical records remain under `results/2026-10-27/`.
