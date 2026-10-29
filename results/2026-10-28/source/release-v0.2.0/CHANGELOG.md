# Changelog

## 0.2.0 — 2026-10-28

- Add a bounded, two-segment RDMA receive pipeline with per-segment GPU sequence tickets and an HC residual-update kernel. Preserve upstream arithmetic order and BF16 rounding boundaries.
- Integrate prompt/decode switches and preserve the decode HC side-stream schedule. Verify source and feature agreement before weight loading and graph capture.
- Add independently disabled control-fusion, expert work-list, persistent-queue, fragment-chain and remote-DSpark experiments; retain negative results.
- Add paired ordinary-generation, concurrency and long-prefix runs, a return-baseline check, a same-A0 native-engine comparison and frozen programming regressions. See the recorded result tables for gains and non-gains.
- Preserve the exact measured E320 source snapshot; disclose the release's attribution-comment and package-version differences. Correct current TensorFold/Mia licensing attribution while retaining historical third-party notices.
- Add source prebuilding, result recomputation and artifact-consistency tools. The release uses an editable source checkout; no standalone wheel or newly built serving image is claimed as verified.

The selected second-round route keeps first-round lookup, adaptive depth and storage disabled. Combining those opt-in routes has not been revalidated in this round. The new transport is experimental; fault-injected RDMA recovery and a broader long-context coding suite remain unverified.

## 0.1.0 — 2026-10-27

Repeated-code lookup proposals, exact Engram storage experiments and deterministic draft scheduling. The first-round gains concern source-preserving editing tasks; those historical records remain under `results/2026-10-27/`.
