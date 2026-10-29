# Reproducing and extending the experiment

Use one hypothesis per comparison and keep its competing baseline. New modules start disabled. A successful optimization needs runnable code, a reproducible baseline, matched requests, quality evidence, and a clear description of workloads that do not benefit.

## Before changing an inference route

Record hardware, source revisions, image IDs, model and Engram hashes, context/slot limits, prefill chunk size, sampler and thinking settings, request hashes and spill namespace. Snapshot the existing deployment and verify its restoration path. Compile extensions with inference stopped. Keep the hardware memory protection enabled for both arms.

For a same-weight comparison, copy or reference identical checkpoint bytes. Different quantizations or expert selections are different experiments. A reported official score is a separate reference and cannot substitute for a matched local engine regression. Do not publish a league table that silently changes the input prefix, prompt cache state, output length or reasoning setting.

## What the current implementation establishes

- The lookup route makes longer proposals from earlier committed context and submits them to the existing target verifier. The exact suffix index is bounded per stream; the controller's learning state contains only acceptance counts, not another user's source text.
- The allocator and sampler remain upstream components. Verification shape changes can change finite-precision behavior and random-number consumption; run empirical regressions in addition to the algorithm tests.
- At multiple live streams, lookup falls back to DSpark. No claim of an optimized multi-stream lookup scheduler is made.
- The optional cache preserves exact bytes and requires immutable files/descriptors for the serving lifetime. Its tests cover concurrency, collisions, alias descriptors, duplicate fanout and failed short reads.

## Useful next experiments

1. Randomize baseline/candidate order over multiple sessions, using more repositories and edit types. Report complete task time and correctness, including edits with little repeated context.
2. Extend lookup to multiple streams with a shared verification-row budget. Preserve per-stream content isolation and deterministic rank decisions; compare against the existing concurrency baseline before enabling it.
3. Calibrate the choice between lookup and DSpark on a separate public calibration corpus. Keep evaluation tasks out of that calibration and freeze the resulting cost data before the final comparison.
4. Add a checkpoint-format adapter for the unmeasured vLLM implementation, then verify tensor identity and run the exact same request suite. Do not use a different weight pack to claim a same-weight engine victory.

These are future experiments, not implemented results. A new model with improved coding ability would additionally require training and held-out capability evaluation; this repository currently changes the inference system.

## Pull requests

Include the concrete problem, implementation, invariants affected, CPU/native test results, and a pointer to raw measurements. Label invalid measurements and preserve failures. For performance changes, include an ablation with only that feature enabled. Do not upload private endpoints, credentials, model weights, compiled rank caches or generated executables.
