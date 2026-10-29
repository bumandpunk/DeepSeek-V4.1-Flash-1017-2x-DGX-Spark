# Spark1017

<!-- validation-status:start -->
**Source preview:** the final clean-image programming regression and end-of-run restoration are running. [Completed checks and pending work](results/release-v0.2.0-alpha.1/README.md). CPU CI does not certify the unfinished GPU run.
<!-- validation-status:end -->

**Source-preserving code-edit acceleration and inference experiments for two NVIDIA DGX Spark / GB10 hosts.**

`v0.2.0-alpha.1` · Apache-2.0 · Developer preview · [简体中文](README.zh-CN.md)

Spark1017 extends the pinned [TensorFold](https://github.com/bertholomus/TensorFold) engine for DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw. It provides repeated-code proposals, experimental segmented RDMA/HC fusion, and reproducible comparisons. Model execution, EXL3 kernels, tensor parallelism and DSpark are upstream work. This project changes inference; it has not trained a new foundation model.

[Quick start](#quick-start) · [Results](#measured-results) · [Operations and recovery](docs/operations.md) · [Design](docs/design.md) · [Contributing](CONTRIBUTING.md) · [Release notes](docs/releases/v0.2.0-alpha.1.md)

## At a glance

- **Faster source-preserving edits in the measured cases.** Propose repeated code from committed context and let the target model verify it in batches. On three full-file editing fixtures, decoding was 32–58% faster than the measured Mia native engine, with 23–34% less complete-request time and identical outputs.
- **An independent transport experiment.** A bounded two-segment receive pipeline feeds HC residual updates directly, avoiding a gathered intermediate. It improved 128K prompt processing modestly against our Python baseline; Mia remained faster in the matched ordinary-generation/long-input suite.
- **Bounded Engram workspace.** Deterministic row-tiled gathers write the two-rank sum directly into BF16 output slices. At the tested prefill size, extra peak allocation fell from 600 MiB to 132 MiB on both GB10s, with bit-exact checks. This is a workspace saving, not a standalone speed claim.
- **Explicit deployment profiles.** Baseline, edit acceleration, and experimental transport are separate. Unvalidated combinations and unsuccessful experiments stay off.
- **Reproducible evidence.** Pinned source/weight references, complete responses, paired quality checks, failures and original measured source snapshots are included.

The edit benefit is strongest when the output retains substantial existing code. Short patches, entirely new code and concurrent editing may benefit less. There is no claim of universal speed leadership, improved coding capability, or full-model quality preservation.

## Measured results

Same two GB10s, same A0 weight bytes, same requests and output budgets. The editing rows use lookup; the ordinary/long-input rows use the separate transport profile. They are not one combined configuration.

This table is the historical 2026-10-27/28 comparison with a 512K shared window on both sides. The release validation uses a 256K window and reports its baseline/edit/transport results separately; Mia was not rerun at that capacity.

| Workload | Mia native | Spark1017 | Observation |
|---|---:|---:|---|
| Full-file `bisect` edit | 112.35 tok/s | 148.41 tok/s | Decode +32.1%; request time −23.1% |
| Full-file `graphlib` edit | 120.06 tok/s | 189.18 tok/s | Decode +57.6%; request time −28.9% |
| Full-file `textwrap` edit | 120.06 tok/s | 188.35 tok/s | Decode +56.9%; request time −34.1% |
| Ordinary short code | 96.32 tok/s | 95.08 tok/s | Mia faster |
| 128K prompt processing | 2341.60 input tok/s | 2047.74 input tok/s | Mia faster |

Editing: two repetitions per fixture, identical complete responses, 836 functional assertions per engine. Ordinary generation: nine repetitions per prompt; long inputs: three cold-prefix repetitions. These are small same-day sequential experiments, not a broad coding leaderboard or a statistically established cross-day speed guarantee.

Frozen E320 regressions retained **37/50 SciCode steps** and **7/12 CanItEdit cases in each repetition**, with paired code and verdicts unchanged. The SciCode reference control itself passed 48/50; all failures stay in the denominator. These are development regressions, not Artificial Analysis official scores.

[Editing results and raw evidence](results/2026-10-27/README.md) · [Ordinary generation and long input](results/2026-10-28/README.md) · [Comparison protocol](benchmarks/competitors.md)

## Requirements

- Two DGX Spark / GB10 hosts with about 128 GB unified memory each, Docker and NVIDIA GPU access. The recorded environment exposes 121.69 GiB per node.
- Working two-rail ConnectX-7/RoCE connectivity and key-based SSH from head to worker. Configure the fabric first; see NVIDIA's [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks) guide. The scripts inspect existing networking and do not reconfigure it.
- The compatible EXL3 checkpoint and Engram tables readable **locally on both nodes**. Source revisions are in [checkpoints/sources.json](checkpoints/sources.json). Weights are not distributed here. Give the cache at least 130 GiB of additional free space per node, plus room for the container image and logs; model and Engram storage is separate.
- Enough available memory to load the model. The example requires at least 110 GiB before startup and keeps a 2.5 GiB runtime host-memory guard. Stop other GPU inference first.
- The release default is a 256K shared token window across two slots. Its long-input regression reaches 128K plus 512 generated tokens. A 512K cold-start attempt failed safe admission on this pair; it is not a validated release configuration.
- Python 3.11+ on both hosts, `ssh`, Docker, and a working systemd user manager for timed recovery. A 4090 is optional: in our experiments its host runs measurement, Linux CI and isolated graders, not target-model inference.

## Quick start

Run these commands from a source checkout on the head. The image build fetches the pinned public TensorFold commit; no existing private engine installation is needed.

**1. Configure the pair.**

```bash
cp config/cluster.example.json config/cluster.local.json
```

Fill in the actual addresses, RDMA devices/GID, model/Engram paths, and separate writable cache/state paths on each host. Create those writable directories as your deployment user. Generate an API key outside the repository and install the same key on both hosts with mode 0600. The API defaults to loopback and requires the key.

**2. Check, build and precompile.**

```bash
python3 scripts/cluster.py doctor
python3 scripts/build.py --no-cache
python3 scripts/cluster.py prebuild
```

The build verifies the official arm64 NGC image ID, applies the published patches to the fixed TensorFold commit, preserves NGC's torch stack, and distributes the built image to the worker. Precompilation uses idle GPUs, one compiler job and an 8 GiB container memory limit. A completely fresh cache is supported.

**3. Start and verify.**

```bash
./start.sh --profile edit
./verify.sh
./status.sh
```

The API is `http://127.0.0.1:19890/v1`, model `DeepSeek-V4.1-Flash-1017`, unless changed in your configuration. Startup waits for authenticated readiness and five real API checks. Each start gets an empty prompt-spill namespace.

An automatic rollback deadline is armed before startup. After inspection, use `python3 scripts/cluster.py confirm` only if you intentionally want to keep the experimental deployment running beyond that deadline. Memory protection remains enabled.

**Taking over an existing service:** configure `previous_service`, `previous_health`, `previous_paths` and `previous_containers`, then use `./start.sh --profile edit --takeover`. It snapshots first, arms recovery, and stops only that configured service. Full details are in [operations](docs/operations.md).

For a package mirror, pass `--package-index https://pypi.tuna.tsinghua.edu.cn/simple` to `scripts/build.py`. Downloaded wheels must still match the pinned official PyPI SHA-256 values. The option is scoped to this build.

## Everyday use

| Action | Command |
|---|---|
| Inspect both ranks | `./status.sh` |
| Print both Docker commands without starting containers | `python3 scripts/cluster.py plan --profile edit` |
| Verify the authenticated API | `./verify.sh` |
| Stop only this deployment | `./stop.sh` |
| Restore and verify the previous service | `python3 scripts/cluster.py restore` |
| Keep the experiment past its recovery deadline | `python3 scripts/cluster.py confirm` |
| Run CPU and artifact CI checks | `bash scripts/ci.sh` |

| Profile | Lookup | RDMA/HC fusion |
|---|---|---|
| `baseline` | off | off |
| `edit` | on | off |
| `prefill-experimental` | off | on |

Stop before switching profiles. Lookup falls back to DSpark for multiple live streams. All other experimental routes are disabled. [Advanced switches](docs/runtime.md) are documented for experiments; they are not supported combinations by default.

## What is original here

The bounded repeated-context index and proposal arbitration, Engram cache/controller extensions and row-tiled reduction integration, segmented receive/HC integration, invariants, measurements and lifecycle tooling are this project's contributions. Upstream numerical kernels, registration/proxy/completion transport and DSpark are reused and attributed. Speculative decoding, tiling and communication/computation fusion have prior work; no claim of inventing those concepts is made.

[First-round design](docs/design.md) · [Transport design and negative results](docs/round2-design.md) · [Engram workspace](docs/engram-reduction.md) · [Third-party attribution](NOTICE)

## Validation and limitations

Historical measured sources are archived beside their results. The release adds a clean-build and lifecycle validation pass; its final evidence and scope are linked from [release notes](docs/releases/v0.2.0-alpha.1.md). The public CI runs CPU, failure-path and recorded-artifact checks; it does not imply GPU/RDMA validation on every pull request.

The retained encoder-prefill/bounded-decoder-replay behavior is upstream behavior, not full exact decoder prefill. No 512K real-code workload or RDMA fault-injection guarantee is claimed. Lookup plus transport fusion has not been validated together. All new routes are experimental; a standalone wheel and production uptime guarantee are not provided.

The optional E320 export prunes 384 routed experts to 320. Retained tensor bytes remain exact, but pruning changes the model function. It is not lossless model compression. See [checkpoint transform](checkpoints/README.md).

## Related work and license

[TensorFold](https://github.com/bertholomus/TensorFold) supplies the engine foundation. [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks) is a measured native-engine reference. [Elevatormusic](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark), [lucataco](https://github.com/lucataco/DeepSeek-V4.1-Flash-2x-DGX-Spark), and [sudoingX](https://github.com/sudoingX/dgx-spark-deepseek-v41) are related deployment approaches. The README's status/quick-start/operations organization is informed by these public recipes; their performance claims are not substituted for our measurements.

New code is Apache-2.0. Original licenses and notices are preserved in `LICENSES/` and benchmark directories. Model weights, private configuration, credentials and compiled rank caches are not part of this repository. Contributions are welcome through [the contribution guide](CONTRIBUTING.md).
