# Spark1017

Run DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw on two DGX Spark machines, with code-edit acceleration and two-node deployment scripts.

`v0.2.0-alpha.1` · [简体中文](README.zh-CN.md) · [Deployment](docs/operations.md) · [Results](results/release-v0.2.0-alpha.1/README.md)

<!-- validation-status:start -->
Alpha: repeated 128K input after a cold deployment has a known allocation failure. [Details and test results](docs/known-issues.md)
<!-- validation-status:end -->

Built on [TensorFold](https://github.com/bertholomus/TensorFold), with four additions:

- **Code-edit acceleration:** reuse code from the context as proposals for batched target-model verification.
- **Tiled Engram reduction:** reduce temporary prefill buffers. At the tested size, extra peak allocation fell from 600 MiB to 132 MiB, with bit-exact checks passing on both GB10s.
- **Experimental communication fusion:** feed segmented RDMA receives into HC residual updates.
- **Two-node management:** build, precompile, start, inspect, stop and restore the previous service.

## Requirements

- Two DGX Spark / GB10 hosts connected through two ConnectX-7/RoCE rails. Configure networking with NVIDIA's [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks) guide.
- Docker, NVIDIA container support and Python 3.11+ on both hosts; key-based SSH from head to worker and a working systemd user manager.
- Local model and Engram files on each host. See [checkpoint sources](checkpoints/README.md); weights are downloaded separately.
- At least 130 GiB of additional cache storage per node and 110 GiB of available memory before startup.

The default is a 256K shared token window across two slots. Long-input tests use 128K input tokens plus 512 output tokens. The 512K cold-start check did not pass.

An RTX 4090 is optional. We use its host for measurement, CI and isolated grading.

## Quick start

On the head node:

```bash
git clone https://github.com/bumandpunk/DeepSeek-V4.1-Flash-1017-2x-DGX-Spark.git
cd DeepSeek-V4.1-Flash-1017-2x-DGX-Spark
cp config/cluster.example.json config/cluster.local.json
```

Edit `config/cluster.local.json`: set both node addresses, network devices/GID, model and Engram paths, and separate cache/state directories owned by the deployment user.

Store the API key outside the repository. Use the same key on both nodes, set its file mode to `0600`, and point each node's `key_file` at it.

```bash
python3 scripts/cluster.py doctor
python3 scripts/build.py --no-cache
python3 scripts/cluster.py prebuild
./start.sh --profile edit
./verify.sh
```

The build pins the upstream revision and distributes the same image to the worker. The first run creates native and rank caches; later starts can reuse them.

For a PyPI mirror, add `--package-index`. Downloads are still checked against pinned official SHA-256 values:

```bash
python3 scripts/build.py --no-cache --package-index https://pypi.tuna.tsinghua.edu.cn/simple
```

The default API is `http://127.0.0.1:19890/v1`, with model name `DeepSeek-V4.1-Flash-1017`. Requests require the API key.

Startup arms a two-hour recovery timer by default. To keep the deployment running after inspection:

```bash
python3 scripts/cluster.py confirm
```

## Profiles and operations

| Profile | Use |
|---|---|
| `baseline` | Disable code proposals and communication fusion for comparison |
| `edit` | Enable repeated-code proposals for edits that retain much of the original file |
| `prefill-experimental` | Enable RDMA/HC fusion separately for long-input experiments |

Run `./stop.sh` before switching profiles. Code proposals and communication fusion are not enabled together.

| Action | Command |
|---|---|
| Inspect both nodes | `./status.sh` |
| Check the API | `./verify.sh` |
| Stop this deployment | `./stop.sh` |
| Restore the previous service | `python3 scripts/cluster.py restore` |

For an existing deployment, configure the previous service, health check, files and containers using the [operations guide](docs/operations.md), then run `./start.sh --profile edit --takeover`. Takeover saves a snapshot first; restoration checks the original files, container settings and API.

## Benchmarks

The measured speed gains are concentrated in full-file edits that retain substantial existing code. Ordinary generation and long-input processing do not yet show a consistent speed advantage.

- [Full-file edits, CanItEdit and SciCode](results/2026-10-27/README.md)
- [Ordinary generation, concurrency and long inputs](results/2026-10-28/README.md)
- [This version's deployment and regression records](results/release-v0.2.0-alpha.1/README.md)
- [Comparison projects and test conditions](benchmarks/competitors.md)

These records include configurations, raw responses, functional checks and failed runs. Run the CPU checks when changing the code:

```bash
bash scripts/ci.sh
```

## Development and license

[Code proposals](docs/design.md) · [Communication fusion](docs/round2-design.md) · [Engram workspace](docs/engram-reduction.md) · [Contributing](CONTRIBUTING.md)

New code is Apache-2.0. TensorFold supplies model execution, EXL3 kernels, tensor parallelism and DSpark. Third-party code and dataset licenses are listed in [NOTICE](NOTICE) and [LICENSES](LICENSES/).
