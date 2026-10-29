# Runtime integration

For the supported two-host startup and recovery path, use [operations](operations.md). This page describes the lower-level experiments and historical measurement settings.

The prepared TensorFold source imports `spark1017` from the installed package. Apply identical switches to both ranks before starting either process. Do not modify code, cost files or source descriptors while serving. Existing production services and model files should be snapshotted, stopped for an experiment, and restored afterward.

This experimental release is a source checkout with an editable install (`pip install -e .`), as used by the Dockerfile. Keep `native/` beside `src/`: native loaders resolve generated sources relative to that checkout. A standalone, self-contained wheel is not supplied.

| Switch | Default | Behavior |
|---|---:|---|
| `SPARK1017_STORAGE` | 0 | Enable exact scale residency / hot rows / coalescing |
| `SPARK1017_RESIDENT_SCALES` | 1 | Admit complete local scale partitions if budget allows |
| `SPARK1017_STORAGE_GIB` | 4 | Maximum new host storage allocation budget |
| `SPARK1017_RESERVE_GIB` | 12 | Minimum host headroom at initialization, never below 12 GiB |
| `SPARK1017_HOT_MIB` | 512 | Upper hot-cache budget; actual set capacity rounds down |
| `SPARK1017_COALESCE` | 1 | Merge duplicate pending reads in one AIO batch |
| `SPARK1017_ADAPTIVE_DEPTH` | 0 | Enable calibrated joint depth allocation |
| `SPARK1017_COSTS` | empty | Optional identical frozen JSON file on both ranks |
| `SPARK1017_LOOKUP` | 0 | Enable repeated-code proposal / DSpark arbitration |
| `SPARK1017_RDMA_POST` | 0 | `prefill`, `decode`, or `all`: pipelined RDMA receive plus exact HC post |
| `SPARK1017_PREFILL_SIDE` | 0 | Overlap prompt HC coefficient work with attention/MoE; explicit join before post |
| `SPARK1017_CONTROL_FUSION` | 0 | Combine the existing plan agreement and memory vote |
| `SPARK1017_SEGMENTED_RDMA` | 0 | Experimental plain segmented gather; slower on large prompts, keep off |
| `SPARK1017_DRAFT_HELPER` | empty | Experimental `host:port` DSpark helper; slower on the measured LAN, keep off |
| `SPARK1017_PROFILE` | empty | Append round-cost calibration JSONL on rank 0 |
| `SPARK1017_KERNEL_TRACE` | empty | Trace exactly three rounds after 50 rounds, rank 0 only |

The default-off flags make every new route explicitly experimental. The historical first two rounds used TP=2, context=524288, two engine slots, five DSpark drafts and a 2048-token prefill chunk. Their E320 tests kept vision capability enabled; the separate A0 text-only/native comparison omitted `--vision` on both Python arms. The release profiles use a 262144-token shared window with vision off: a fresh 512K start did not leave enough safe request workspace on the measured hosts. The 128K long-input workload and target sampling remain unchanged. Register an API key and bind rank 0 to loopback behind authenticated ingress. Mount the checkpoint at `/model`, Engram files at `/engram`, and a separate experiment cache at `/cache`.

Inside a container with the verified CUDA/RDMA device mounts and environment, the engine command is:

```bash
tensorfold serve /model --tp 2 --rank "$RANK" --master "$HEAD_FABRIC_IP" \
  --host 127.0.0.1 --port 19890 --name DeepSeek-V4.1-Flash-1017 \
  --context 262144 --parallel 2 --mtp-drafts 5 \
  --api-key-file /run/secrets/api-key
```

Start rank 0 first, wait for its distributed store to listen, then start rank 1 with the same master address. The Dockerfile sets `tensorfold` as ENTRYPOINT, so a `docker run` command passes `serve ...` directly. Consult the pinned TensorFold deployment for fabric setup and extension warm-up. This command alone is not a complete host provisioning script. Both ranks fingerprint the new Python modules, feature switches and cost-file contents at startup and refuse inconsistent configurations; the separate overlay manifest identifies the patched engine files.

Pass `--api-key-file /run/secrets/api-key` with a read-only key-file mount. An environment variable alone does not enable authentication in the pinned engine. Configure `TF_DS_ENGRAM=/engram`, `TF_DS_RANK_CACHE=/cache/tf-ds-rank`, `TF_DS_PREFILL_CHUNK=2048`, and the verified RDMA/NCCL interfaces for the actual fabric. Set `TF_DS_SPILL_DIR` to a **new directory per experiment**. Shared disk prompt caches invalidate cold-prefill comparisons even when the server process restarts. Check engine logs for `cached=0` on every cold-prefill request.

The release controller sets `TF_DS_WORKSPACE_GIB=1`, `TF_DS_HOST_MIN_GIB=3.0` with the example 2.5 GiB external guard, and `TF_DS_MEM_FLOOR_INCLUDES_KERNEL=1`. The historical 2 GiB engine reserve is retained only in archived measurements; it conflicts with the release guard and is not the release default. Never globally drop caches or change kernel memory reserves just to improve a benchmark.

Benchmark an already-running authenticated endpoint:

```bash
python benchmarks/run_suite.py --label baseline --context-capacity 262144 --base http://127.0.0.1:19890 --key-file /absolute/key-file --output results/baseline
```

Run the identical command against the candidate with a new output directory and fresh spill namespace. The wrapper records every request and full result, plus hashes. Decode timing excludes the first token. For fixed-prefix runs, compare the request hashes before comparing long-context output speed. End-to-end code-task timing must use completed generation, not a fixed fragment.

Build all native extensions before loading model weights. `python3 scripts/cluster.py prebuild` does this on both idle ranks, in the prepared image and serving cache paths, with `MAX_JOBS=1` and an explicit build memory limit. Copying generated source with new timestamps can trigger a rebuild; do the final source transfer before prebuilding. Never clear a lock while a compiler is running. The Linux AIO test exercises the same native source without a model. The lifecycle tools stop only their owned containers or the explicitly configured previous service with `--takeover`; they do not manage SSH credentials or alter RDMA configuration. The authenticated API binds to loopback by default.

The 2026-10-28 protocol retains upstream `TF_DS_REPLAY=1` and `TF_DS_REPLAY_FLOOR=1`: encoder prefill plus bounded decoder replay. It does not claim full exact decoder prefill at 128K. Use the same setting for every compared engine, and give each run an empty spill namespace. Microbenchmarks in `benchmarks/transport_post.py`, `postmix.py`, and `prefill_hc.py` require idle GPUs and must not run beside a serving benchmark.

Second-round speed and quality runs disable lookup and every other experimental route. The existing first-round editing results do not establish the performance or correctness of enabling lookup and RDMA-post simultaneously; that combination has not been rerun. All features remain opt-in.
