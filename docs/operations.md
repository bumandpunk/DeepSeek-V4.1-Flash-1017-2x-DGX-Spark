# Two-node operation and recovery

Run the controller on the head. It uses key-based, noninteractive SSH to the
worker. No script changes host networking, global proxy settings, clocks, kernel
memory settings or unrelated containers. Keep the project checkout in place.

## Configuration and ownership

Copy `config/cluster.example.json` to `config/cluster.local.json`. Set both local
model/Engram paths, dedicated cache and state directories, the worker SSH target,
two active RoCE devices, GID index and fabric addresses. Paths must be absolute;
model and Engram mounts are read-only. A cache/state path inside either immutable
source is rejected, including symlink aliases on the actual node.
`model` must point directly to the directory containing `config.json`,
`tokenizer.json` and `model.safetensors.index.json`, not to the parent bundle
directory that contains separate `model/` and `engram/` folders.

The release default is `context=262144`, shared across two engine slots. The
fixed long-input regression still uses 131072 input tokens plus 512 output
tokens. A 524288-token cold start left too little physical headroom for safe
request admission on the measured pair, even after ordinary allocator cleanup.
Do not describe the historical 512K window setting as a validated release
capacity. Increasing it requires a new memory and quality validation pass.

Allocator limits are refreshed at the existing synchronized admission, prefill
and decode memory checks. The limit is current reserved CUDA memory plus current
`MemAvailable` above the protected host floor, rounded down to 32 MiB steps.
It can increase after host memory is reclaimed and decrease under pressure.
Held CUDA blocks are not revoked, and unused graph-private reservations are not
treated as free host memory. The original two-rank vote, eviction and external
2.5 GiB guard remain in force. This avoids treating a boot-time allocator cap as
the machine's permanent capacity.

Use the same API key on both hosts, stored outside the repository with mode 0600.
The default API binds to 127.0.0.1. An explicit 0.0.0.0 binding enables LAN access
with the same required authentication. Never put the key itself in JSON, a command
line flag, or a GitHub issue.

The deployment name owns only `<deployment>-r0` and `<deployment>-r1`, identified
by a Docker label. A container with the same name and a different/missing label
is never stopped or replaced. Precompilation uses separately owned `<deployment>-prebuild-r0/r1` containers; `stop` and `restore` also remove these if a compilation was interrupted. There is no automatic `docker prune` or cache deletion.

## Build and start

```bash
python3 scripts/cluster.py doctor
python3 scripts/build.py --no-cache
python3 scripts/cluster.py prebuild
./start.sh --profile edit
./verify.sh
./status.sh
```

`doctor` checks paths, shard presence, API-key permissions, local fabric/GID,
Docker and GB10. `doctor --idle` also applies the startup memory threshold. It is
a local configuration/link check, not a substitute for a two-node RDMA transfer.
Startup requires identical image IDs on both ranks; the engine also checks its
runtime sources and feature configuration across ranks.

The build uses the pinned official NVIDIA arm64 base and a fixed public
TensorFold commit. NGC's torch/torchvision/triton/numpy versions stay constrained.
The public-source fetch uses HTTP/1.1 and at most three attempts, then fails instead of selecting another revision. The image includes source and installed-package manifests. It is sent over SSH
to the worker and its image ID is compared before use.

For an offline base, use `scripts/build.py --base-image <local-tag> --no-cache`.
The tag is accepted only if its image/config digest is exactly the pinned
official base in `docker/base-image.json`. An arbitrary former serving image is
rejected. `HTTP_PROXY`/`HTTPS_PROXY` may be supplied per build invocation; the
scripts do not change daemon-wide proxy configuration. A public HTTPS package mirror can be selected per build:

```bash
python3 scripts/build.py --no-cache --package-index https://pypi.tuna.tsinghua.edu.cn/simple
```

Downloaded wheels must match the official PyPI hashes in `docker/package-hashes.json`, regardless of the selected mirror. The installed-image manifest records the verified filenames and hashes. No global pip configuration or TLS verification is changed.

`prebuild` compiles on both idle GPUs with MAX_JOBS=1 and an 8 GiB container
memory limit. It uses the same fresh cache path as serving and does not load
weights. Stop other inference before this step. The first model load creates a
new rank cache; no old private source mounts or compiled extensions are required.

## Take over an existing service

Configure `previous_service` as the head's systemd **user** unit. Configure
`previous_health` with its `base`, `key_file` and `model`. Set each node's
`previous_paths` to the actual small configuration files/directories and
`previous_containers` to the containers the service manages. Avoid weight/cache
directories in snapshots. Paths larger than 16 MiB are rejected rather than silently skipped.

```bash
python3 scripts/cluster.py snapshot
./start.sh --profile edit --takeover
```

Snapshots contain config archives/hashes, private container specs and pinned old
image tags, with private filesystem permissions. The previous unit's state is
recorded. Snapshot creation does not stop it. If an existing server must be stopped to free memory for `prebuild`, take this snapshot while it is still active, then stop the configured user unit and run `prebuild`. On preparation failure, run `cluster.py restore`; on success, continue with `start --takeover`. The startup recovery timer is armed by `start`, so keep the preparation window attended. `--takeover` is required to stop a
previously active configured service. Other services remain outside this lifecycle.

Before takeover, an independent systemd timer is armed. The recovery script and
config are copied into the state directory. If startup fails, immediate recovery
is attempted. The timer retries failed recovery, including a temporarily busy
lifecycle lock. At the configured deadline it stops the experimental containers
and restores the original active service. Keep the user manager alive across logout
(for example, an already-enabled user linger setup); the script does not change it.

## Profiles and stopping

| Profile | Repeated-code lookup | RDMA/HC fusion | Intended use |
|---|---|---|---|
| `baseline` | off | off | Paired reference |
| `edit` | on | off | Source-preserving edits |
| `prefill-experimental` | off | on | Experimental prompt/decode transport |

Every other experimental route is off. No profile combines lookup with transport
fusion. Lookup falls back to DSpark with multiple live streams; it is not an
optimized concurrent editing scheduler.

```bash
./stop.sh
./start.sh --profile baseline
```

Stopping preserves model files, compiled extensions, rank caches, snapshots and
server logs. Each start uses a new prompt-spill namespace. The rollback deadline
remains active across profile changes. `stop` alone does not restore the previous
service. `status` reports each rank separately.

## Restore and verify

```bash
python3 scripts/cluster.py restore
```

Recovery stops only owned experimental containers, verifies backup integrity and
that original files are unchanged, restarts the original unit only if it was
previously active, compares image/command/environment/mounts/running states, and
runs authenticated API checks. Only then is the timer disarmed. If original
files changed concurrently, recovery stops and preserves both versions for an
operator to inspect; it never blindly overwrites them from a stale backup.

The five API checks cover model listing, chat, completions, streamed completion
and rejection of a wrong key. Check any separate ingress/web application with
its own health procedure before returning external traffic.

For an intentional long-running experimental deployment, `cluster.py confirm`
reruns API checks and disarms the scheduled rollback. It does not certify
production stability. The engine admission/cache budget reserves the configured guard floor plus 0.5 GiB of host headroom (3.0 GiB with the example guard). This keeps normal cache retention above the external stop threshold; it does not relax the guard. Memory guards remain active and stop an owned rank after
the configured low-memory hold time; they never kill an unrelated process.

## Troubleshooting

Use `status` and each node's state directory (`last-server.log`,
`memory-guard.log`, `lifecycle.json`). Raw snapshots/logs may contain private paths
and container settings; redact them before sharing. A Docker daemon restart or
machine reboot is outside the in-process recovery guarantee; preserved config
archives and pinned images support manual recovery.

If build or preparation fails, leave the original service running or restore it.
If a rank is killed by its guard, inspect memory pressure and lower workload
requirements; keep the 2.5 GiB guard enabled. Do not clear global caches, change
clocks, delete original model data, or reuse a corrupt rank cache to force a result.

### Engram workspace validation

The release includes a two-rank FP32 sum/BF16 output path and an idle-GPU
exactness/peak-allocation gate. See [Engram workspace](engram-reduction.md).
It preserves the memory reserve and external guard; do not lower those guards
to make a failed deployment appear healthy. The release evidence separates
failed allocator diagnostics from completed benchmark runs.
