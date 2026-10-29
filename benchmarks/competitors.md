# Engine comparison protocol

## 2026-10-28: ordinary generation and long context

The new round uses the same fixed-prefix client (seed `20261027 + (length << 4) + occurrence`). E320 baseline/candidate/return-baseline keep vision enabled; the separate A0 Python/native comparison disables the Python vision tower to match the text-only native process. We do not combine A0 and E320 rows to infer engine speedups. Each final full suite has nine repetitions per short prompt, three per cold prefix length (8K, 32K, 128K), concurrency two/four, and one 512-token decode after a 128K prefix. Incomplete arms remain incomplete.

For the new candidate, `SPARK1017_RDMA_POST=all`; lookup, adaptive depth, storage, prompt-side HC scheduling, control fusion, persistent experts and the remote drafter are disabled. Baseline disables the RDMA-post route too. Capacity remains 524288 with two slots, prefill chunk 2048, five DSpark drafts and the joint policy. Both use the disclosed safety overlay and the same model bytes. An independent 2.5 GiB host guard stays active; it must not be disabled to make a long-context result complete.

The engine's encoder-prefill / bounded-decoder-replay behavior is retained (`TF_DS_REPLAY=1`, `TF_DS_REPLAY_FLOOR=1`). These timings are not measurements of full exact decoder prefill. Cold prefixes use new token sequences and empty spill namespaces; server logs establish prefix-cache misses. This does not mean the operating system's page cache or Engram reads are cold; global cache dropping is not part of the protocol.

Mia's deployment HEAD was checked again on 2026-10-28 and remains the pinned `27741e176a9e5b98c10c9c7d012fd3e4fcdc19b6`. We reuse its source-built binaries, public patches and the same A0 rank cache; its recommended downloaded quantization is not substituted. The earlier lucataco and vLLM investigations below are historical evidence, not new second-round full reruns. Elevatormusic's related recipe was reviewed, but its published numbers are not inserted into our measured ranking.

The RTX4090 runs the common client and isolated graders during final comparisons. Its remote-DSpark experiment is reported separately: it uses extra GPU inference resources, fewer screening repetitions and tracing, and must not be treated as a final matched ranking. Final source hashes, raw outputs, negative experiments and a return-baseline check are in `results/2026-10-28/`.

## 2026-10-27 protocol and source provenance

The comparison measures public engines on the same two GB10 nodes with the same A0 EXL3 2.9 bpw weight bytes. A0 is a quantized checkpoint, not the original full-precision model. The client and isolated graders run on the RTX4090 host. No external GPU serves any target or draft tokens.

All compared engines use a 524,288-token capacity, two engine slots, the same dual-rail fabric, and the same HTTP request bodies. Requests override server sampling defaults. Short tests use 384 output tokens; full-file edits must finish within a 16,384-token budget and pass the supplied executable assertions. Performance for incomplete runs is retained as incomplete, not assigned a synthetic speed. The original production weights, configurations and caches are kept separate.

## TensorFold and Spark1017

- TensorFold `07aa02585f64cfb04c60a96de6554c0d4708ed46`, with the disclosed pre-existing safety patch in `compat/`.
- A0 is an identity view; E320 is the separately documented expert-pruned view. A0 and E320 rows are never used to estimate the isolated effect of lookup.
- Historical deployment uses prefill chunks of 512. Matched final E320 and edit comparisons use 2048, five DSpark drafts, the joint policy, Engram DONTNEED=1, 4 GiB memory floor, 1 GiB workspace and 2 GiB host guard setting. An external host memory guard remains active.
- `SPARK1017_LOOKUP=1` enables the candidate; storage and calibrated depth allocation stay off in its final comparisons. Every run gets a fresh spill directory. Cold-prefill logs must show `cached=0`.

## Mia native engine

- Deployment source: `MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks@27741e176a9e5b98c10c9c7d012fd3e4fcdc19b6`.
- Engine: TensorFold `629c524a0ea57512a5342abb498d1b7c25353343`, plus all 26 public engine patches in filename order.
- GPU kit: `abe07c7d509a0199cc7fec15263be2aa52ac652f`, 222 files verified. The Zig compiler archive SHA256 is `9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8`.
- Public ReleaseFast / sm_121 build, native server and two-rank lane executable. Binaries and kit run in the same preserved NVIDIA-based container environment as the other engines; this is a source-build comparison, not a pull of the deployment's prebuilt serving image.
- Rank-cache bytes come from the same A0 view. Model, Engram, kit and cache are mounted read-only. Context/pool=524288, streams/parallel=2; graphs, side kernels, L2 prefetch and Engram AIO enabled. Native prompt chunk is compiled as 2048 (`dsv41/model.zig` and `native.zig`).
- TF_DS_RDMA_IDLE_S=2, TF_DS_RDMA_NAP_US=500, TF_DS_PREFILL_INTERLEAVE_MS=250, TF_DS_REPLAY_FLOOR=1. Fabric identifiers must match the actual host; the measured environment uses two CX7 rails and GID index 3.
- The current native engine supports the A0 384-expert geometry. It was not ported to E320 for these measurements.

## lucataco Python deployment

Deployment `7b46f9d9d8b21357f8998ccce8bdfa857a9ef1ce`, engine `8e0966e2b09250abe2f6d9cc51edc20630e6eb1a`. It uses the same A0 bytes and shared client. The first inherited 4 GiB floor setup hit its allocator ceiling. The public 2 GiB default was then tested explicitly: short requests and 8K prefill completed, but at 32K the independent host guard observed 1876 MiB available for over four seconds and stopped only the experimental rank. The failure is part of the result; no completed 32K or 128K number is claimed.

## sudoingX vLLM deployment

Deployment `f8f20b370e41a3a220bc7d1ab96a61e4daa84964`, vLLM `6e517b15c1833cf72a7f557ee32524d98682e617`. Static inspection found that its EXL3 adapter rejects the current compressed dense drafter format, including `attn.wo_a`. Its own 2.8 bpw checkpoint would change the weight bytes. This engine was therefore not built or measured; it cannot appear in a measured ranking.

## Measurement limits

Arms ran sequentially on the same day. Frequency, temperature and host-memory samples are included; GPU clocks and power limits were not changed for the comparison. Full-file editing has only two repetitions per task. A larger randomized, interleaved, multi-day experiment is still needed before treating small timing differences as robust. Historical public-client random prefixes included wall time in their seeds; those long-context decode results are not paired inputs. Only the final `e320-fixed-baseline` and `spark-fixed-lookup` suites use the frozen prefix scheme with separate spill namespaces.
