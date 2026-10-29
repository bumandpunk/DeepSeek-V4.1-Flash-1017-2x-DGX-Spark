# Bounded Engram reduction workspace

The two-rank Engram projection produces FP32 partials, gathers both, adds rank 0
then rank 1, and converts the result to BF16 before the Engram gate. At 2048 rows
by 25600 columns, the projection is 200 MiB per rank. The upstream operation
allocates a 400 MiB gather and clones a 200 MiB accumulator.

Spark1017 keeps the full projection unchanged, allocates the BF16 output once,
and gathers deterministic row tiles. Each tile is added in FP32 directly into
its slice of that BF16 output. For this projection width, each gather contains
at most 163 rows (less than 32 MiB across both ranks); the 2048-row operation
uses 13 collectives. Both ranks derive the same tile sequence from tensor shape.
The gather is released before the next tile allocates, avoiding overlapping
receive buffers. Input and gather storage are never mutated.

Decode-sized tensors use one gather. Other rank counts or non-FP32 inputs keep
the upstream reduction. Rows are kept intact; a hypothetical row wider than the
budget is the minimum tile, and non-matrix shapes use the single-gather path.
The supported model's 25600-column projection is covered by the GPU gate.

The implementation is [`src/spark1017/engram.py`](../src/spark1017/engram.py).
The pinned-source patch refuses missing or ambiguous projection anchors. Tiling
and addition into a lower-precision output are established techniques; this
project's contribution is their model-specific integration, bounded ownership,
paired validation and deployment evidence.

## Exactness and allocation gate

The [PyTorch addition API](https://docs.pytorch.org/docs/stable/generated/torch.add.html)
documents type promotion and the output tensor. The installed CUDA build is
checked directly by [`tests/test_engram_cuda.py`](../tests/test_engram_cuda.py):
random decode/prefill sizes, a split row boundary and short final tile, rounding
ties, cancellation, signed zeros, subnormals, infinities, NaNs, unchanged inputs
and gathered values, fallback ranks/types, and the full representative size.
BF16 results are compared as bits. Run it while the GPU is idle:

```bash
docker run --rm --gpus all --network none \
  --entrypoint python3 spark1017:0.2.0-alpha.1 \
  /opt/spark1017/tests/test_engram_cuda.py
```

The final image passed nine cases on each GB10 with the pinned NGC torch
build. Representative extra peak allocation fell from 600 MiB to 132 MiB, a
468 MiB saving. The image-bound checks and request-level measurements are
recorded separately in release evidence. A workspace saving alone is not an
end-to-end speed or quality claim.

## Why the whole gather was removed

An initial sum/BF16-output fusion reduced the peak from 600 MiB to 500 MiB, but
still allocated the full 400 MiB gather. Its clean baseline passed while its
edit profile failed at the first 8K request after API and concurrency checks.
That [rejected candidate](../results/release-v0.2.0-alpha.1/failed-full-gather/README.md)
is retained. It demonstrates why a single successful profile and a microtest
were insufficient. The tiled route is probed across all three profiles before
another clean application build and full release validation.

The allocator cap is also refreshed at the existing collective admission
boundary using `memory_reserved + max(0, MemAvailable - host_floor)`. Host
reserve and external guard are unchanged. CUDA device free memory is not
substituted for host MemAvailable, and graph-private slack receives no credit.
The refresh-only failure is retained too.
