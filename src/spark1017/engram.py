"""Two-rank Engram reduction with bounded gather storage.

Keep the projection and rank-ordered FP32 sum / BF16 boundary unchanged. Gather
rows in deterministic tiles, then add directly into the corresponding BF16
output slice. Each rank derives the same collective sequence from the shape.
Neither inputs nor gathered partials are modified. Other rank counts and input
types retain the upstream operation.
"""

GATHER_BYTES = 32 * 2**20


def sum_bf16(comm, partial):
    import torch

    if comm.world != 2 or partial.dtype != torch.float32:
        return comm.sum(partial).to(torch.bfloat16)
    result = torch.empty_like(partial, dtype=torch.bfloat16)
    if partial.ndim != 2 or partial.numel() * partial.element_size() * 2 <= GATHER_BYTES:
        gathered = comm.gather(partial)
        torch.add(gathered[0], gathered[1], out=result)
        return result
    # Engram's 25600-column FP32 projection fits 163 rows per gather. Keep
    # rows intact; for an unfamiliar width one row is the minimum tile.
    rows = max(1, GATHER_BYTES // (2 * partial.shape[1] * partial.element_size()))
    for begin in range(0, partial.shape[0], rows):
        end = min(begin + rows, partial.shape[0])
        gathered = comm.gather(partial[begin:end])
        torch.add(gathered[0], gathered[1], out=result[begin:end])
        # Release before the next gather allocates, rather than retaining two
        # receive buffers during Python's assignment of the next result.
        del gathered
    return result
