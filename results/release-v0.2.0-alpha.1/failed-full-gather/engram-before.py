"""Two-rank Engram reduction with the reference FP32 sum / BF16 boundary.

The gather owns fresh storage. Write the FP32 addition directly to a separate
BF16 output instead of allocating a full FP32 accumulator and casting it later.
Neither the input nor the gathered partials are modified. Other rank counts and
input types retain the upstream operation.
"""


def sum_bf16(comm, partial):
    import torch

    if comm.world != 2 or partial.dtype != torch.float32:
        return comm.sum(partial).to(torch.bfloat16)
    gathered = comm.gather(partial)
    result = torch.empty_like(partial, dtype=torch.bfloat16)
    torch.add(gathered[0], gathered[1], out=result)
    return result
