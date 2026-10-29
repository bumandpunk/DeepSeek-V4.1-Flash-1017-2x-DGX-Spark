"""GPU release gate: exact FP32 reduction bits, ownership and peak allocation.

Run in the pinned runtime image with an idle CUDA device. This is a GPU check,
not part of CPU-only CI. JSON on stdout is suitable for recording as evidence.
"""
import gc
import json
import torch
from spark1017.engram import GATHER_BYTES, sum_bf16


class Gather:
    def __init__(self, other, world=2):
        self.other, self.world = other, world
        self.observed = None
        self.calls = 0
        self.max_bytes = 0

    def gather(self, x):
        other = self.other
        if x.shape != other.shape:
            assert x.ndim == other.ndim == 2 and x.shape[1] == other.shape[1]
            begin = x.storage_offset() // x.stride(0)
            other = other[begin:begin + x.shape[0]]
        result = torch.stack([x] + [other] * (self.world - 1))
        self.calls += 1
        self.max_bytes = max(self.max_bytes, result.numel() * result.element_size())
        # A detached copy outside the measured path is used for ownership tests.
        if self.observed is not None:
            self.observed.append(result)
        return result

    def sum(self, x):
        if self.world == 1:
            return x
        gathered = self.gather(x)
        acc = gathered[0].clone()
        for rank in range(1, self.world):
            acc += gathered[rank]
        return acc


def identical(a, b):
    assert a.dtype == b.dtype == torch.bfloat16
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def main():
    torch.manual_seed(1017)
    device = 'cuda'
    cases = []
    for shape in [(1, 25600), (6, 25600), (37, 1003), (164, 25600)]:
        x, y = [torch.randn(shape, device=device, dtype=torch.float32) for _ in range(2)]
        cases.append((x, y))
    # BF16 rounding ties, cancellation, signed zero, subnormals, infinities and
    # NaNs. NaN payloads are compared as bits rather than floating equality.
    edge = torch.tensor([0., -0., 1., -1., 1.00390625, -1.00390625,
                         1.01171875, 3.4e38, -3.4e38, 1e-40, -1e-40,
                         float('inf'), -float('inf'), float('nan')], device=device)
    cases.append((edge[:, None].expand(-1, edge.numel()).contiguous(),
                  edge[None, :].expand(edge.numel(), -1).contiguous()))
    for x, y in cases:
        comm = Gather(y)
        expected = comm.sum(x).to(torch.bfloat16)
        before_x, before_y = x.clone(), y.clone()
        comm.observed = []
        identical(sum_bf16(comm, x), expected)
        assert torch.equal(x.view(torch.int32), before_x.view(torch.int32))
        assert torch.equal(y.view(torch.int32), before_y.view(torch.int32))
        assert torch.equal(torch.cat([g[0] for g in comm.observed]).view(torch.int32), x.view(torch.int32))
        assert torch.equal(torch.cat([g[1] for g in comm.observed]).view(torch.int32), y.view(torch.int32))
    for world, dtype in [(1, torch.float32), (3, torch.float32), (2, torch.bfloat16)]:
        x, y = [torch.randn((7, 83), device=device, dtype=dtype) for _ in range(2)]
        comm = Gather(y, world)
        identical(sum_bf16(comm, x), comm.sum(x).to(torch.bfloat16))

    # Exact failing prefill allocation size: 2048 * 25600 * sizeof(float).
    x, y = [torch.randn((2048, 25600), device=device) for _ in range(2)]
    comm = Gather(y)
    reference = comm.sum(x).to(torch.bfloat16)
    identical(sum_bf16(comm, x), reference)
    del reference
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    measurements = {}
    for name, fn in [('reference', lambda: comm.sum(x).to(torch.bfloat16)),
                     ('sum_cast', lambda: sum_bf16(comm, x))]:
        torch.cuda.empty_cache()
        held = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        comm.calls, comm.max_bytes = 0, 0
        output = fn()
        torch.cuda.synchronize()
        measurements[name] = {'peak_extra_bytes': torch.cuda.max_memory_allocated() - held,
                              'gather_calls': comm.calls, 'max_gather_bytes': comm.max_bytes}
        del output
    saved = measurements['reference']['peak_extra_bytes'] - measurements['sum_cast']['peak_extra_bytes']
    assert saved >= 450 * 2**20, measurements
    assert measurements['sum_cast']['max_gather_bytes'] <= GATHER_BYTES
    assert measurements['sum_cast']['gather_calls'] > 1
    print(json.dumps({'passed': True, 'torch': torch.__version__,
                      'gpu': torch.cuda.get_device_name(), 'exact_cases': len(cases) + 4,
                      'measurements': measurements, 'peak_saved_bytes': saved}))


if __name__ == '__main__':
    main()
