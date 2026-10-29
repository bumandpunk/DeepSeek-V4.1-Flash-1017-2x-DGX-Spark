"""Generate bounded large-tensor RDMA transport from pinned upstream primitives.

The proxy, registration, doorbells and failure handling remain upstream. Our
change is a segmented gather writing directly into rank-strided destinations,
so large prompt tensors reuse the existing small registered rings without an
extra tensor reassembly copy. Every segment follows stage then collect order.
"""
import argparse
import hashlib
from pathlib import Path


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('unsupported transport source: '+old[:60])
    return text.replace(old,new,1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    expected={'rdma_gather.cu':'4892bfbc016dc98e152e0982ed6d4bfcaf5400d875a38d2633f19e51625ac1b7',
              'rdma.py':'f18332f865fc1231dea3c06f3e9f6813899f5865774e15e5837ec48770d31430'}
    for name,digest in expected.items():
        if hashlib.sha256((args.source/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('unsupported transport revision: '+name)
    source=(args.source/'rdma_gather.cu').read_text()
    source=once(source,'uint64_t* seq = nullptr;', 'uint64_t* tickets = nullptr;\n    uint64_t* seq = nullptr;')
    source=once(source,'float4* __restrict__ own_dst) {','float4* __restrict__ own_dst, uint64_t* ticket = nullptr) {')
    source=once(source,'        *seq = q;','        *seq = q;\n        if (ticket) *ticket = q;')
    source=once(source,'    C10_CUDA_CHECK(cudaMemset(g->seq, 0, 8));',
                '    C10_CUDA_CHECK(cudaMemset(g->seq, 0, 8));\n    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->tickets), 16));')
    # An optional rank stride changes only destination addresses, never bytes.
    source=once(source,'int unroll) {','int unroll, int64_t output_stride4 = 0) {')
    source=once(source,'float4* out = dst + (size_t)p * n4;',
                'float4* out = dst + (size_t)p * (output_stride4 ? output_stride4 : n4);')
    begin=source.index('void gather(int64_t h, const at::Tensor& send, at::Tensor recv) {')
    end=source.index('\n}  // namespace',begin)
    function=source[begin:end]
    function=function.replace('void gather(', 'void gather_segmented(',1)
    function=once(function,'n % 4 == 0 && (uint64_t)n * 4 <= g.max_bytes',
                  'n > 0 && n % 4 == 0 && (uint64_t)n * 4 <= (256ull << 20)')
    function=once(function,'const int n4 = (int)(n / 4);','const int64_t total4 = n / 4;')
    anchor='    const int sblocks = '
    at=function.index(anchor)
    function=function[:at]+'''    const auto* input = reinterpret_cast<const float4*>(send.data_ptr());
    auto* output = reinterpret_cast<float4*>(recv.data_ptr());
    const int64_t capacity4 = g.max_bytes / 16;
    // Sequential stage/collect retains the upstream ring reuse and abort proof.
    // The CPU proxy sends both rails while the own-rank slice is copied.
    for (int64_t offset4 = 0; offset4 < total4; offset4 += capacity4) {
        const int n4 = (int)std::min(capacity4, total4-offset4);
'''+function[at:]
    function=once(function,'reinterpret_cast<float4*>(recv.data_ptr()) + (size_t)g.rank * n4',
                  'output + offset4 + (size_t)g.rank * total4')
    function=once(function,'reinterpret_cast<const float4*>(send.data_ptr()), n4,',
                  'input + offset4, n4,')
    function=once(function,'reinterpret_cast<const float4*>(send.data_ptr()), reinterpret_cast<float4*>(recv.data_ptr()), n4,',
                  'input + offset4, output + offset4, n4,')
    function=once(function,'own_early ? 1 : 0, collect_unroll ? 1 : 0);',
                  'own_early ? 1 : 0, collect_unroll ? 1 : 0, total4);')
    function=function.rstrip()[:-1]+'    }\n}\n'
    source=source[:end]+'\n'+function+source[end:]
    source=once(source,'#include <cuda_runtime.h>','#include <cuda_runtime.h>\n#include <cuda_bf16.h>')
    source=once(source,'\n}  // namespace','\n#include "fused_post.inc"\n\n}  // namespace')
    source=once(source,'    m.def("gather", &gather);',
                '    m.def("gather", &gather);\n    m.def("gather_segmented", &gather_segmented);\n    m.def("gather_post", &gather_post);\n    m.def("gather_post_pipeline", &gather_post_pipeline);')
    dest=root/'native/transport';dest.mkdir(parents=True,exist_ok=True)
    (dest/'rdma_gather.cu').write_text('// TensorFold Apache-2.0 transport; Spark1017 segmented destination extension.\n'+source)
    wrapper=(args.source/'rdma.py').read_text()
    wrapper=once(wrapper,'here = Path(__file__).parent',
                 'here = Path(__file__).resolve().parents[2] / "native" / "transport"')
    wrapper=once(wrapper,'name="tensorfold_rdma_gather_v5"','name="spark1017_rdma_segmented_v3"')
    wrapper=once(wrapper,'self.devices = device_names() if devices is None else list(devices)',
                 'self.segment_limit = 256 << 20\n        self.policy = (os.environ.get("SPARK1017_SEGMENTED_RDMA", "0"), os.environ.get("SPARK1017_RDMA_POST", "0"))\n        self.devices = device_names() if devices is None else list(devices)')
    wrapper=once(wrapper,'info = pickle.dumps((ext.local_info(self.h), slots, self.max_bytes))',
                 'info = pickle.dumps((ext.local_info(self.h), slots, self.max_bytes, self.segment_limit, self.policy))')
    wrapper=once(wrapper,'shapes = {(i[1], i[2]) for i in infos}',
                 'shapes = {(i[1], i[2], i[3], i[4]) for i in infos}')
    wrapper=once(wrapper,'            self.nccl.all_gather(send, recv)',
                 '''            if (self.rdma.policy[0] == "1" and send.dtype == torch.float32 and send.numel() % 4 == 0
                    and send.numel()*4 <= self.rdma.segment_limit
                    and send.is_contiguous() and recv.is_contiguous()):
                self.rdma.ext.gather_segmented(self.rdma.h, send, recv)
            else:
                self.nccl.all_gather(send, recv)''')
    (root/'src/spark1017/transport.py').write_text(wrapper)
    print('Generated segmented transport; registered ring capacity is unchanged.')


if __name__=='__main__':main()
