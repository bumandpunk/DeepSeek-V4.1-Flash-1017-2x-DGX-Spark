"""Preserve decode's side-stream HC work after a network-fused post.

The network has already written the exact BF16 residual. Collapse/norm can
therefore run immediately; the next HC coefficients only need to be ready by
the sublayer's existing join. Reuse upstream kernels and reduction trees.
This schedule derives from TensorFold's MIT hc_pre2 dots path.
"""


def after_post(h, fn, scale, base, pre_in, norm_w, eps, hc_eps, iters,
               out, pre_out, post, comb, part, *, rot=None, sink=None):
    import torch
    import triton
    from tensorfold.families.deepseek_v41.cuda import kernels as K

    rows, d = h.shape[0], h.shape[-1]
    if sink is None or not K.hc_dots_on(d):
        return K.hc_pre2(h, fn, scale, base, pre_in, norm_w, eps, hc_eps, iters,
                         out, pre_out, post, comb, part, rot=rot, sink=sink)
    rot = rot or []
    assert not rot or (K.hc_rot_on(d) and len(rot) <= 4)
    st = [t for pair in rot for t in pair] + [out, out] * (4-len(rot))
    rs = d//1024 if rot and rows <= K.HC_ROT_SPLIT_ROWS else 1
    K._hc_finish_s[(rows, 1+len(rot)*rs)](
        h, part, base, scale, pre_in, norm_w, out, pre_out, post, comb, eps,
        hc_eps, *st, D=d, NB=K.HC_BLOCKS, ITERS=iters, BLOCK=1024,
        NROT=len(rot), RSPLIT=rs, num_warps=8, SINK=False, **K._pdl())
    sink.wait_stream(torch.cuda.current_stream())
    rb=K._hc_rb(rows)
    with torch.cuda.stream(sink):
        K._hc_mix_part[(triton.cdiv(rows,rb), K.HC_BLOCKS, 3)](
            h,h,h,0,post,comb,fn,part,rows,WIDE=4*d,D=d,NB=K.HC_BLOCKS,
            SUB=128,MB=8,WORLD=1,POSTED=False,RB=rb,num_warps=2)
        K._hc_sinkhorn[(rows,)](part,base,scale,pre_out,post,comb,eps,hc_eps,
                               D=d,NB=K.HC_BLOCKS,ITERS=iters,num_warps=8)
    return h
