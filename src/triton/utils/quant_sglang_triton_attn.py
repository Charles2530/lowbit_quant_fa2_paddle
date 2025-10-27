import logging
import pdb

import paddle
import triton
import triton.language as tl


def is_hip() -> bool:
    """Return whether it is HIP on the AMD ROCm platform."""
>>>>>>    return torch.version.hip is not None


is_hip_ = is_hip()
import time

logger = logging.getLogger(__name__)
logger.warn(
    "The following error message 'operation scheduled before its operands' can be ignored."
)


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


def cvt_dim2seq(dim, bits):
    return dim * 32 // bits


@triton.jit
def cvt_seq2dim(len, bits):
    return len * bits // 32


@triton.jit
def cvt_seq2scale(seq, group_size):
    return seq // group_size


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K,
    Kscale,
    Kmn,
    V,
    Vscale,
    Vmn,
    group_size: tl.constexpr,
    bits: tl.constexpr,
    feat_per_int: tl.constexpr,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kb,
    stride_buf_kd,
    stride_buf_kh,
    stride_buf_vb,
    stride_buf_vs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV // feat_per_int)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv // feat_per_int
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    qq = tl.load(Q + off_q, mask=mask_d, other=0.0)
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    e_max = -float("inf")
    e_sum = 0.0
    meta_dtype = tl.float32
    acc = tl.zeros([BLOCK_DV], dtype=meta_dtype)
    if split_kv_end > split_kv_start:
        num = 255 >> 8 - bits
        ratio = bits * group_size // 32
        arange_1 = tl.arange(0, BLOCK_N // feat_per_int) * feat_per_int
        arange_2 = tl.arange(0, BLOCK_N // group_size) * group_size
        shift_in_int_k = tl.arange(0, BLOCK_N) % feat_per_int * bits
        arange_3 = tl.arange(0, BLOCK_DV // group_size) * group_size
        arange_4 = tl.arange(0, BLOCK_N)
        shift_in_int_v = tl.arange(0, BLOCK_DMODEL) % feat_per_int * bits
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n_1 = start_n + arange_1
            k_loc = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_1,
                mask=offs_n_1 < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                k_loc[None, :] // feat_per_int
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None] * stride_buf_kd
            )
            k = tl.load(
                K + offs_buf_k,
                mask=(offs_n_1[None, :] < split_kv_end) & mask_d[:, None],
                other=0.0,
            )
            offs_n_scale = start_n + arange_2
            kv_loc_scale = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch_req_idx
                + offs_n_scale,
                mask=offs_n_scale < split_kv_end,
                other=0,
            )
            offs_buf_kscale = (
                kv_loc_scale[None, :] // group_size
                + cur_kv_head * (stride_buf_kh // ratio)
                + offs_d[:, None] * (stride_buf_kd // ratio)
            )
            kscale = tl.load(
                Kscale + offs_buf_kscale,
                mask=(offs_n_scale[None, :] < split_kv_end) & mask_d[:, None],
                other=0.0,
            )
            kmn = tl.load(
                Kmn + offs_buf_kscale,
                mask=(offs_n_scale[None, :] < split_kv_end) & mask_d[:, None],
                other=0.0,
            )
            k = (
                k.expand_dims(2)
                .broadcast_to(BLOCK_DMODEL, BLOCK_N // feat_per_int, feat_per_int)
                .reshape(BLOCK_DMODEL, BLOCK_N)
            )
            kscale = (
                kscale.expand_dims(2)
                .broadcast_to(BLOCK_DMODEL, BLOCK_N // group_size, group_size)
                .reshape(BLOCK_DMODEL, BLOCK_N)
            )
            kmn = (
                kmn.expand_dims(2)
                .broadcast_to(BLOCK_DMODEL, BLOCK_N // group_size, group_size)
                .reshape(BLOCK_DMODEL, BLOCK_N)
            )
            t_k = k >> shift_in_int_k[None, :] & num
            t_k = tl.fma(t_k, kscale, kmn)
            qk = tl.sum(qq[:, None] * t_k, 0)
            qk *= sm_scale
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            offs_n = start_n + arange_4
            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))
            v_loc = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_v = (
                v_loc[:, None] * stride_buf_vs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & mask_dv[None, :],
                other=0.0,
            )
            offs_dv_scale = start_n + arange_3
            offs_buf_vscale = (
                v_loc[:, None] * (stride_buf_vs // ratio)
                + cur_kv_head * (stride_buf_kh // ratio)
                + offs_dv_scale[None, :] // group_size
            )
            mask_dv_scale = offs_dv_scale < Lv
            vscale = tl.load(
                Vscale + offs_buf_vscale,
                mask=(offs_n[:, None] < split_kv_end) & mask_dv_scale[None, :],
                other=0.0,
            )
            vmn = tl.load(
                Vmn + offs_buf_vscale,
                mask=(offs_n[:, None] < split_kv_end) & mask_dv_scale[None, :],
                other=0.0,
            )
            v = (
                v.expand_dims(2)
                .broadcast_to(BLOCK_N, BLOCK_DMODEL // feat_per_int, feat_per_int)
                .reshape(BLOCK_N, BLOCK_DMODEL)
            )
            vscale = (
                vscale.expand_dims(2)
                .broadcast_to(BLOCK_N, BLOCK_DMODEL // group_size, group_size)
                .reshape(BLOCK_N, BLOCK_DMODEL)
            )
            vmn = (
                vmn.expand_dims(2)
                .broadcast_to(BLOCK_N, BLOCK_DMODEL // group_size, group_size)
                .reshape(BLOCK_N, BLOCK_DMODEL)
            )
            t_v = v >> shift_in_int_v[None, :] & num
            v = tl.fma(t_v, vscale, vmn)
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)
            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max
        offs_dv2 = tl.arange(0, BLOCK_DV)
        mask_dv2 = offs_dv2 < Lv
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv2
        )
        tl.store(Att_Out + offs_mid_o, acc / e_sum, mask=mask_dv2)
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )
        tl.store(Att_Out + offs_mid_o_1, e_max + tl.log(e_sum))


@triton.jit
def _fwd_kernel_stage1_cesu_load(
    Q,
    K,
    Kscale,
    Kmn,
    V,
    Vscale,
    Vmn,
    group_size: tl.constexpr,
    bits: tl.constexpr,
    feat_per_int: tl.constexpr,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kb,
    stride_buf_kd,
    stride_buf_kh,
    stride_buf_vb,
    stride_buf_vs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV // feat_per_int)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv // feat_per_int
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    qq = tl.load(Q + off_q, mask=mask_d, other=0.0)
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    e_max = -float("inf")
    e_sum = 0.0
    meta_dtype = tl.float32
    acc = tl.zeros([BLOCK_DV], dtype=meta_dtype)
    if split_kv_end > split_kv_start:
        arange_1 = tl.arange(0, BLOCK_N // feat_per_int)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            offs_n_1 = start_n + arange_1
            k_loc = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_1,
                mask=offs_n_1 < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                k_loc[None, :]
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None] * stride_buf_kd
            )
            k = tl.load(
                K + offs_buf_k,
                mask=(offs_n_1[None, :] < split_kv_end) & mask_d[:, None],
                other=0.0,
            )
            """ offs_n_scale = start_n + arange_2
            kv_loc_scale = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_scale, 
                mask=offs_n_scale < split_kv_end,
                other=0,
            ) # kv idx
            offs_buf_kscale = (
                kv_loc_scale[None, :] // group_size #  stride_buf_kbs=head*dim
                + cur_kv_head * (stride_buf_kh//ratio)
                + offs_d[:, None] * (stride_buf_kd//ratio)
            )
            
            kscale = tl.load(
                Kscale + offs_buf_kscale,
                mask=(offs_n_scale[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            kmn = tl.load(
                Kmn + offs_buf_kscale, 
                mask=(offs_n_scale[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            ) """
            """ kscale = kscale.expand_dims(2).broadcast_to(BLOCK_DMODEL, BLOCK_N // group_size, group_size).reshape(BLOCK_DMODEL, BLOCK_N)
            kmn = kmn.expand_dims(2).broadcast_to(BLOCK_DMODEL, BLOCK_N // group_size, group_size).reshape(BLOCK_DMODEL, BLOCK_N)

            t_k = (k >> shift_in_int_k[None, :]) & num   # 从每个 int32 中取出第 idx 个 int4
            # for debug: t_k = tl.fma(t_k.to(tl.float16), kscale.to(tl.float16), kmn.to(tl.float16))  # [8, 64] * [2, 64]
            t_k = tl.fma(t_k, kscale, kmn)  # [8, 64] * [2, 64]
            qk = tl.sum(qq[:, None] * t_k, 0)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap) """
            """ qk = tl.where(offs_n < split_kv_end, qk, float("-inf")) """
            """ v_loc = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n, 
                mask=offs_n < split_kv_end,
                other=0, 
            ) # kv idx """
            offs_buf_v = (
                offs_n[:, None] * stride_buf_vs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & mask_dv[None, :],
                other=0.0,
            )
            """ offs_dv_scale = start_n + arange_3
            offs_buf_vscale = (
                v_loc[:, None] * (stride_buf_vs//ratio) #  stride_buf_kbs=head*dim
                + cur_kv_head * (stride_buf_kh//ratio)
                + offs_dv_scale[None, :] // group_size
            )
            mask_dv_scale = offs_dv_scale < Lv
            vscale = tl.load(
                Vscale + offs_buf_vscale,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv_scale[None, :]),
                other=0.0, 
            )
            vmn = tl.load(
                Vmn + offs_buf_vscale, 
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv_scale[None, :]),
                other=0.0,
            ) """
            """ vscale = vscale.expand_dims(2).broadcast_to(BLOCK_N, BLOCK_DMODEL // group_size, group_size).reshape(BLOCK_N, BLOCK_DMODEL)
            vmn = vmn.expand_dims(2).broadcast_to(BLOCK_N, BLOCK_DMODEL // group_size, group_size).reshape(BLOCK_N, BLOCK_DMODEL)

            t_v = (v >> shift_in_int_v[None, :]) & num   # 从每个 int32 中取出第 idx 个 int4
            v = tl.fma(t_v, vscale, vmn)  # [8, 64] * [2, 64]

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)
            
            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_dv2 = tl.arange(0, BLOCK_DV)  # 0~64
        mask_dv2 = offs_dv2 < Lv
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv2
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum, 
            mask=(mask_dv2),
        ) 

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
        ) """


def _decode_att_m_fwd(
    qq,
    k_buffer,
    v_buffer,
    quant_info,
    att_out,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    logit_cap,
):
    BLOCK = 128
    NUM_KV_SPLITS = num_kv_splits
    group_size, bits = quant_info
    feat_per_int = 32 // bits
    Lk = k_buffer[0].shape[1]
    Lv = cvt_dim2seq(v_buffer[0].shape[-1], bits)
    batch, head_num = B_req_idx.shape[0], qq.shape[1]
    grid = batch, head_num, NUM_KV_SPLITS
    kv_group_num = qq.shape[1] // k_buffer[0].shape[2]
    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)
    _fwd_kernel_stage1[grid](
        qq,
        k_buffer[0],
        k_buffer[1],
        k_buffer[2],
        v_buffer[0],
        v_buffer[1],
        v_buffer[2],
        group_size,
        bits,
        feat_per_int,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        qq.stride(0),
        qq.stride(1),
        k_buffer[0].stride(0),
        k_buffer[0].stride(1),
        k_buffer[0].stride(2),
        v_buffer[0].stride(0),
        v_buffer[0].stride(1),
        v_buffer[0].stride(2),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    O,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv
    for split_kv_id in range(0, NUM_KV_SPLITS):
        tv = tl.load(
            Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
        )
        tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - n_e_max)
        acc += exp_logic * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max
    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(logits, qq, o, v_buffer, quant_info, num_kv_splits):
    batch, head_num = qq.shape[0], qq.shape[1]
    group_size, bits = quant_info
    Lv = cvt_dim2seq(v_buffer[0].shape[-1], bits)
    BLOCK_DV = triton.next_power_of_2(Lv)
    NUM_KV_SPLITS = num_kv_splits
    extra_kargs = {}
    if is_hip_:
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}
    grid = batch, head_num
    _fwd_kernel_stage2[grid](
        logits,
        o,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs
    )


def decode_attention_fwd_normal(
    qq,
    k_buffer,
    v_buffer,
    quant_info,
    o,
    req_to_token,
    b_req_idx,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    _decode_att_m_fwd(
        qq,
        k_buffer,
        v_buffer,
        quant_info,
        attn_logits,
        req_to_token,
        b_req_idx,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(attn_logits, qq, o, v_buffer, quant_info, num_kv_splits)


def decode_attention_fwd(
    qq,
    k_buffer: tuple,
    v_buffer: tuple,
    quant_info: tuple,
    o,
    req_to_token,
    b_req_idx,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    assert num_kv_splits == attn_logits.shape[2]
    kv_group_num = qq.shape[1] // v_buffer[0].shape[2]
    if kv_group_num == 1:
        decode_attention_fwd_normal(
            qq,
            k_buffer,
            v_buffer,
            quant_info,
            o,
            req_to_token,
            b_req_idx,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            logit_cap,
        )
