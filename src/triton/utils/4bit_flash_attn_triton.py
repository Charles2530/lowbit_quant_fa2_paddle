import sys

sys.path.append("/data/charles/codes/flash-attn-v0")
import math
import os
import pdb
import random
import time

import numpy as np
import paddle
import triton
import triton.language as tl
from helper import *
from paddle_utils import *

DEBUG = True


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_kv"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
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
    Bias,
    Out,
    Lse,
    TMP,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_kv: tl.constexpr,
    seqlen_q_rounded: tl.constexpr,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_SEQ_K: tl.constexpr,
    QUANT_DIM_V: tl.constexpr,
    KSCALE_DIM: tl.constexpr,
    VSCALE_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    num = 255 >> 8 - bits
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = (
        Q
        + off_b * stride_qb
        + off_h * stride_qh
        + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    offs_seqk = tl.arange(0, QUANT_SEQ_K)
    k_ptrs = (
        K
        + off_b * stride_kb
        + off_h * stride_kh
        + (offs_d[:, None] * stride_kn + offs_seqk[None, :])
    )
    offs_dimv = tl.arange(0, QUANT_DIM_V)
    offs_seqlen_kv = tl.arange(0, seqlen_kv)
    v_ptrs = (
        V
        + off_b * stride_vb
        + off_h * stride_vh
        + (offs_seqlen_kv[:, None] * stride_vn + offs_dimv[None, :])
    )
    offs_kscale = tl.arange(0, KSCALE_DIM)
    stride_kbs = (stride_kb * (32 / bits / group_size)).to(tl.int32)
    stride_khs = (stride_kh * (32 / bits / group_size)).to(tl.int32)
    stride_kns = (stride_kn * (32 / bits / group_size)).to(tl.int32)
    kscale_ptr = (
        Kscale
        + off_b * stride_kbs
        + off_h * stride_khs
        + (offs_d[:, None] * stride_kns + offs_kscale[None, :])
    )
    kmn_ptr = (
        Kmn
        + off_b * stride_kbs
        + off_h * stride_khs
        + (offs_d[:, None] * stride_kns + offs_kscale[None, :])
    )
    offs_vscale = tl.arange(0, VSCALE_DIM)
    stride_vbs = (stride_vb * (32 / bits / group_size)).to(tl.int32)
    stride_vhs = (stride_vh * (32 / bits / group_size)).to(tl.int32)
    stride_vns = (stride_vn * (32 / bits / group_size)).to(tl.int32)
    vscale_ptr = (
        Vscale
        + off_b * stride_vbs
        + off_h * stride_vhs
        + (offs_n[:, None] * stride_vns + offs_vscale[None, :])
    )
    vmn_ptr = (
        Vmn
        + off_b * stride_vbs
        + off_h * stride_vhs
        + (offs_n[:, None] * stride_vns + offs_vscale[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    elif EVEN_HEADDIM:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    else:
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            other=0.0,
        )
    idx_range = tl.arange(0, seqlen_kv).expand_dims(0).broadcast_to(BLOCK_M, seqlen_kv)
    meta_dtype = tl.float32
    q = q.to(meta_dtype)
    end_n = (
        seqlen_kv if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, QUANT_SEQ_K)
    )
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
                kscale = tl.load(kscale_ptr + start_n * stride_kns)
                kmn = tl.load(kmn_ptr + start_n * stride_kns)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=offs_d[:, None] < headdim,
                    other=0.0,
                )
                kscale = tl.load(
                    kscale_ptr + start_n * stride_kns,
                    mask=offs_d[:, None] < headdim,
                    other=0.0,
                )
                kmn = tl.load(
                    kmn_ptr + start_n * stride_kns,
                    mask=offs_d[:, None] < headdim,
                    other=0.0,
                )
        elif EVEN_HEADDIM:
            k = tl.load(
                k_ptrs + start_n * stride_kn,
                mask=(start_n + offs_seqk)[None, :] < QUANT_SEQ_K,
                other=0.0,
            )
            kscale = tl.load(
                kscale_ptr + start_n * stride_kns,
                mask=(start_n + offs_kscale)[None, :] < KSCALE_DIM,
                other=0.0,
            )
            kmn = tl.load(
                kmn_ptr + start_n * stride_kns,
                mask=(start_n + offs_kscale)[None, :] < KSCALE_DIM,
                other=0.0,
            )
        else:
            k = tl.load(
                k_ptrs + start_n * stride_kn,
                mask=((start_n + offs_seqk)[None, :] < QUANT_SEQ_K)
                & (offs_d[:, None] < headdim),
                other=0.0,
            )
            kscale = tl.load(
                kscale_ptr + start_n * stride_kns,
                mask=((start_n + offs_kscale)[None, :] < KSCALE_DIM)
                & (offs_d[:, None] < headdim),
                other=0.0,
            )
            kmn = tl.load(
                kmn_ptr + start_n * stride_kns,
                mask=((start_n + offs_kscale)[None, :] < KSCALE_DIM)
                & (offs_d[:, None] < headdim),
                other=0.0,
            )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        kscale = (
            kscale.expand_dims(2)
            .broadcast_to(BLOCK_N, KSCALE_DIM, 4)
            .reshape(BLOCK_N, KSCALE_DIM * 4)
        )
        kmn = (
            kmn.expand_dims(2)
            .broadcast_to(BLOCK_N, KSCALE_DIM, 4)
            .reshape(BLOCK_N, KSCALE_DIM * 4)
        )
        for idx in range(feat_per_int):
            shift_in_int = idx % feat_per_int * bits
            t_k = k >> shift_in_int & num
            t_k = tl.fma(t_k.to(meta_dtype), kscale, kmn)
            t_qk = tl.dot(q, t_k)
            t_qk = (
                t_qk.expand_dims(2)
                .broadcast_to(BLOCK_M, BLOCK_N // feat_per_int, feat_per_int)
                .reshape(BLOCK_M, BLOCK_N)
            )
            qk = tl.where(idx_range % feat_per_int == idx, t_qk, qk)
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_kv, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(
                offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf")
            )
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=start_n + offs_n < seqlen_kv, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_kv),
                        other=0.0,
                    ).to(tl.float32)
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
                vscale = tl.load(vscale_ptr + start_n * stride_vns)
                vmn = tl.load(vmn_ptr + start_n * stride_vns)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=offs_dimv[None, :] < QUANT_DIM_V,
                    other=0.0,
                )
                vscale = tl.load(
                    vscale_ptr + start_n * stride_vns,
                    mask=offs_vscale[None, :] < VSCALE_DIM,
                    other=0.0,
                )
                vmn = tl.load(
                    vmn_ptr + start_n * stride_vns,
                    mask=offs_vscale[None, :] < VSCALE_DIM,
                    other=0.0,
                )
        elif EVEN_HEADDIM:
            v = tl.load(
                v_ptrs + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_kv,
                other=0.0,
            )
            vscale = tl.load(
                vscale_ptr + start_n * stride_vns,
                mask=(start_n + offs_n)[:, None] < seqlen_kv,
                other=0.0,
            )
            vmn = tl.load(
                vmn_ptr + start_n * stride_vns,
                mask=(start_n + offs_n)[:, None] < seqlen_kv,
                other=0.0,
            )
        else:
            v = tl.load(
                v_ptrs + start_n * stride_vn,
                mask=((start_n + offs_n)[:, None] < seqlen_kv)
                & (offs_dimv[None, :] < QUANT_DIM_V),
                other=0.0,
            )
            vscale = tl.load(
                vscale_ptr + start_n * stride_vns,
                mask=((start_n + offs_n)[:, None] < seqlen_kv)
                & (offs_vscale[None, :] < VSCALE_DIM),
                other=0.0,
            )
            vmn = tl.load(
                vmn_ptr + start_n * stride_vns,
                mask=((start_n + offs_n)[:, None] < seqlen_kv)
                & (offs_vscale[None, :] < VSCALE_DIM),
                other=0.0,
            )
        p = p.to(meta_dtype)
        vscale = (
            vscale.expand_dims(2)
            .broadcast_to(BLOCK_N, VSCALE_DIM, 4)
            .reshape(BLOCK_N, VSCALE_DIM * 4)
        )
        vmn = (
            vmn.expand_dims(2)
            .broadcast_to(BLOCK_N, VSCALE_DIM, 4)
            .reshape(BLOCK_N, VSCALE_DIM * 4)
        )
        for idx in range(feat_per_int):
            shift_in_int = idx % feat_per_int * bits
            t_v = v >> shift_in_int & num
            t_v = tl.fma(t_v.to(meta_dtype), vscale, vmn)
            t_ao = tl.dot(p, t_v)
            t_ao = (
                t_ao.expand_dims(2)
                .broadcast_to(
                    seqlen_q_rounded, BLOCK_HEADDIM // feat_per_int, feat_per_int
                )
                .reshape(seqlen_q_rounded, BLOCK_HEADDIM)
            )
            acc_o = tl.where(idx_range % feat_per_int == idx, t_ao, acc_o)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    elif EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
    else:
        tl.store(
            out_ptrs,
            acc_o,
            mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        )


def cdiv(x, y):
    return (x + y - 1) // y


def _quantized_flash_attn_forward(
    q,
    kcode,
    kscale,
    kmn,
    vcode,
    vscale,
    vmn,
    group_size=None,
    bits=None,
    bias=None,
    causal=False,
    softmax_scale=None,
):
    bits_per_int = 32
    batch, seqlen_q, nheads, d = q.shape
    _, _, _, qseqlen_k = kcode.shape
    _, seqlen, _, qd = vcode.shape
    assert kcode.shape == (batch, d, nheads, qseqlen_k)
    assert vcode.shape == (batch, seqlen, nheads, qd)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert (
        qd == d * bits / bits_per_int
    ), "head dimention of quantized value is not correct"
    assert (
        qseqlen_k == seqlen * bits / bits_per_int
    ), "seqlen of quantized key is not correct"
    assert kcode.dtype == vcode.dtype, "Key and Value tensors must have the same type"
    assert q.dtype in [
        paddle.float16,
        paddle.bfloat16,
    ], "Only support query to be fp16 and bf16"
    assert (
        q.is_cuda
        and kcode.is_cuda
        and vcode.is_cuda
        and kscale.is_cuda
        and vscale.is_cuda
        and kmn.is_cuda
        and vmn.is_cuda
    )
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, paddle.float32]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen) or (seqlen_q, seqlen)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen)
    bias_strides = (
        (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    )
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = paddle.empty(
        (batch, nheads, seqlen_q_rounded), device=q.place, dtype=paddle.float32
    )
    tmp = paddle.empty(
        (batch, nheads, seqlen_q_rounded), device=q.place, dtype=paddle.float32
    )
    o = paddle.empty_like(q)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4
    feat_per_int = int(bits_per_int / bits)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        kcode,
        kscale,
        kmn,
        vcode,
        vscale,
        vmn,
        group_size,
        bits,
        feat_per_int,
        bias,
        o,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        kcode.stride(0),
        kcode.stride(2),
        kcode.stride(1),
        vcode.stride(0),
        vcode.stride(2),
        vcode.stride(1),
        *bias_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        qseqlen_k // 32,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        QUANT_SEQ_K=qseqlen_k,
        QUANT_DIM_V=qd,
        KSCALE_DIM=seqlen // group_size,
        VSCALE_DIM=BLOCK_HEADDIM // group_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale


def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [paddle.float16, paddle.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, paddle.float32]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (
        (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    )
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = paddle.empty(
        (batch, nheads, seqlen_q_rounded), device=q.place, dtype=paddle.float32
    )
    tmp = paddle.empty(
        (batch, nheads, seqlen_q_rounded), device=q.place, dtype=paddle.float32
    )
    o = paddle.empty_like(q)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        o,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale


def main(tree_width, seq_len, head_num, feat_dim):
    causal = False
    window_size = -1, -1
    softcap = 0.0
    alibi_slopes = None
    deterministic = False
    return_attn_probs = False
    dropout_p = 0.0
    query_states = (
        paddle.randn(tree_width, 1, head_num, feat_dim).cuda().to(paddle.float16)
    )
    key_states = (
        paddle.randn(tree_width, seq_len, head_num, feat_dim).cuda().to(paddle.float16)
    )
    value_states = (
        paddle.randn(tree_width, seq_len, head_num, feat_dim).cuda().to(paddle.float16)
    )
    from quant.new_pack import (triton_quantize_and_pack_along_last_dim,
                                unpack_and_dequant_kcache,
                                unpack_and_dequant_vcache)

    prepare_k = key_states.transpose(1, 3)
    k_code, k_scale, k_mn = triton_quantize_and_pack_along_last_dim(
        prepare_k, group_size=32, bit=4
    )
    v_code, v_scale, v_mn = triton_quantize_and_pack_along_last_dim(
        value_states, group_size=32, bit=4
    )
    dequant_k = unpack_and_dequant_kcache(k_code, k_scale, k_mn, group_size=32, bits=4)
    dequant_v = unpack_and_dequant_vcache(v_code, v_scale, v_mn, group_size=32, bits=4)
    err_k = (prepare_k - dequant_k).abs().mean()
    err_v = (value_states - dequant_v).abs().mean()
    print(f"average quant error for a single element, K: {err_k}, V: {err_v}")
    softmax_scale = query_states.shape[-1] ** -0.5
    iter_num = 1
    for i in range(1000):
        out1, softmax_lse1, softmax_scale1 = _quantized_flash_attn_forward(
            query_states,
            k_code,
            k_scale,
            k_mn,
            v_code,
            v_scale,
            v_mn,
            group_size=32,
            bits=4,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    st_time1 = time.time()
    for i in range(iter_num):
        out1, softmax_lse1, softmax_scale1 = _quantized_flash_attn_forward(
            query_states,
            k_code,
            k_scale,
            k_mn,
            v_code,
            v_scale,
            v_mn,
            group_size=32,
            bits=4,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    ed_time1 = time.time()
    for i in range(1000):
        (
            out2,
            q2,
            k2,
            v2,
            out_padded,
            softmax_lse2,
            S_dmask,
            rng_state,
>>>>>>        ) = flash_attn.flash_attn_interface._flash_attn_forward(
            query_states,
            dequant_k.transpose(1, 3),
            dequant_v,
            dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=False,
        )
    st_time2 = time.time()
    for i in range(iter_num):
        (
            out2,
            q2,
            k2,
            v2,
            out_padded,
            softmax_lse2,
            S_dmask,
            rng_state,
>>>>>>        ) = flash_attn.flash_attn_interface._flash_attn_forward(
            query_states,
            dequant_k.transpose(1, 3),
            dequant_v,
            dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=False,
        )
    ed_time2 = time.time()
    for i in range(1000):
        (
            out3,
            softmax_lse1,
            softmax_scale1,
>>>>>>        ) = flash_attn.flash_attn_triton._flash_attn_forward(
            query_states,
            dequant_k.transpose(1, 3),
            dequant_v,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    st_time3 = time.time()
    for i in range(iter_num):
        (
            out3,
            softmax_lse1,
            softmax_scale1,
>>>>>>        ) = flash_attn.flash_attn_triton._flash_attn_forward(
            query_states,
            dequant_k.transpose(1, 3),
            dequant_v,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    ed_time3 = time.time()
    err_o = (out1 - out2).abs().mean()
    print("average quant error for a single element: %.6f" % err_o)
    print(
        "time for 4bit triton: %.12f, fp16 cuda: %.12f, fp16 triton: %.12f (repeat %d)"
        % (
            (ed_time1 - st_time1) / iter_num,
            (ed_time2 - st_time2) / iter_num,
            (ed_time3 - st_time3) / iter_num,
            iter_num,
        )
    )


def seed_all(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)
    paddle.seed(seed=seed)
    paddle.seed(seed)
    PaddleFlag.cudnn_benchmark = False
    PaddleFlag.cudnn_deterministic = True


start_values = end_values = 8, 128, 24, 128
increments = 1, 1, 1, 1
seed_all()
for tree_width in range(start_values[0], end_values[0] + 1, increments[0]):
    for seq_len in range(start_values[1], end_values[1] + 1, increments[1]):
        for head_num in range(start_values[2], end_values[2] + 1, increments[2]):
            for feat_dim in range(start_values[3], end_values[3] + 1, increments[3]):
                print(
                    f"tree_width={tree_width}, seq_len={seq_len}, head_num={head_num}, feat_dim={feat_dim}"
                )
                main(tree_width, seq_len, head_num, feat_dim)
