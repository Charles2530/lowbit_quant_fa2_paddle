import paddle

"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import math
from pickletools import int4

import triton
import triton.language as tl
from src.triton.utils.quant.new_pack import (unpack_and_dequant_kcache,
                                             unpack_and_dequant_qcache)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_scale,
    kv_len,
    K_ptrs,
    K_scale_ptr,
    V_ptrs,
    stride_kn,
    stride_vn,
    start_m,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < kv_len - start_n
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        q_low = q & 15
        q_high = q >> 4
        k_low = k & 15
        k_high = k >> 4
        qk = (
            (
                tl.dot(q_high, k_high)
                + tl.dot(q_high, k_low)
                + tl.dot(q_low, k_high)
                + tl.dot(q_low, k_low)
            ).to(tl.float32)
            * q_scale
            * k_scale
        )
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask=offs_n[:, None] < kv_len - start_n)
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    Out,
    Lse,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_on,
    qo_len,
    kv_len,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(
        kv_len, BLOCK_N
    )
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = (
        Q
        + (off_z * stride_qz + off_h * stride_qh)
        + offs_m[:, None] * stride_qn
        + offs_k[None, :]
    )
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = (
        K
        + (off_z * stride_kz + off_h // num_kv_groups * stride_kh)
        + offs_n[None, :] * stride_kn
        + offs_k[:, None]
    )
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = (
        V
        + (off_z * stride_vz + off_h // num_kv_groups * stride_vh)
        + offs_n[:, None] * stride_vn
        + offs_k[None, :]
    )
    O_block_ptr = (
        Out
        + (off_z * stride_oz + off_h * stride_oh)
        + offs_m[:, None] * stride_on
        + offs_k[None, :]
    )
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        q_scale,
        kv_len,
        K_ptrs,
        K_scale_ptr,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        4 - STAGE,
        offs_m,
        offs_n,
    )
    o_scale = 1.0 / l_i
    acc = acc * o_scale[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=offs_m[:, None] < qo_len)
    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i_log = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i_log, mask=offs_m < qo_len)


def forward(
    qcode,
    qscale,
    qmn,
    kcode,
    kscale,
    kmn,
    v,
    tensor_layout="HND",
    output_dtype=paddle.float16,
    return_lse=False,
    group_size=32,
    bits=4,
):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 1
    o = paddle.empty(qcode.shape, dtype=output_dtype, device=qcode.place)
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = qcode.shape
        _, h_kv, kv_len, _ = kcode.shape
        stride_bz_q, stride_h_q, stride_seq_q = (
            qcode.stride(0),
            qcode.stride(1),
            qcode.stride(2),
        )
        stride_bz_k, stride_h_k, stride_seq_k = (
            kcode.stride(0),
            kcode.stride(1),
            kcode.stride(2),
        )
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = qcode.shape
        _, kv_len, h_kv, _ = kcode.shape
        stride_bz_q, stride_h_q, stride_seq_q = (
            qcode.stride(0),
            qcode.stride(2),
            qcode.stride(1),
        )
        stride_bz_k, stride_h_k, stride_seq_k = (
            kcode.stride(0),
            kcode.stride(2),
            kcode.stride(1),
        )
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    assert qo_len == kv_len, "qo_len and kv_len must be equal for causal attention"
    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv
    if return_lse:
        lse = paddle.empty([b, h_qo, qo_len], dtype=paddle.float32, device=q.place)
    else:
        lse = paddle.empty([0], dtype=paddle.float32, device="cpu")
    grid = triton.cdiv(qo_len, BLOCK_M), h_qo, b
    _attn_fwd[grid](
        qcode,
        kcode,
        v,
        qscale,
        kscale,
        o,
        lse,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_bz_o,
        stride_h_o,
        stride_seq_o,
        qo_len,
        kv_len,
        h_qo,
        num_kv_groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4,
    )
    return o, lse


def forward_merging(
    qcode,
    qscale,
    kcode,
    kscale,
    v,
    tensor_layout="HND",
    output_dtype=paddle.float16,
    return_lse=False,
    group_size=32,
    bits=4,
):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 1
    o = paddle.empty(qcode.shape, dtype=output_dtype, device=qcode.place)
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = qcode.shape
        _, h_kv, kv_len, _ = kcode.shape
        stride_bz_q, stride_h_q, stride_seq_q = (
            qcode.stride(0),
            qcode.stride(1),
            qcode.stride(2),
        )
        stride_bz_k, stride_h_k, stride_seq_k = (
            kcode.stride(0),
            kcode.stride(1),
            kcode.stride(2),
        )
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = qcode.shape
        _, kv_len, h_kv, _ = kcode.shape
        stride_bz_q, stride_h_q, stride_seq_q = (
            qcode.stride(0),
            qcode.stride(2),
            qcode.stride(1),
        )
        stride_bz_k, stride_h_k, stride_seq_k = (
            kcode.stride(0),
            kcode.stride(2),
            kcode.stride(1),
        )
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv
    if return_lse:
        lse = paddle.empty([b, h_qo, qo_len], dtype=paddle.float32, device=qcode.place)
    else:
        lse = paddle.empty([0], dtype=paddle.float32, device="cpu")
    grid = triton.cdiv(qo_len, BLOCK_M), h_qo, b
    _attn_fwd[grid](
        qcode,
        kcode,
        v,
        qscale,
        kscale,
        o,
        lse,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_bz_o,
        stride_h_o,
        stride_seq_o,
        qo_len,
        kv_len,
        H=h_qo,
        num_kv_groups=num_kv_groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4,
    )
    return o, lse
