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
import imp
import math
from struct import unpack
from turtle import pd

import triton
import triton.language as tl
from utils.utils import (unpack_and_dequant, unpack_and_dequant_ocache,
                         unpack_tensor)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    kv_len,
    K_ptrs,
    K_scale_ptr,
    K_mn_ptr,
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
    group_size: tl.constexpr,
    bits: tl.constexpr,
    feat_per_int: tl.constexpr,
    q_scale: tl.constexpr,
    k_scale: tl.constexpr,
):
    lo, hi = 0, kv_len
    num = (1 << bits) - 1
    meta_dtype = tl.float32
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < kv_len - start_n
        k_code = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        k_mn = tl.load(K_mn_ptr)
        k = tl.zeros([BLOCK_N, HEAD_DIM], dtype=meta_dtype)
        tl.static_print(BLOCK_N)
        for idx in range(feat_per_int):
            shift = idx * bits
            t_k = k_code >> shift & num
            t_k = tl.fma(t_k, k_scale, k_mn)
            dim_idx = tl.arange(0, HEAD_DIM) // feat_per_int * feat_per_int + idx
            dim_idx = tl.reshape(dim_idx, (1, HEAD_DIM))
            dim_idx = tl.broadcast_to(dim_idx, t_k.shape)
            k += tl.where(dim_idx < HEAD_DIM, t_k, 0)
        qk = tl.dot(q, tl.trans(k)).to(meta_dtype)
        qk = qk * q_scale * k_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask=offs_n[:, None] < kv_len - start_n)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        K_mn_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["qo_len"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["kv_len"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["H"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _attn_fwd(
    Q,
    Qscale,
    Qmn,
    K,
    Kscale,
    Kmn,
    V,
    group_size: tl.constexpr,
    bits: tl.constexpr,
    feat_per_int: tl.constexpr,
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
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2)
    off_h = tl.program_id(1)
    num = (1 << bits) - 1
    meta_dtype = tl.float32
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_HEADDIM)
    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(
        kv_len, BLOCK_N
    )
    Q_ptrs = (
        Q
        + (off_z * stride_qz + off_h * stride_qh)
        + offs_m[:, None] * stride_qn
        + offs_k[None, :] // feat_per_int
    )
    q_code = tl.load(
        Q_ptrs, mask=(offs_m[:, None] < qo_len) & (offs_k[None, :] < BLOCK_HEADDIM)
    )
    q_scale = tl.load(Qscale + q_scale_offset + start_m)
    q_mn = tl.load(Qmn + q_scale_offset + start_m)
    q = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=meta_dtype)
    for idx in range(feat_per_int):
        shift = idx * bits
        t_q = q_code >> shift & num
        t_q = t_q.to(meta_dtype) * q_scale + q_mn
        dim_idx = offs_k // feat_per_int * feat_per_int + idx
        q += tl.where(
            (offs_m[:, None] < qo_len) & (dim_idx[None, :] < BLOCK_HEADDIM), t_q, 0
        )
    K_ptrs = (
        K
        + (off_z * stride_kz + off_h // num_kv_groups * stride_kh)
        + offs_n[None, :] * stride_kn
        + offs_k[:, None] // feat_per_int
    )
    V_ptrs = (
        V
        + (off_z * stride_vz + off_h // num_kv_groups * stride_vh)
        + offs_n[:, None] * stride_vn
        + offs_k[None, :]
    )
    K_scale_ptr = Kscale + k_scale_offset
    K_mn_ptr = Kmn + k_scale_offset
    m_i = tl.zeros([BLOCK_M], dtype=meta_dtype) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=meta_dtype) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=meta_dtype)
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        kv_len,
        K_ptrs,
        K_scale_ptr,
        K_mn_ptr,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        BLOCK_M,
        BLOCK_HEADDIM,
        BLOCK_N,
        0,
        offs_m,
        offs_n,
        group_size,
        bits,
        feat_per_int,
        q_scale,
        1.0,
    )
    o_scale = 1.0 / l_i
    acc = acc * o_scale[:, None]
    tl.store(
        Out
        + (off_z * stride_oz + off_h * stride_oh)
        + offs_m[:, None] * stride_on
        + offs_k[None, :],
        acc.to(Out.type.element_ty),
        mask=offs_m[:, None] < qo_len,
    )
    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        tl.store(lse_ptrs, tl.log2(l_i) + m_i, mask=offs_m < qo_len)


def forward(
    q,
    k,
    v,
    q_scale,
    k_scale,
    tensor_layout="HND",
    output_dtype=paddle.int8,
    return_lse=False,
):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 1
    o = paddle.empty(q.shape, dtype=output_dtype, device=q.place)
    o_scale = paddle.empty(q_scale.shape, dtype=paddle.float16, device=q.place)
    o_mn = paddle.empty(q_scale.shape, dtype=paddle.float16, device=q.place)
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
        stride_osz, stride_osh, stride_osn = (
            o_scale.stride(0),
            o_scale.stride(1),
            o_scale.stride(2),
        )
        stride_omz, stride_omh, stride_omn = (
            o_mn.stride(0),
            o_mn.stride(1),
            o_mn.stride(2),
        )
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
        stride_osz, stride_osh, stride_osn = (
            o_scale.stride(0),
            o_scale.stride(1),
            o_scale.stride(2),
        )
        stride_omz, stride_omh, stride_omn = (
            o_mn.stride(0),
            o_mn.stride(1),
            o_mn.stride(2),
        )
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv
    if return_lse:
        lse = paddle.empty([b, h_qo, qo_len], dtype=paddle.float32, device=q.place)
    else:
        lse = paddle.empty([0], dtype=paddle.float32, device="cpu")
    grid = triton.cdiv(qo_len, BLOCK_M), h_qo, b
    _attn_fwd[grid](
        q,
        k,
        v,
        q_scale,
        k_scale,
        o,
        lse,
        o_scale,
        o_mn,
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
        stride_osz,
        stride_osh,
        stride_osn,
        stride_omz,
        stride_omh,
        stride_omn,
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
    o = unpack_and_dequant_ocache(o, o_scale, o_mn, group_size=32, bits=4)
    return o, lse, o_scale, o_mn


def forward_merging(
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
    o = paddle.empty_like(v)
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
    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
    BLOCK = 1024
    num_kv_groups = h_qo // h_kv
    bits_per_int = 32
    feat_per_int = int(bits_per_int / bits)
    if return_lse:
        lse = paddle.empty([b, h_qo, qo_len], dtype=paddle.float32, device=qcode.place)
    else:
        lse = paddle.empty([0], dtype=paddle.float32, device="cpu")
    grid = lambda META: (triton.cdiv(qo_len, META["BLOCK_M"]), h_qo, b)
    _attn_fwd[grid](
        qcode,
        qscale,
        qmn,
        kcode,
        kscale,
        kmn,
        v,
        bits,
        feat_per_int,
        group_size,
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
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4,
    )
    return o, lse
