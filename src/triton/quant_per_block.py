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
import triton
import triton.language as tl


@triton.jit
def quant_per_block_int4_unpack_kernel(
    Input,
    Output,
    Scale,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_oz,
    stride_oh,
    stride_on,
    stride_sz,
    stride_sh,
    sm_scale,
    C: tl.constexpr,
    BLK: tl.constexpr,
):
    """
    only quantize to int4 but not pack to int8(correct)
    """
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    input_ptrs = (
        Input
        + off_b * stride_iz
        + off_h * stride_ih
        + offs_n[:, None] * stride_in
        + offs_k[None, :]
    )
    output_ptrs = (
        Output
        + off_b * stride_oz
        + off_h * stride_oh
        + offs_n[:, None] * stride_on
        + offs_k[None, :]
    )
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk
    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 7.0
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


@triton.jit
def quant_per_block_int4_kernel(
    Input,
    Output,
    Scale,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_oz,
    stride_oh,
    stride_on,
    stride_sz,
    stride_sh,
    sm_scale,
    C: tl.constexpr,
    BLK: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    input_ptrs = (
        Input
        + off_b * stride_iz
        + off_h * stride_ih
        + offs_n[:, None] * stride_in
        + offs_k[None, :]
    )
    output_ptrs = (
        Output
        + off_b * stride_oz
        + off_h * stride_oh
        + offs_n[:, None] * stride_on
        + offs_k[None, :]
    )
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk
    x_high_4 = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x_low_4 = tl.load(input_ptrs + 1, mask=offs_n[:, None] < L)
    x_high_4 = x_high_4.to(tl.float32)
    x_low_4 = x_low_4.to(tl.float32)
    scale = tl.where(
        tl.max(tl.abs(x_high_4)) > tl.max(tl.abs(x_low_4)),
        tl.max(tl.abs(x_high_4)) / 7.0,
        tl.max(tl.abs(x_low_4)) / 7.0,
    )
    x_high_4_int4 = x_high_4 / scale
    x_high_4_int4 += 0.5 * tl.where(x_high_4_int4 >= 0, 1, -1)
    x_low_4_int4 = x_low_4 / scale
    x_low_4_int4 += 0.5 * tl.where(x_low_4_int4 >= 0, 1, -1)
    x_high_4_int4 = x_high_4_int4.to(tl.int8)
    x_low_4_int4 = x_low_4_int4.to(tl.int8)
    x_int4 = (x_high_4_int4 << 4) + x_low_4_int4
    x_int4 = x_int4.to(tl.int8)
    tl.store(output_ptrs, x_int4, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


@triton.jit
def quant_per_block_int8_kernel(
    Input,
    Output,
    Scale,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_oz,
    stride_oh,
    stride_on,
    stride_sz,
    stride_sh,
    sm_scale,
    C: tl.constexpr,
    BLK: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    input_ptrs = (
        Input
        + off_b * stride_iz
        + off_h * stride_ih
        + offs_n[:, None] * stride_in
        + offs_k[None, :]
    )
    output_ptrs = (
        Output
        + off_b * stride_oz
        + off_h * stride_oh
        + offs_n[:, None] * stride_on
        + offs_k[None, :]
    )
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk
    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 127.0
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

def check_strides(tensor,strides):
    import torch
    tensor = torch.tensor(tensor.numpy())
    if tensor.stride(0) != strides[0] or tensor.stride(1) != strides[1] or tensor.stride(2) != strides[2]:
        print(tensor.stride(0), tensor.stride(1), tensor.stride(2))
        print(strides[0], strides[1], strides[2])
        import pdb; pdb.set_trace()
        return False
    return True

def get_strides(tensor):
    strides = []
    if tensor.dtype == paddle.float32:
        strides = [i//4 for i in tensor.numpy().strides]
    elif tensor.dtype == paddle.int8:
        strides = [i for i in tensor.numpy().strides]
    elif tensor.dtype == paddle.float16 or tensor.dtype == paddle.bfloat16:
        strides = [i//2 for i in tensor.numpy().strides]
    else:
        raise ValueError(f"Unknown tensor dtype: {tensor.dtype}")
    # return strides
    if(check_strides(tensor,strides)):
        return strides
    else:
        import pdb; pdb.set_trace()
        return None


def per_block_int8(
    q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"
):
    q_int8 = paddle.empty(q.shape, dtype=paddle.int8).to(q.place)
    k_int8 = paddle.empty(k.shape, dtype=paddle.int8).to(k.place)
    if km is not None:
        k = k - km
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        q_strides = get_strides(q)
        k_strides = get_strides(k)
        q_int8_strides = get_strides(q_int8)
        k_int8_strides = get_strides(k_int8)
        # import torch
        # q =torch.tensor(q.numpy())
        # print(q.stride(0), q.stride(1), q.stride(2))
        # print(q_strides[0], q_strides[1], q_strides[2])
        # import pdb; pdb.set_trace()
        stride_bz_q, stride_h_q, stride_seq_q = q_strides[0], q_strides[1], q_strides[2]
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8_strides[0], q_int8_strides[1], q_int8_strides[2]
        stride_bz_k, stride_h_k, stride_seq_k = k_strides[0], k_strides[1], k_strides[2]
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8_strides[0], k_int8_strides[1], k_int8_strides[2]
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        q_strides = [i//2 for i in q.numpy().strides]
        k_strides = [i//2 for i in k.numpy().strides]
        q_int8_strides = [i//2 for i in q_int8.numpy().strides]
        k_int8_strides = [i//2 for i in k_int8.numpy().strides]
        stride_bz_q, stride_h_q, stride_seq_q = q_strides[0], q_strides[2], q_strides[1]
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8_strides[0], q_int8_strides[2], q_int8_strides[1]
        stride_bz_k, stride_h_k, stride_seq_k = k_strides[0], k_strides[2], k_strides[1]
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8_strides[0], k_int8_strides[2], k_int8_strides[1]
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    q_scale = paddle.empty(
        (b, h_qo, (qo_len + BLKQ - 1) // BLKQ), dtype=paddle.float32
    ).to(q.place)
    k_scale = paddle.empty(
        (b, h_kv, (kv_len + BLKK - 1) // BLKK), dtype=paddle.float32
    ).to(q.place)
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    grid = (qo_len + BLKQ - 1) // BLKQ, h_qo, b
    q_scale_strides = get_strides(q_scale)
    k_scale_strides =   get_strides(k_scale)
    quant_per_block_int8_kernel[grid](
        q,
        q_int8,
        q_scale,
        qo_len,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_qo,
        stride_h_qo,
        stride_seq_qo,
        q_scale_strides[0],
        q_scale_strides[1],
        sm_scale=sm_scale * 1.44269504,
        C=head_dim,
        BLK=BLKQ,
    )
    grid = (kv_len + BLKK - 1) // BLKK, h_kv, b
    quant_per_block_int8_kernel[grid](
        k,
        k_int8,
        k_scale,
        kv_len,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_ko,
        stride_h_ko,
        stride_seq_ko,
        k_scale_strides[0],
        k_scale_strides[1],
        sm_scale=1.0,
        C=head_dim,
        BLK=BLKK,
    )
    return q_int8, q_scale, k_int8, k_scale


def per_block_int4_unpack(
    q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"
):
    q_int8 = paddle.empty(q.shape, dtype=paddle.int8, device=q.place)
    k_int8 = paddle.empty(k.shape, dtype=paddle.int8, device=k.place)
    if km is not None:
        k = k - km
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        q_strides = q.numpy().strides
        k_strides = k.numpy().strides
        q_int8_strides = q_int8.numpy().strides
        k_int8_strides = k_int8.numpy().strides
        stride_bz_q, stride_h_q, stride_seq_q = q_strides[0], q_strides[1], q_strides[2]
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8_strides[0], q_int8_strides[1], q_int8_strides[2]
        stride_bz_k, stride_h_k, stride_seq_k = k_strides[0], k_strides[1], k_strides[2]
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8_strides[0], k_int8_strides[1], k_int8_strides[2]
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        q_strides = q.numpy().strides
        k_strides = k.numpy().strides
        q_int8_strides = q_int8.numpy().strides
        k_int8_strides = k_int8.numpy().strides
        stride_bz_q, stride_h_q, stride_seq_q = q_strides[0], q_strides[2], q_strides[1]
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8_strides[0], q_int8_strides[2], q_int8_strides[1]
        stride_bz_k, stride_h_k, stride_seq_k = k_strides[0], k_strides[2], k_strides[1]
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8_strides[0], k_int8_strides[2], k_int8_strides[1]
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    q_scale = paddle.empty(
        (b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.place, dtype=paddle.float32
    )
    k_scale = paddle.empty(
        (b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.place, dtype=paddle.float32
    )
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    grid = (qo_len + BLKQ - 1) // BLKQ, h_qo, b
    quant_per_block_int4_unpack_kernel[grid](
        q,
        q_int8,
        q_scale,
        qo_len,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_qo,
        stride_h_qo,
        stride_seq_qo,
        q_scale.stride(0),
        q_scale.stride(1),
        sm_scale=sm_scale * 1.44269504,
        C=head_dim,
        BLK=BLKQ,
    )
    grid = (kv_len + BLKK - 1) // BLKK, h_kv, b
    quant_per_block_int4_unpack_kernel[grid](
        k,
        k_int8,
        k_scale,
        kv_len,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_ko,
        stride_h_ko,
        stride_seq_ko,
        k_scale.stride(0),
        k_scale.stride(1),
        sm_scale=1.0,
        C=head_dim,
        BLK=BLKK,
    )
    return q_int8, q_scale, k_int8, k_scale


def per_block_int4(
    q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"
):
    q_int8 = paddle.empty(q.shape, dtype=paddle.int8, device=q.place)
    k_int8 = paddle.empty(k.shape, dtype=paddle.int8, device=k.place)
    if km is not None:
        k = k - km
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        q_strides = q.numpy().strides
        k_strides = k.numpy().strides
        q_int8_strides = q_int8.numpy().strides
        k_int8_strides = k_int8.numpy().strides
        stride_bz_q, stride_h_q, stride_seq_q = q_strides[0], q_strides[1], q_strides[2]
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8_strides[0], q_int8_strides[1], q_int8_strides[2]
        stride_bz_k, stride_h_k, stride_seq_k = k_strides[0], k_strides[1], k_strides[2]
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8_strides[0], k_int8_strides[1], k_int8_strides[2]
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        q_strides = q.numpy().strides
        k_strides = k.numpy().strides
        q_int8_strides = q_int8.numpy().strides
        k_int8_strides = k_int8.numpy().strides
        stride_bz_q, stride_h_q, stride_seq_q = q_strides[0], q_strides[2], q_strides[1]
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8_strides[0], q_int8_strides[2], q_int8_strides[1]
        stride_bz_k, stride_h_k, stride_seq_k = k_strides[0], k_strides[2], k_strides[1]
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8_strides[0], k_int8_strides[2], k_int8_strides[1]
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    q_scale = paddle.empty(
        (b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.place, dtype=paddle.float32
    )
    k_scale = paddle.empty(
        (b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.place, dtype=paddle.float32
    )
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    grid = (qo_len + BLKQ - 1) // BLKQ // 2, h_qo, b
    quant_per_block_int4_kernel[grid](
        q,
        q_int8,
        q_scale,
        qo_len,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_qo,
        stride_h_qo,
        stride_seq_qo,
        q_scale.stride(0),
        q_scale.stride(1),
        sm_scale=sm_scale * 1.44269504,
        C=head_dim,
        BLK=BLKQ * 2,
    )
    grid = (kv_len + BLKK - 1) // BLKK // 2, h_kv, b
    quant_per_block_int4_kernel[grid](
        k,
        k_int8,
        k_scale,
        kv_len,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_ko,
        stride_h_ko,
        stride_seq_ko,
        k_scale.stride(0),
        k_scale.stride(1),
        sm_scale=1.0,
        C=head_dim,
        BLK=BLKK,
    )
    return q_int8, q_scale, k_int8, k_scale


def per_block_q_int8_k_int4(
    q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"
):
    q_int8 = paddle.empty(q.shape, dtype=paddle.int8, device=q.place)
    k_int8 = paddle.empty(k.shape, dtype=paddle.int8, device=k.place)
    if km is not None:
        k = k - km
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        q_strides = q.numpy().strides
        k_strides = k.numpy().strides
        q_int8_strides = q_int8.numpy().strides
        k_int8_strides = k_int8.numpy().strides
        stride_bz_q, stride_h_q, stride_seq_q = q_strides[0], q_strides[1], q_strides[2]
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8_strides[0], q_int8_strides[1], q_int8_strides[2]
        stride_bz_k, stride_h_k, stride_seq_k = k_strides[0], k_strides[1], k_strides[2]
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8_strides[0], k_int8_strides[1], k_int8_strides[2]
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        q_strides = q.numpy().strides
        k_strides = k.numpy().strides
        q_int8_strides = q_int8.numpy().strides
        k_int8_strides = k_int8.numpy().strides   
        stride_bz_q, stride_h_q, stride_seq_q = q_strides[0], q_strides[2], q_strides[1]
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8_strides[0], q_int8_strides[2], q_int8_strides[1]
        stride_bz_k, stride_h_k, stride_seq_k = k_strides[0], k_strides[2], k_strides[1]
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8_strides[0], k_int8_strides[2], k_int8_strides[1]
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    q_scale = paddle.empty(
        (b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.place, dtype=paddle.float32
    )
    k_scale = paddle.empty(
        (b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.place, dtype=paddle.float32
    )
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    grid = (qo_len + BLKQ - 1) // BLKQ, h_qo, b
    quant_per_block_int8_kernel[grid](
        q,
        q_int8,
        q_scale,
        qo_len,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_qo,
        stride_h_qo,
        stride_seq_qo,
        q_scale.stride(0),
        q_scale.stride(1),
        sm_scale=sm_scale * 1.44269504,
        C=head_dim,
        BLK=BLKQ * 2,
    )
    grid = (kv_len + BLKK - 1) // BLKK // 2, h_kv, b
    quant_per_block_int4_kernel[grid](
        k,
        k_int8,
        k_scale,
        kv_len,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_ko,
        stride_h_ko,
        stride_seq_ko,
        k_scale.stride(0),
        k_scale.stride(1),
        sm_scale=1.0,
        C=head_dim,
        BLK=BLKK,
    )
    return q_int8, q_scale, k_int8, k_scale
