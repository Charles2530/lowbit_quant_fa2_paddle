import random

import numpy as np
import paddle
import triton
import triton.language as tl


def quant_and_pack_qcache(q: paddle.FloatTensor, group_size: int, bits: int):
    assert len(q.shape) == 4
    shape = q.shape
    B, nh, T, D = shape
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = B, nh, num_groups, group_size, D
    max_int = 2**bits - 1
    data = q.view(new_shape)
    mn = paddle.compat.min(data, dim=-2, keepdim=True)[0]
    mx = paddle.compat.max(data, dim=-2, keepdim=True)[0]
    scale = (mx - mn) / max_int
    data = data - mn
    data.div_(scale)
    data = data.clip_(0, max_int).round_().to(paddle.int32)
    data = data.view(shape)
    code = pack_tensor(data, bits, pack_dim=2)
    return code, scale, mn


def quant_and_pack_kcache(k: paddle.FloatTensor, group_size: int, bits: int):
    assert len(k.shape) == 4
    shape = k.shape
    B, nh, T, D = shape
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = B, nh, num_groups, group_size, D
    max_int = 2**bits - 1
    data = k.view(new_shape)
    mn = paddle.compat.min(data, dim=-2, keepdim=True)[0]
    mx = paddle.compat.max(data, dim=-2, keepdim=True)[0]
    scale = (mx - mn) / max_int
    data = data - mn
    data.div_(scale)
    data = data.clip_(0, max_int).round_().to(paddle.int32)
    data = data.view(shape)
    code = pack_tensor(data, bits, pack_dim=2)
    return code, scale, mn


def quant_and_pack_vcache(v: paddle.FloatTensor, group_size: int, bits: int):
    shape = v.shape
    assert len(shape) == 4
    assert v.shape[-1] % group_size == 0
    num_groups = shape[-1] // group_size
    new_shape = shape[:-1] + (num_groups, group_size)
    max_int = 2**bits - 1
    data = v.view(new_shape)
    mn = paddle.compat.min(data, dim=-1, keepdim=True)[0]
    mx = paddle.compat.max(data, dim=-1, keepdim=True)[0]
    scale = (mx - mn) / max_int
    data = data - mn
    data.div_(scale)
    data = data.clip_(0, max_int).round_().to(paddle.int32)
    data = data.view(shape)
    code = pack_tensor(data, bits, pack_dim=3)
    return code, scale, mn


def unpack_and_dequant_kcache(
    k_code: paddle.FloatTensor,
    scale: paddle.FloatTensor,
    mn: paddle.FloatTensor,
    group_size: int,
    bits: int,
):
    pack_dim = 3
    assert bits in [1, 2, 4, 8]
    assert len(k_code.shape) == 4
    data = unpack_tensor(k_code, bits, pack_dim=pack_dim)
    shape = data.shape
    num_groups = shape[pack_dim] // group_size
    data = data.view(
        shape[:pack_dim] + (num_groups, group_size) + shape[pack_dim + 1 :]
    )
    data = data.to(paddle.float16)
    if scale.dim() == 4 and data.dim() == 5:
        scale = scale.unsqueeze(pack_dim - 4)
        mn = mn.unsqueeze(pack_dim - 4)
    try:
        data = data * scale + mn
    except:
        import pdb

        pdb.set_trace()
    return data.view(shape)


def unpack_and_dequant_qcache(
    q_code: paddle.FloatTensor,
    scale: paddle.FloatTensor,
    mn: paddle.FloatTensor,
    group_size: int,
    bits: int,
):
    pack_dim = 3
    assert bits in [1, 2, 4, 8]
    assert len(q_code.shape) == 4
    data = unpack_tensor(q_code, bits, pack_dim=pack_dim)
    shape = data.shape
    num_groups = shape[pack_dim] // group_size
    data = data.view(
        shape[:pack_dim] + (num_groups, group_size) + shape[pack_dim + 1 :]
    )
    data = data.to(paddle.float16)
    if scale.dim() == 4 and data.dim() == 5:
        scale = scale.unsqueeze(pack_dim - 4)
        mn = mn.unsqueeze(pack_dim - 4)
    try:
        data = data * scale + mn
    except:
        import pdb

        pdb.set_trace()
    return data.view(shape)


def unpack_and_dequant_vcache(
    v_code: paddle.FloatTensor,
    scale: paddle.FloatTensor,
    mn: paddle.FloatTensor,
    group_size: int,
    bits: int,
):
    assert bits in [2, 4, 8]
    assert len(v_code.shape) == 4
    data = unpack_tensor(v_code, bits, pack_dim=3)
    shape = data.shape
    num_groups = shape[-1] // group_size
    data = data.view(shape[:-1] + (num_groups, group_size))
    data = data.to(paddle.float16)
    if scale.dim() < data.dim():
        scale = scale.unsqueeze(-1)
        mn = mn.unsqueeze(-1)
    data = data * scale + mn
    return data.view(shape)


def pack_tensor(data, bits, pack_dim):
    shape = data.shape
    feat_per_int = 32 // bits
    assert bits in [2, 4, 8], "Only 2, 4, 8 bits are supported"
    assert (
        shape[pack_dim] % feat_per_int == 0
    ), "Dimension length must be divisible by number of features per int"
    code = paddle.zeros(
        shape[:pack_dim] + (shape[pack_dim] // feat_per_int,) + shape[pack_dim + 1 :],
        dtype=paddle.int32,
        device=data.place,
    )
    i = 0
    row = 0
    unpacked_indices = [slice(None)] * len(data.shape)
    packed_indices = [slice(None)] * len(data.shape)
    while row < code.shape[pack_dim]:
        packed_indices[pack_dim] = row
        for j in range(i, i + 32 // bits):
            unpacked_indices[pack_dim] = j
            code[packed_indices] |= data[unpacked_indices] << bits * (j - i)
        i += 32 // bits
        row += 1
    return code


def unpack_tensor(code: paddle.FloatTensor, bits: int, pack_dim: int):
    assert bits in [1, 2, 4, 8]
    shape = code.shape
    feat_per_int = 8 // bits
    new_shape = (
        shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim + 1 :]
    )
    unpacked_v_code = paddle.zeros(new_shape, dtype=paddle.int8, device=code.place)
    i = paddle.arange(new_shape[pack_dim], device=code.place) // feat_per_int
    j = paddle.arange(new_shape[pack_dim], device=code.place) % feat_per_int
    num = 255 >> 8 - bits
    packed_indices = [slice(None)] * len(new_shape)
    packed_indices[pack_dim] = i
    code = code.to(paddle.int16)
    if pack_dim == 2:
        unpacked_v_code = (code[packed_indices] >> (j * bits)[None, None, :, None]).to(
            paddle.int16
        ) & num
    elif pack_dim == 3:
        unpacked_v_code = (code[packed_indices] >> j * bits).to(paddle.int16) & num
    else:
        raise NotImplementedError
    return unpacked_v_code.to(paddle.int8)


@triton.jit
def _pack_along_last_dim(
    bits: tl.constexpr,
    intensor_ptr,
    code_ptr,
    N,
    num_feats: tl.constexpr,
    feat_per_int: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    num_int_per_y_dim = num_feats // feat_per_int
    bid = tl.program_id(axis=0)
    yid = tl.program_id(axis=1)
    offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int
    packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
    for i in range(feat_per_int):
        ptr = block_start + i
        element = tl.load(ptr, mask=offs_N < N, other=0.0)
        element = element << i * bits
        packed = packed | element
    tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)


@triton.jit
def _minmax_along_last_dim(
    x_ptr,
    mn_ptr,
    mx_ptr,
    total_elements: tl.constexpr,
    N: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    _minmax_along_last_dim: compute the min and max of x along the last dimension
    """
    bid = tl.program_id(axis=0)
    offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
    mask = offsets < total_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    mx_val = tl.max(x, axis=1)
    mn_val = tl.min(x, axis=1)
    tl.store(mn_ptr + offsets_b, mn_val, mask=offsets_b < N * num_groups)
    tl.store(mx_ptr + offsets_b, mx_val, mask=offsets_b < N * num_groups)


def triton_quantize_and_pack_along_last_dim(
    data: paddle.Tensor, group_size: int, bit: int
):
    assert len(data.shape) == 4
    shape = data.shape
    B, D, nh, T = shape
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = B * nh * D, num_groups, group_size
    scale_mn_shape = B, D, nh, num_groups
    data = data.reshape(new_shape)
    mx = paddle.empty((B * nh * D, num_groups), device=data.place, dtype=data.dtype)
    mn = paddle.empty((B * nh * D, num_groups), device=data.place, dtype=data.dtype)
    BLOCK_SIZE_N = 128
    grid = lambda meta: (triton.cdiv(data.shape[0] * data.shape[1], BLOCK_SIZE_N),)
    _minmax_along_last_dim[grid](
        data,
        mn,
        mx,
        data.size,
        data.shape[0],
        num_groups,
        group_size,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=8,
    )
    scale = (mx - mn) / (2**bit - 1)
    data = data - mn.unsqueeze(-1)
    data.div_(other=scale.unsqueeze(-1))
    data = data.clip_(0, 2**bit - 1).round_().to(paddle.int32)
    data = data.view(-1, T)
    feat_per_int = 8 // bit
    packshape = np.prod(shape[:-1]), shape[-1] // feat_per_int
    code = paddle.zeros(*packshape, device=data.place, dtype=paddle.int8)
    grid = lambda meta: (
        triton.cdiv(data.shape[0], BLOCK_SIZE_N),
        data.shape[1] // feat_per_int,
    )
    import pdb

    pdb.set_trace()
    _pack_along_last_dim[grid](
        bit,
        data,
        code,
        data.shape[0],
        data.shape[1],
        feat_per_int,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=8,
    )
    return (
        code.view(B, D, nh, -1),
        scale.reshape(scale_mn_shape),
        mn.reshape(shape=scale_mn_shape),
    )
