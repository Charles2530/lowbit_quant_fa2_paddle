import paddle
import triton
import triton.language as tl
from src.triton.utils.quant.new_pack import unpack_tensor


def unpack_and_dequant_ocache(
    o_code: paddle.FloatTensor,
    scale: paddle.FloatTensor,
    mn: paddle.FloatTensor,
    group_size: int,
    bits: int,
):
    pack_dim = 3
    assert bits in [1, 2, 4, 8]
    assert len(o_code.shape) == 4
    data = unpack_tensor(code=o_code, bits=bits, pack_dim=pack_dim)
    shape = data.shape
    num_groups = shape[pack_dim] // group_size
    data = data.view(
        size=shape[:pack_dim] + (num_groups, group_size) + shape[pack_dim + 1 :]
    )
    data = data.to(dtype=paddle.float16)
    if scale.dim() == 4 and data.dim() == 5:
        scale = scale.unsqueeze(pack_dim - 4)
        mn = mn.unsqueeze(pack_dim - 4)
    try:
        data = data * scale + mn
    except:
        import pdb

        pdb.set_trace()
    return data.view(shape)


def unpack_and_dequant(
    o_code: paddle.FloatTensor,
    scale: paddle.FloatTensor,
    mn: paddle.FloatTensor,
    group_size: int,
    bits: int,
):
    unpacked = paddle.zeros((*o_code.shape, 2), dtype=paddle.int8)
    unpacked[..., 0] = o_code >> 4
    unpacked[..., 1] = o_code & 15
    unpacked = (unpacked.to(paddle.int8) << 8 - bits).to(paddle.int8) >> 8 - bits
    return unpacked * scale.unsqueeze(-1) + mn.unsqueeze(-1)
