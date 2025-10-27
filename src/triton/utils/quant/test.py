import os

import paddle

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
import random

import numpy as np
import triton
from matmul import triton_bmm_fA_qB_outer
from new_pack import (quant_and_pack_kcache, quant_and_pack_vcache,
                      triton_quantize_and_pack_along_last_dim,
                      unpack_and_dequant_kcache, unpack_and_dequant_vcache)
from timeit_v2 import py_benchmark


def set_seed(seed):
    np.random.seed(seed)
    paddle.manual_seed(seed)
    random.seed(seed)


def test_vcache():
    paddle.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    B, nh, T, hd = 555, 32, 433, 128
    v = paddle.randn((B, nh, T, hd), device="cuda", dtype=paddle.float16)
    group_size = 64
    for bits in [2, 4, 8]:
        code, scale, mn = triton_quantize_and_pack_along_last_dim(v, group_size, bits)
        dequant_v = unpack_and_dequant_vcache(
            code, scale.unsqueeze(-1), mn.unsqueeze(-1), group_size, bits
        )
        assert not dequant_v.isnan().any()
        gap = (dequant_v - v) / v
        gap = paddle.nan_to_num(x=gap)
        print(f"bit {bits}, mean v rel arr: {paddle.mean(paddle.abs(x=gap))}")


def test_kcache():
    paddle.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    BS, nh, T, D = 11, 32, 4096, 128
    k = paddle.randn((BS, nh, T, D), device="cuda", dtype=paddle.float16)
    group_size = 64
    for bits in [2, 4, 8]:
        code, scale, mn = triton_quantize_and_pack_along_last_dim(
            k.transpose(2, 3).contiguous(), group_size, bits
        )
        dequant_k = unpack_and_dequant_vcache(
            code, scale.unsqueeze(-1), mn.unsqueeze(-1), group_size, bits
        )
        assert not dequant_k.isnan().any()
        gap = (dequant_k.transpose(2, 3) - k) / k
        gap = paddle.nan_to_num(x=gap)
        print(f"bit {bits}, k mean rel arr: {paddle.mean(paddle.abs(x=gap))}")


def test_bmm_speed():
    BS, nh, T, D = 64, 32, 512, 128
    bits = 2
    key_state = paddle.randn((BS, nh, T, D), device="cuda", dtype=paddle.float16)
    val_state = paddle.randn((BS, nh, T, D), device="cuda", dtype=paddle.float16)
    group_size = 64
    query_len = 1
    query_state = paddle.randn(
        (BS, nh, query_len, D), device="cuda", dtype=paddle.float16
    )
    code, scale, mn = triton_quantize_and_pack_along_last_dim(
        key_state.transpose(2, 3).contiguous(), group_size, bits
    )
    code = code.contiguous()
    scale = scale.contiguous()
    mn = mn.contiguous()
    stmt = "triton_quantize_and_pack_along_last_dim(key_state.transpose(2,3).contiguous(), group_size, bits)"
    t_triton_quant = py_benchmark(
        stmt,
        {**globals(), **locals()},
        min_repeat_second=3,
        setup="torch.cuda.synchronize()",
        finish="torch.cuda.synchronize()",
    )
    print(f"our triton quant & pack impl: {t_triton_quant * 1000} ms")
    stmt = "quant_and_pack_kcache(key_state, group_size, bits)"
    t_quant = py_benchmark(
        stmt,
        {**globals(), **locals()},
        min_repeat_second=3,
        setup="torch.cuda.synchronize()",
        finish="torch.cuda.synchronize()",
    )
    print(f"vanilla pytorch quant & pack impl: {t_quant * 1000} ms")
    stmt = "triton_bmm_fA_qB_outer(group_size, query_state, code, scale, mn, bits)"
    t_qk = py_benchmark(
        stmt,
        {**globals(), **locals()},
        min_repeat_second=3,
        setup="torch.cuda.synchronize()",
        finish="torch.cuda.synchronize()",
    )
    print(f"batch size {BS} seqlen {T} our fused batch qk impl: {t_qk * 1000: .2f} ms")
    stmt = "torch.matmul(query_state, key_state.transpose(2, 3))"
    t_qk_ref = py_benchmark(
        stmt,
        {**globals(), **locals()},
        min_repeat_second=3,
        setup="torch.cuda.synchronize()",
        finish="torch.cuda.synchronize()",
    )
    print(
        f"batch size {BS} seqlen {T} pytorch batch qk impl: {t_qk_ref * 1000: .2f} ms"
    )
    attn_weight = paddle.randn(
        (BS, nh, query_len, T), device="cuda", dtype=paddle.float16
    )
    code, scale, mn = triton_quantize_and_pack_along_last_dim(
        val_state, group_size, bits
    )
    stmt = "triton_bmm_fA_qB_outer(group_size, attn_weight, code, scale, mn, bits)"
    t_av = py_benchmark(
        stmt,
        {**globals(), **locals()},
        min_repeat_second=3,
        setup="torch.cuda.synchronize()",
        finish="torch.cuda.synchronize()",
    )
    print(f"batch size {BS} seqlen {T} our fused batch av impl: {t_av * 1000: .2f} ms")
    stmt = "torch.matmul(attn_weight, val_state)"
    t_av_ref = py_benchmark(
        stmt,
        {**globals(), **locals()},
        min_repeat_second=3,
        setup="torch.cuda.synchronize()",
        finish="torch.cuda.synchronize()",
    )
    print(
        f"batch size {BS} seqlen {T} pytorch batch av impl: {t_av_ref * 1000: .2f} ms"
    )


def test_streaming_kvcache():
    BS, nh, T, D = 1, 32, 340, 128
    our_attn_output = None
    group_size = 64
    query_len = 1
    bits = 2
    key_states = paddle.randn((BS, nh, T, D), device="cuda", dtype=paddle.float16)
    value_states = paddle.randn((BS, nh, T, D), device="cuda", dtype=paddle.float16)
    key_states_quant = key_states[
        :, :, : -(key_states.shape[-2] % group_size), :
    ].contiguous()
    key_states_full = key_states[
        :, :, -(key_states.shape[-2] % group_size) :, :
    ].contiguous()
    value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(
        value_states, group_size, bits
    )
    (
        key_states_quant_trans,
        key_scale_trans,
        key_mn_trans,
    ) = triton_quantize_and_pack_along_last_dim(
        key_states_quant.transpose(2, 3).contiguous(), group_size, bits
    )
    for i in range(16):
        if our_attn_output is None:
            query_states = paddle.randn(
                (BS, nh, query_len, D), device="cuda", dtype=paddle.float16
            )
        else:
            query_states = our_attn_output
        key_states_new = paddle.randn(
            (BS, nh, query_len, D), device="cuda", dtype=paddle.float16
        )
        value_states_new = paddle.randn(
            (BS, nh, query_len, D), device="cuda", dtype=paddle.float16
        )
        att_qkquant = triton_bmm_fA_qB_outer(
            group_size,
            query_states,
            key_states_quant_trans,
            key_scale_trans,
            key_mn_trans,
            bits,
        )
        key_states_full = paddle.cat([key_states_full, key_states_new], dim=2)
        att_qkfull = paddle.matmul(query_states, key_states_full.transpose(2, 3))
        our_att_weights = paddle.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(D)
        our_att_weights = paddle.softmax(our_att_weights, dim=-1)
        value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(
            value_states_new, group_size, bits
        )
        value_states_quant = paddle.cat(
            [value_states_quant, value_states_quant_new], dim=2
        )
        value_scale = paddle.cat([value_scale, scale], dim=2)
        value_mn = paddle.cat([value_mn, mn], dim=2)
        our_attn_output = triton_bmm_fA_qB_outer(
            group_size, our_att_weights, value_states_quant, value_scale, value_mn, bits
        )
        key_states = paddle.cat([key_states, key_states_new], dim=2)
        value_states = paddle.cat([value_states, value_states_new], dim=2)
        ref_att_weights = paddle.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(D)
        ref_att_weights = paddle.softmax(ref_att_weights, dim=-1)
        ref_att_out = paddle.matmul(ref_att_weights, value_states)
        att_weight_gap = (ref_att_weights - our_att_weights) / ref_att_weights
        print(
            f"i {i} bit {bits}, mean att weight rel arr: {paddle.mean(paddle.abs(x=att_weight_gap))}"
        )
        att_out_gap = (ref_att_out - our_attn_output) / ref_att_out
        print(
            f"i {i} bit {bits}, mean att out rel arr: {paddle.mean(paddle.abs(x=att_out_gap))}"
        )


def test_4d_qmatmul():
    paddle.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    query_len = 1
    BS, nh, T, D = 16, 32, 1024, 128
    group_size = 64
    k = paddle.randint(low=0, high=10, shape=(BS, nh, T, D), dtype=paddle.int32).to(paddle.float16)
    query_state = paddle.randint(low=0, high=5, shape=(BS, nh, query_len, D), dtype=paddle.int32).to(
        paddle.float16
    )
    for bits in [8, 4, 2]:
        code, scale, mn = quant_and_pack_kcache(k, group_size, bits)
        dequant_k = unpack_and_dequant_kcache(code, scale, mn, group_size, bits)
        code = code.transpose(2, 3)
        scale = scale.view(BS, nh, -1, D).transpose(2, 3)
        mn = mn.view(BS, nh, -1, D).transpose(2, 3)
        our_out = triton_bmm_fA_qB_outer(group_size, query_state, code, scale, mn, bits)
        ref_out = paddle.matmul(query_state, k.transpose(2, 3))
        assert not our_out.isnan().any()
        assert not ref_out.isnan().any()
        gap = (our_out - ref_out) / ref_out
        gap = paddle.nan_to_num(x=gap)
        err = paddle.mean(paddle.abs(x=gap)).item()
        print(f"bits {bits}, err: {err}")


if __name__ == "__main__":
    set_seed(114514)
    test_bmm_speed()
