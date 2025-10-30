from .core import (
    # Legacy names (backward-compatible)
    sageattn,
    sageattn_varlen,
    sageattn_qk_int8_pv_fp16_triton,
    sageattn_qk_int8_pv_fp16_cuda,
    sageattn_qk_int8_pv_fp8_cuda,
    sageattn_qk_int4_pv_fp16_triton,
    # Preferred new names
    lowbit_fa_attn,
    lowbit_fa_varlen,
    lowbit_fa_multi_precision,
    lowbit_fa_qk_int8_pv_fp16_triton,
    lowbit_fa_qk_int8_pv_fp16_cuda,
    lowbit_fa_qk_int8_pv_fp8_cuda,
    lowbit_fa_qk_int4_pv_fp16_triton,
)
