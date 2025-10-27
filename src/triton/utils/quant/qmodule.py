import dequant_cuda
import paddle
from pack import quantize_and_pack


def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError
    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


def dequantize_weight(qweight, d_out, d_in, w_bit, scales, zeros, group_size):
    data = qweight.reshape(-1)
    N, num_features = d_out, d_in
    weight_fp = dequant_cuda.unpack_single_precision(
        data, w_bit, scales, zeros, N, num_features // group_size, group_size
    )
    return weight_fp.view(d_out, d_in)


class MatMul4Bit(paddle.autograd.Function):
    @staticmethod
    def forward(ctx, A, qweight, bias, d_out, d_in, w_bit, scales, zeros, group_size):
        weight_fp = dequantize_weight(
            qweight, d_out, d_in, w_bit, scales, zeros, group_size
        )
        output = paddle.nn.functional.linear(
            x=A, weight=weight_fp.to(A.dtype).T, bias=bias
        )
        ctx.state = d_out, d_in, w_bit, scales, zeros, group_size
        ctx.tensors = qweight
        return output

    @staticmethod
    def backward(ctx, grad_output):
        req_gradA, _, req_gradBias = ctx.needs_input_grad[:3]
        qweight = ctx.tensors
        d_out, d_in, w_bit, scales, zeros, group_size = ctx.state
        grad_A, grad_bias = None, None
        if req_gradBias:
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)
        if req_gradA:
            weight_fp = dequantize_weight(
                qweight, d_out, d_in, w_bit, scales, zeros, group_size
            )
            grad_A = paddle.matmul(grad_output, weight_fp.to(grad_output.dtype))
            if grad_A.isnan().any():
                import ipdb

                ipdb.set_trace()
        return grad_A, None, grad_bias, None, None, None, None, None, None


class WQLinearForTrain(paddle.nn.Layer):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = 32 // self.w_bit
        self.register_buffer(
            name="qweight",
            tensor=paddle.zeros(
                (out_features, in_features // pack_num), dtype=paddle.int32, device=dev
            ),
        )
        self.register_buffer(
            name="zeros",
            tensor=paddle.zeros(
                (out_features, calculate_zeros_width(in_features, self.group_size)),
                dtype=paddle.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            name="scales",
            tensor=paddle.zeros(
                (
                    out_features,
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                ),
                dtype=paddle.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                name="bias",
                tensor=paddle.zeros(out_features, dtype=paddle.float16, device=dev),
            )
        else:
            self.bias = None

    def forward(self, x):
        out = MatMul4Bit.apply(
            x,
            self.qweight,
            self.bias,
            self.out_features,
            self.in_features,
            self.w_bit,
            self.scales,
            self.zeros,
            self.group_size,
        )
        return out

    def dequantize_weight(self):
        data = self.qweight.reshape(-1)
        N, num_features = self.out_features, self.in_features
        weight_fp = dequant_cuda.unpack_single_precision(
            data,
            self.w_bit,
            self.scales,
            self.zeros,
            N,
            num_features // self.group_size,
            self.group_size,
        )
        return weight_fp.view(self.out_features, self.in_features)

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        q_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.place,
        )
        if init_only:
            return q_linear
        quantized, scales, mn = quantize_and_pack(
            linear.weight, group_size, w_bit, simulate=False
        )
        q_linear.qweight = quantized
        q_linear.scales = scales
        q_linear.zeros = mn
        return q_linear
