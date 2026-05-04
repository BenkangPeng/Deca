"""
Conv2D operator implementation (TIR-based).

This file defines a naive NCHW Conv2D compute in TIR via tvm.script.
"""

from __future__ import annotations

import tvm
from tvm.script import ir as I
from tvm.script import tir as T

from operators.Op import Operator


def _pair(v: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(v, int):
        return (v, v)
    assert len(v) == 2
    return (int(v[0]), int(v[1]))


def _conv2d_nchw_out_shape(
    H: int,
    W: int,
    R: int,
    S: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    OH = (H + 2 * pad_h - dil_h * (R - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dil_w * (S - 1) - 1) // stride_w + 1
    return OH, OW


def create_conv2d_nchw_ir_module(
    N: int,
    C: int,
    H: int,
    W: int,
    K: int,
    R: int,
    S: int,
    stride: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, int] = (0, 0),
    dilation: int | tuple[int, int] = (1, 1),
    dtype: str = "float32",
):
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    OH, OW = _conv2d_nchw_out_shape(H, W, R, S, (stride_h, stride_w), (pad_h, pad_w), (dil_h, dil_w))

    @I.ir_module
    class Conv2D_NCHW_IRModule:
        @T.prim_func
        def main(a: T.handle, w: T.handle, y: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, (N, C, H, W), dtype=dtype)
            Wt = T.match_buffer(w, (K, C, R, S), dtype=dtype)
            Y = T.match_buffer(y, (N, K, OH, OW), dtype=dtype)

            for n, k, oh, ow, rc, rr, rs in T.grid(N, K, OH, OW, C, R, S):
                with T.block("Y"):
                    vn, vk, voh, vow, vrc, vrr, vrs = T.axis.remap(
                        "SSSSRRR", [n, k, oh, ow, rc, rr, rs]
                    )
                    with T.init():
                        Y[vn, vk, voh, vow] = T.cast(0.0, dtype)

                    ih = voh * stride_h + vrr * dil_h - pad_h
                    iw = vow * stride_w + vrs * dil_w - pad_w
                    if (0 <= ih) and (ih < H) and (0 <= iw) and (iw < W):
                        Y[vn, vk, voh, vow] = (
                            Y[vn, vk, voh, vow] + A[vn, vrc, ih, iw] * Wt[vk, vrc, vrr, vrs]
                        )

    return Conv2D_NCHW_IRModule


class Conv2D(Operator):
    def __init__(
        self,
        N: int,
        C: int,
        H: int,
        W: int,
        K: int,
        R: int,
        S: int,
        stride: int | tuple[int, int] = (1, 1),
        padding: int | tuple[int, int] = (0, 0),
        dilation: int | tuple[int, int] = (1, 1),
        dtype: str = "float32",
        name: str = "Conv2D",
    ):
        self.Nb = int(N)
        self.C = int(C)
        self.H = int(H)
        self.W = int(W)
        self.Ko = int(K)
        self.R = int(R)
        self.S = int(S)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.dtype = dtype

        OH, OW = _conv2d_nchw_out_shape(
            self.H, self.W, self.R, self.S, self.stride, self.padding, self.dilation
        )
        self.OH = int(OH)
        self.OW = int(OW)

        # GEMM-like metadata for reuse in existing benchmarking utilities.
        # M ~ batch*OH*OW, N ~ K, K ~ C*R*S
        self.M = self.Nb * self.OH * self.OW
        self.N = self.Ko
        self.K = self.C * self.R * self.S

        self.num_flops = self.Nb * self.Ko * self.OH * self.OW * self.C * self.R * self.S * 2

        self.ir_module = create_conv2d_nchw_ir_module(
            N=self.Nb,
            C=self.C,
            H=self.H,
            W=self.W,
            K=self.Ko,
            R=self.R,
            S=self.S,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            dtype=self.dtype,
        )
        self.sch = tvm.tir.Schedule(self.ir_module)
        super().__init__(name=name)

    def apply_without_tc(
        self, rtile0: list | tuple, rtile1: list | tuple, rtile2: list | tuple
    ) -> tvm.tir.Schedule:
        # Keep a valid schedule object; caller may further tune/transform it.
        return self.sch

    def apply_with_tc(self, compute_pattern: list[tuple[int, int, int]]):
        raise NotImplementedError("TensorCore scheduling for Conv2D is not implemented yet.")

