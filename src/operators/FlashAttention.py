"""
FlashAttention (forward) operator.

Unlike GEMM, the kernel is not derived from a naive compute through schedule
primitives: a tile configuration (`FAConfig`) fully determines a low-level
TIR kernel produced by `kernels.flash_attn_tir.make_flash_attn_fwd`.
"""

import math

import tvm

from operators.Op import Operator
from kernels.flash_attn_tir import FAConfig, make_flash_attn_fwd, check_config


class FlashAttention(Operator):
    def __init__(
        self,
        B: int,
        H: int,
        S_q: int,
        S_kv: int,
        D: int,
        dtype: str = "float16",
        softmax_scale: float | None = None,
        causal: bool = False,
        name: str = "FlashAttention",
    ):
        """
        Args:
            B, H: batch size and number of heads
            S_q, S_kv: query / key-value sequence lengths
            D: head dimension
            dtype: only "float16" is supported (fp32 accumulation)
            softmax_scale: defaults to 1 / sqrt(D)
            causal: top-left aligned causal mask (requires S_q == S_kv)
        """
        assert dtype == "float16", "FlashAttention only supports float16 inputs"
        self.B, self.H, self.S_q, self.S_kv, self.D = B, H, S_q, S_kv, D
        self.dtype = dtype
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
        assert not causal or S_q == S_kv, "causal attention requires S_q == S_kv"
        self.causal = causal
        self.num_flops = 4 * B * H * S_q * S_kv * D // (2 if causal else 1)
        super().__init__(name=name)

    @property
    def shape(self):
        return (self.B, self.H, self.S_q, self.S_kv, self.D)

    def is_valid(self, cfg: FAConfig) -> bool:
        try:
            check_config(cfg, self.S_q, self.S_kv, self.D)
            return True
        except AssertionError:
            return False

    def build_mod(self, cfg: FAConfig) -> tvm.IRModule:
        return make_flash_attn_fwd(
            self.B, self.H, self.S_q, self.S_kv, self.D, cfg, self.softmax_scale, self.causal
        )
