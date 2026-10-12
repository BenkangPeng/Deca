"""
RMSNorm over the last dimension of an [M, N] tensor with a weight (forward):
y = x * rsqrt(mean(x^2) + eps) * w
"""

from operators.Rowwise import RowwiseOperator


class RMSNorm(RowwiseOperator):
    op_kind = "rmsnorm"

    def __init__(
        self, M: int, N: int, dtype: str = "float16", eps: float = 1e-6, name: str = "RMSNorm"
    ):
        """
        Args:
            M: number of rows (tokens)
            N: hidden size
            dtype: "float16" or "bfloat16" (fp32 math)
            eps: added to the mean square
        """
        super().__init__(M, N, dtype, eps, name=name)
