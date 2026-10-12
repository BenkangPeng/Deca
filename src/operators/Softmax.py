"""
Softmax over the last dimension of an [M, N] tensor (forward).
"""

from operators.Rowwise import RowwiseOperator


class Softmax(RowwiseOperator):
    op_kind = "softmax"

    def __init__(self, M: int, N: int, dtype: str = "float16", name: str = "Softmax"):
        """
        Args:
            M: number of rows (e.g. tokens)
            N: reduced dimension (hidden / vocab size)
            dtype: "float16" or "bfloat16" (fp32 math)
        """
        super().__init__(M, N, dtype, name=name)
