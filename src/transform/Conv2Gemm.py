"""
A pass to transform a convolution operator to a GEMM operator by Img2Col.
"""

from operators.Op import Operator
from operators.GEMM import GEMM

class Conv2Gemm(Pass):
    def __init__(self):
        super().__init__("Conv2Gemm")

    def run(self, operator: Operator):
        pass