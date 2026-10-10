from arch.Arch import Arch
from operators.Op import Operator

class CostModel:
    def __init__(self):
        pass

    def cost(self, kernel: Operator, arch: Arch, compute_pattern: list[tuple[int, int, int]]):
        pass