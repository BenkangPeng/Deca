from arch.Arch import Arch
from arch.RTX4090 import RTX4090
class Tuner:
    def __init__(self, operator: str = "GEMM"):
        assert operator in ["GEMM", "GEMV"]
        
        self.operator = operator
        self.arch = RTX4090()

    def tune_cpt_0(self):
        """
        Tune the compute level 0(register level) of the operator.
        """
        pass

    def tune_cpt_1(self):
        """
        Tune the compute level 1(shared memory level) of the operator.
        """
        pass

    def tune_cpt_2(self):
        """
        Tune the compute level 2(global memory level) of the operator.
        """
        pass