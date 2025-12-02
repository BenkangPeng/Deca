

class Tuner:
    def __init__(self, operator: str = "GEMM"):
        assert operator in ["GEMM", "GEMV"]
        
        self.operator = operator

    def 