from arch.RTX4090 import RTX4090
from arch.Arch import Arch
from Solver import solve_reg, solve_shared


class CLayer:
    def __init__(
        self,
        level: int,
        arch: Arch = RTX4090(),
        type_size: int = 4,
        rtiles0: list[tuple[int, int, int]] = None,
        use_tc: bool = False,
        shape=None,
    ):
        """
        Args:
            level: int, the level of the compute layer
            arch: Arch, the architecture of the GPU
            type_size: int, the size of the data type in bytes, 4 bytes for fp32, 2 bytes for fp16
        """
        self.level = (
            level  # 0: register level, 1: shared memory level, 2: global memory level
        )
        self.arch = arch
        self.type_size = type_size
        self.rtiles0 = rtiles0
        self.use_tc = use_tc
        assert len(shape) == 3, "The shape should be a 3D array"
        self.M, self.N, self.K = shape

    def solve(self):
        if self.level == 0:  # register level
            rtiles0 = solve_reg(
                self.arch.reg_cap[1],
                self.type_size,
                self.use_tc,
                self.M,
                self.N,
                self.K,
                count=30,
            )
            return rtiles0
        elif self.level == 1:  # shared memory level
            compute_pattern = []
            for rtile0 in self.rtiles0:
                pattern = [rtile0]
                rtiles1 = solve_shared(
                    self.arch.smem_cap[0],
                    self.arch.reg_cap[0],
                    rtile0,
                    self.type_size,
                    self.use_tc,
                    self.M,
                    self.N,
                    self.K,
                    count=80,
                )
                if rtiles1 is not None:
                    for rtile1 in rtiles1:
                        compute_pattern.append(pattern + [rtile1])
            return compute_pattern
        elif self.level == 2:  # global memory level
            mem_usage = 1
            for size in self.rtiles0:
                mem_usage *= size
            if mem_usage > self.arch.smem_cap[0]:
                raise ValueError(
                    f"The size of matrix is too large: {mem_usage} > {self.arch.smem_cap[0]}"
                )
            else:
                return [self.rtiles0]
        else:
            raise ValueError(f"Invalid level: {self.level}")

    def setConstraints(self):
        # All the Constraints are defined in the Solver.py now.
        # TODO: Move the constraints into this func and add more constraints here.
        pass
