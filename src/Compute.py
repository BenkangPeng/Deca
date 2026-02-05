from arch.RTX4090 import RTX4090
from arch.Arch import Arch
from Solver import solve_reg, solve_shared


class Compute:
    def __init__(
        self,
        level: int,
        arch: Arch,
        type_size: int = 4,
        rtiles0_all: list[tuple[int, int, int]] = None,
    ):
        self.level = level
        self.arch = arch
        self.type_size = type_size
        self.rtiles0_all = rtiles0_all

    def solve(self):
        if self.level == 0:  # register level
            rtiles0 = solve_reg(self.arch.reg_cap[1], self.type_size, count=5)
            return rtiles0
        elif self.level == 1:  # shared memory level
            compute_pattern = []
            for rtiles0 in self.rtiles0_all:
                pattern = [rtiles0]
                rtiles1 = solve_shared(
                    self.arch.smem_cap[0],
                    self.arch.reg_cap[0],
                    rtiles0,
                    self.type_size,
                    count=20,
                )
                if rtiles1 is not None:
                    for rtile1 in rtiles1:
                        compute_pattern.append(pattern + [rtile1])
            return compute_pattern
        else:
            raise ValueError(f"Invalid level: {self.level}")


if __name__ == "__main__":
    arch = RTX4090()
    compute1 = Compute(0, arch, type_size=4)
    rtiles0_all = compute1.solve()
    compute = Compute(1, arch, type_size=4, rtiles0_all=rtiles0_all)
    compute_patterns = compute.solve()
    print(f'Find {len(compute_patterns)} compute patterns')
    print(compute_patterns)
