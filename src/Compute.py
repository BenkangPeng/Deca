from arch.Arch import Arch
from Solver import solve_reg, solve_shared
class Compute:
    def __init__(self, level: int, arch: Arch):
        self.level = level
        self.arch = arch

    def solve(self):
        if self.level == 0: # register level
            rtiles = solve_reg(self.arch.reg_cap[self.level], type_size=4)
        elif self.level == 1: # shared memory level
            rtiles = solve_shared(self.arch.mem_cap[self.level])
        return rtiles
