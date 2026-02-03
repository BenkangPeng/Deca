from z3 import *

def solve_reg(num_regs, type_size=4, count=5):
    """
    Solve the register tiling problem for the given number of registers.
    Args:
        num_regs: int, the number of registers that can be used by a single thread
        type_size: int, the size of the data type in bytes, 4 bytes for fp32, 2 bytes for fp16
        count: int, the number of solutions to find
    Returns:
        list of tuples, each tuple contains the tiling sizes for the register level
    """
    s = Solver()
    # Each thread computes a tile of size rm x rn x rk
    rm, rn, rk = BitVecs('rm rn rk', 64)

    m_bv = BitVecVal(num_regs, 64)
    # reduce dimension to 1
    s.add(rk == 1)
    for var in [rm, rn, rk]:
        s.add(UGT(var, 0))
        s.add(var & (var - 1) == 0)
        
    used_reg = (rm*rk + rn*rk + rm*rn) * type_size / 4 # Each register can store 4 bytes of data
    # used_reg > M / 2 && used_reg <= M
    s.add(ULE(used_reg, m_bv))
    s.add(UGT(used_reg, m_bv >> 1))
    s.add(Or(rm == rn, rm == rn >> 1, rn == rm >> 1))

    results = []
    for i in range(count):
        if s.check() == sat:
            m = s.model()
            res = (m[rm].as_long(), m[rn].as_long(), m[rk].as_long())
            results.append(res)
            
            # BLOCKING CLAUSE: 
            # "Find a new solution where (rm,rn,rk) is NOT the current (rm,rn,rk)"
            s.add(Or(rm != m[rm], rn != m[rn], rk != m[rk]))
        else:
            break
    return results

def solve_shared(M, count=5):
    # Use Solver instead of Optimize to find multiple points
    s = Solver()
    
    a, b, c = BitVecs('a b c', 64)
    m_bv = BitVecVal(M, 64)
    
    for var in [a, b, c]:
        s.add(UGT(var, 0))
        s.add(var & (var - 1) == 0)
        s.add(ULT(var, m_bv))
        
    total = (a*c) + (b*c) + (a*b)
    s.add(UGT(total, m_bv / 4))
    s.add(ULE(total, m_bv))

    s.add(Or(a == b, a == b >> 1, b == a >> 1))
    s.add(And(a > c, b > c))

    
    results = []
    for i in range(count):
        if s.check() == sat:
            m = s.model()
            res = (m[a].as_long(), m[b].as_long(), m[c].as_long())
            results.append(res)
            
            # BLOCKING CLAUSE: 
            # "Find a new solution where (a,b,c) is NOT the current (a,b,c)"
            s.add(Or(a != m[a], b != m[b], c != m[c]))
        else:
            break
    return results

print(solve_reg(255))
print(solve_shared(49152))
