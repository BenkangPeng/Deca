from z3 import *

def solve_reg(M, type_size=4, count=5):
    # type_size: 4 bytes for fp32, 2 bytes for fp16
    s = Solver()
    a, b, c = BitVecs('a b c', 64)
    m_bv = BitVecVal(M, 64)
    # reduce dimension to 1
    s.add(c == 1)
    for var in [a, b, c]:
        s.add(UGT(var, 0))
        s.add(var & (var - 1) == 0)
        
    used_reg = (a*c + b*c + a*b) * type_size / 4 # Every register can store 4 bytes of data
    # total > M / 2 && total <= M
    s.add(ULE(used_reg, m_bv))
    s.add(UGT(used_reg, M / 2))
    s.add(Or(a == b, a == b >> 1, b == a >> 1))

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
print(solve_shared(65536))
