import numpy as np
import mpmath

def are_equal(U1, U2):
    near_identity = U1.conj().T @ U2
    
    # just check that the trace is 2
    trace = np.trace(near_identity)
    return np.isclose(2, np.abs(trace))

def mpmath_allclose(M1, M2, rel_eps=1e-6):
    diff = M1 - M2
    maxm1 = max(M1)
    maxm2 = max(M2)
    return mpmath.almosteq(mpmath.norm(diff), 0, abs_eps=max(maxm1, maxm2) * rel_eps)
