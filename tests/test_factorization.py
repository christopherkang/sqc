

import numpy as np
import parameters as P

import factorization as f
from int_field import ZetaInt, ZSqrt2Int


def test_factorize_zsq2():
    f.factorize_unit(ZSqrt2Int((-7, 5)))
    
    for _ in range(P.MANY_TRIALS):
        zsq2 = ZSqrt2Int((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)))
        print(f"Factoring {zsq2}")
        factorization = f.factorize_zsqrt2(zsq2)
            
        reconstructed = f.evaluate_factorization(factorization)
        
        q, r = zsq2 / reconstructed
        assert r == ZSqrt2Int((0, 0)), f"zsq2={zsq2}, reconstructed={reconstructed}, r={r}"
        assert np.abs((~q * q).coeffs[0]) == 1, f"q={q}, q^* q={(~q * q).coeffs[0]}, {factorization}"

def test_factorize_xi():
    f.factorize_xi(ZSqrt2Int((3175, 228)))

def test_stress_xi_factorization():
    # test factorization with larger and larger numbers
    # in particular, this is made to stress the quadratic residue solver
    
    for _ in range(P.MANY_TRIALS):
        t_init = ZetaInt((np.random.randint(100000, 300000), np.random.randint(100000, 300000), np.random.randint(100000, 300000), np.random.randint(100000, 300000)))
        
        t_t_dag = t_init * ~t_init
        xi, _ = t_t_dag.to_zsq()
        
        omega_fact = f.factorize_xi(xi)
        t_val = f.evaluate_factorization(omega_fact, base=ZetaInt((1, 0, 0, 0)))
        
        assert t_val * ~t_val == t_t_dag, f"t_val={t_val}, t_t_dag={t_t_dag}, omega_fact={omega_fact}"
        
        