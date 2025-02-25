
from .int_field import ZSqrt2Int, ZetaInt, gcd
from sympy import factorint
import numpy as np

def solve_quadratic_residue(n, p):
    # find R such that R^2 = n mod p
    # using Tonelli-Shanks
    
    # first, factor p - 1 = q * 2^s
    q = p - 1
    s = 0
    while q % 2 == 0:
        q = q // 2
        s += 1
    
    # find a non-residue
    z = 2
    while pow(z, (p-1) // 2, p) == 1:
        z += 1
    
    # initialize
    M = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)
    
    while True:
        if t == 0:
            return 0
        if t == 1:
            return r
        
        i = 1
        while pow(t, (2**i), p) != 1:
            i += 1
            if i == M:
                return None
        
        if M - i - 1 < 0:
            return None
        
        b = pow(c, 2**(M-i-1), p)
        
        M = i 
        c = pow(b, 2, p)
        t = (t * pow(b, 2, p)) % p
        r = (r * b) % p

def factorize_prime_zsqrt2(p: int):
    """Assumes p prime and positive.
    Lemma C.12 from Selinger synthesis
    produces the factorization of xi in Z[\sqrt{2}]

    Args:
        p (int): prime

    Raises:
        ValueError: Can fail if quadratic residue finding fails

    Returns:
        List: Factorization of p in Z[\sqrt{2}]
    """
    
    assert p > 1
    
    if p == 2:
        return [ZSqrt2Int((0, 1)), ZSqrt2Int((0, 1))]

    modulo = p % 8
    
    if modulo == 3 or modulo == 5:
        return [ZSqrt2Int((p, 0))]
    
    # we are in the hard case of modulo == 1, 7
    # so that p ~ xi^* xi
    
    # first, compute satisfying x where $x^2 = 2 mod p$
    x_val = solve_quadratic_residue(2, p)
    
    if not x_val:
        raise ValueError("Should not happen")
    
    # now, xi = gcd(p, x + \sqrt{2})
    # so x + \sqrt{2} = xi * y for some y
    # suppose xi = a + b\sqrt{2}
    # so (x + \sqrt{2}) * (a - b \sqrt{2}) = xi^* * xi * y = p * y' 
    x_val = ZSqrt2Int((x_val, 1))
    xi = gcd(ZSqrt2Int((p, 0)), x_val)
    
    return [xi, ~xi]

def factorize_zsqrt2(xi: ZSqrt2Int):
    # n = \xi^\bullet * xi
    n = ~xi * xi
    n = np.abs(n.coeffs[0])
    int_factorization = factorint(n)
    zsq2_factorization = []
    
    # print(f"xi={xi}, n={n}, int_factorization={int_factorization}")
    
    for prime, power in int_factorization.items():
        # print(f"prime={prime}, power={power}")
        if prime == 2:
            zsq2_factorization.append((ZSqrt2Int((0, 1)), power))
        elif prime % 8 == 3 or prime % 8 == 5:
            assert power % 2 == 0, f"prime={prime}, power={power}"
            zsq2_factorization.append((ZSqrt2Int((prime, 0)), power // 2))
        elif prime % 8 == 1 or prime % 8 == 7:
            # p = xi^* xi
            # but p^m could be some combination of xi^*, xi
            # need to manually count the number of xi, xi^* factors
            x_val = solve_quadratic_residue(2, prime)
            x_val = ZSqrt2Int((x_val, 1))
            factor_xi = gcd(ZSqrt2Int((prime, 0)), x_val)
            factor_xi_bul = ~factor_xi
            
            for xi_count in range(1, power + 1 + 1):
                _, r = xi / (factor_xi**xi_count)
                if r != ZSqrt2Int((0, 0)):
                    break
            
            xi_count -= 1
            
            zsq2_factorization.append((factor_xi, xi_count))
            zsq2_factorization.append((factor_xi_bul, power - xi_count))
        else:
            raise ValueError(f"Should not happen; {prime}, {power}")
    
    # do some light cleanup
    zsq2_factorization = [(factor, power) for factor, power in zsq2_factorization if power > 0]
    
    return zsq2_factorization


def factorize_zomega(xi: ZSqrt2Int):
    # given some prime xi, factorize it in Z[\omega]
    # Lemma C.20?
    divide_into_prime: int = None
        
    if xi.coeffs[1] == 0:
        divide_into_prime = xi.coeffs[0]
    else:
        divide_into_prime = abs((xi * ~xi).coeffs[0])
    
    assert divide_into_prime > 1
    
    if divide_into_prime == 2:
        return ZetaInt((1, 1, 0, 0)) # 1 + omega
    elif divide_into_prime % 8 in [1, 5]:
        u = solve_quadratic_residue(-1, divide_into_prime)
        t = gcd(xi.to_zeta(), ZetaInt((u, 0, 1, 0))) # gcd (xi, u + i)
        return t
    elif divide_into_prime % 8 in [3]:
        u = solve_quadratic_residue(-2, divide_into_prime)
        t = gcd(xi.to_zeta(), ZetaInt((u, 1, 0, 1))) # gcd (xi, u + i sqrt(2))
        return t
    else:
        return False

def factorize_unit(xi: ZSqrt2Int):
    """Given unit xi find its factorization in terms of Z[\sqrt{2}] units

    Args:
        xi (ZSqrt2Int): Unit
    """
    
    assert xi * ~xi == ZSqrt2Int((1, 0)) or xi * ~xi == ZSqrt2Int((-1, 0)), f"Not unit! xi={xi}"
    
    # all units in Z[\sqrt{2}] take the form (-1)^n \lambda^m for \lambda = 1+\sqrt{2} and \lambda^{-1} = -1 + \sqrt{2}
    
    # first: identify if we have \lambda or \lambda^{-1} by checking signs
    same_sgn = xi.coeffs[0] * xi.coeffs[1] > 0
    
    factor_term: ZSqrt2Int = None
    if same_sgn:
        factor_term = ZSqrt2Int((1, 1))
    else:
        factor_term = ZSqrt2Int((-1, 1))
    
    power = 0
    while xi != ZSqrt2Int((1, 0)) and xi != ZSqrt2Int((-1, 0)):
        xi = xi * ~(factor_term * -1)
        power += 1
        
        # print(f"xi={xi}")
        if power > 100:
            raise ValueError("Should not happen")
    
    neg_factor = 1 if xi == ZSqrt2Int((1, 0)) else -1
    
    return factor_term, power, neg_factor

def factorize_xi(xi: ZSqrt2Int):
    # Lemma C.12 from Selinger synthesis
    # produces the factorization of xi in Z[\sqrt{2}]
    # assume xi positive
    # the key idea is that we will first use the norm-based factorization yielding a series of primes
    # each prime is uniquely identified with a prime in Z[\sqrt{2}]
    
    # assert xi.eval() > 0, f"p={xi}"
    if xi.eval() <= 0:
        # skip but raise warning
        print(f"SKIPPED: xi={xi} is unexpectedly negative")
        return False
    
    omega_factorization = []
    
    # 1. Factor xi into primes over Z[\sqrt2]
    zsq2_factorization = factorize_zsqrt2(xi)
    
    curr_val = ZetaInt((1, 0, 0, 0))
    # 2. For each prime, obtain the factorization in Z[\omega]
    for prime, power in zsq2_factorization:
        # identify prime where xi | p
        
        if power % 2 == 0:
            omega_factorization.append((prime.to_zeta(), power // 2))
            curr_val *= prime.to_zeta()**(power // 2)
        else:
            t_factor = factorize_zomega(prime)
            
            if not t_factor:
                # We have an inert factor and the power is odd; we have failed.
                return False
            
            omega_factorization.append((t_factor, power))
            curr_val *= t_factor**power
    
    # 3. Factorize the unit leader
    # because both quantities are doubly positive, this should just be the square root of the unit
    curr_val = curr_val * ~curr_val
    difference, _ = xi.to_zeta() / curr_val
    difference = difference.to_zsq()[0]
    term, power, neg = factorize_unit(difference)
    
    to_add = term.to_zeta()
    if neg == -1:
        raise ValueError("Should not happen")
    
    if power // 2 > 0:
        omega_factorization.append((to_add, power // 2))
    
    return omega_factorization

def factorize_xi_into_two(xi: ZSqrt2Int):
    factorization = factorize_xi(xi)
    return evaluate_factorization(factorization, base=ZetaInt((1, 0, 0, 0))) if factorization else None

###################
# DEBUG METHODS ###
###################

def evaluate_factorization(arr, base=ZSqrt2Int((1, 0))):
    for factor, power in arr:
        base *= factor**power
    
    return base

