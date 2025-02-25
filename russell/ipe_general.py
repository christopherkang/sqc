import numpy as np
import mpmath
from mpmath import mpf

from .int_field import ZetaInt, ZSqrt2Int, convert_to_zeta
from .intervals import Interval


def find_within_interval(x_int: Interval, y_int: Interval):
    """Lemma 17 from Selinger synthesis. Finds a single point z in ZSqrt2 where z in x_int and z* in y_int. 

    Args:
        x_int (Interval): Primary interval
        y_int (Interval): Sqrt2 conjugated interval
    """
    delta = x_int.width()
    Delta = y_int.width()
    
    assert delta * Delta > (1 + mpmath.sqrt(2)) ** 2
    
    x0 = x_int.imin
    y0 = y_int.imin
    
    # find a, b
    a = int(mpmath.floor((x0 + y0 + Delta) / 2)) + 1
    b = int(mpmath.floor((x0 - y0 - Delta) / (2 * mpmath.sqrt(2)))) + 1
    
    assert a - 1 <= (x0 + y0 + Delta) / 2
    assert (b - 1) * mpmath.sqrt(2) <= (x0 - y0 - Delta) / 2
    
    # assess cases:
    if a - b * mpmath.sqrt(2) <= y0 + Delta: # Case 1
        return ZSqrt2Int((a, b))
    
    if a - b * mpmath.sqrt(2) > y0 + Delta and a + b * mpmath.sqrt(2) <= x0 + 1: # Case 2
        return ZSqrt2Int((a, b + 1))
    
    if a - b * mpmath.sqrt(2) > y0 + Delta and a + b * mpmath.sqrt(2) > x0 + 1: # Case 3
        return ZSqrt2Int((a - 1, b))
    
    raise ValueError("Should not happen")

def iterate_1d(x_int: Interval, y_int: Interval):
    """Finds all ZSqrt2Int points z where z in x_int and z* in y_int.

    Args:
        x_int (Interval): Primary interval
        y_int (Interval): Sqrt2 conjugated interval
    """
    
    lambda_con = 1 + mpmath.sqrt(2)
    lambda_inv = -1 + mpmath.sqrt(2)
    
    mult_offset = int(mpmath.floor(mpmath.log(x_int.width()) / mpmath.log(lambda_con))) + 1
    
    x_int_prime = x_int * lambda_inv**mult_offset
    y_int_prime = y_int * (-1 * lambda_con)**mult_offset
    
    delta = x_int_prime.width()
    
    assert lambda_inv <= delta < 1 or mpmath.isclose(delta, lambda_inv), f"delta={delta}"
    
    # find all valid b's
    x0, x1 = x_int_prime.endpoints()
    y0, y1 = y_int_prime.endpoints()
    
    new_interval_0 = (x0 - y1) / mpmath.sqrt(8)
    new_interval_1 = (x1 - y0) / mpmath.sqrt(8)
    
    assert new_interval_0 < new_interval_1, f"new_interval_0={new_interval_0}, new_interval_1={new_interval_1}"
    
    raw_sols = []
    
    b_min = int(mpmath.ceil(new_interval_0))
    b_max = int(mpmath.floor(new_interval_1))
    
    # now return them back to our original interval
    if mult_offset < 0:
        lambda_cons = ZSqrt2Int((-1, 1))
        lambda_offset = lambda_cons**(-mult_offset)
    else:
        lambda_cons = ZSqrt2Int((1, 1))
        lambda_offset = lambda_cons**mult_offset
    
    for b_val in range(b_min, b_max + 1):
        a_min = int(mpmath.ceil(x0 - b_val * mpmath.sqrt(2)))
        a_max = int(mpmath.floor(x1 - b_val * mpmath.sqrt(2)))
        
        if a_min == a_max:
            candidate = ZSqrt2Int((a_min, b_val))
            if x0 <= candidate.eval() <= x1 and y0 <= (~candidate).eval() <= y1:
                # raw_sols.append(candidate)
                yield candidate * lambda_offset

def enumerate_1d(x_int: Interval, y_int: Interval):
    return list(iterate_1d(x_int, y_int))

def _iterate_2d_even(ax_int: Interval, ay_int: Interval, bx_int: Interval, by_int: Interval):
    alpha_cands = iterate_1d(ax_int, bx_int)
    beta_cands = iterate_1d(ay_int, by_int)
    
    for alpha in alpha_cands:
        for beta in beta_cands:
            u = convert_to_zeta(alpha, beta)
            yield u

def _iterate_2d_odd(ax_int: Interval, ay_int: Interval, bx_int: Interval, by_int: Interval):
    offset = 1/mpmath.sqrt(2)
    alpha_cands = iterate_1d(ax_int - offset, bx_int + offset)
    beta_cands = iterate_1d(ay_int - offset, by_int + offset)
    
    for alpha in alpha_cands:
        for beta in beta_cands:
            u = convert_to_zeta(alpha, beta) + ZetaInt((0, 1, 0, 0))
            yield u

def iterate_2d(ax_int: Interval, ay_int: Interval, bx_int: Interval, by_int: Interval):
    """Finds all ZetaInt points z where Re(z) in ax_int, Im(z) in ay_int, Re(z*) in bx_int, Im(z*) in by_int.

    Args:
        ax_int (Interval): Primary real interval
        ay_int (Interval): Primary imag interval
        bx_int (Interval): Sqrt2 conj real interval
        by_int (Interval): Sqrt2 conj imag interval
    """
    
    even_sols = _iterate_2d_even(ax_int, ay_int, bx_int, by_int)
    odd_sols = _iterate_2d_odd(ax_int, ay_int, bx_int, by_int)
    
    for sol in even_sols:
        yield sol
    
    for sol in odd_sols:
        yield sol

def enumerate_2d(ax_int: Interval, ay_int: Interval, bx_int: Interval, by_int: Interval):
    """Lemma 5.6 from Selinger synthesis. Finds all ZetaInt points z where Re(z) in ax_int, Im(z) in ay_int, Re(z*) in bx_int, Im(z*) in by_int.

    Args:
        ax_int (Interval): Primary real interval
        ay_int (Interval): Primary imag interval
        bx_int (Interval): Sqrt2 conj real interval
        by_int (Interval): Sqrt2 conj imag interval
    """
    return list(iterate_2d(ax_int, ay_int, bx_int, by_int))


