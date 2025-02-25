import numpy as np
import mpmath
from mpmath import mpf, matrix

# import ipe_ellipse as E
# from ipe_ellipse import Ellipse, GridOperator

from . import ipe_ellipse_mpmath as E
from .ipe_ellipse_mpmath import Ellipse, GridOperator

from .int_field import ZSqrt2Int, ZetaInt
from . import ipe_general

from . import factorization as F
# import exact_synthesis as ES
from . import ma_synthesis as MA
from .intervals import Interval


def create_upright_intervals(A_ellipse: Ellipse, B_ellipse: Ellipse):
    # Given two original ellipses A, B
    # produce two new ellipses A', B' and the operator G to interchange between the two
    
    # we need det(M) ~ 1
    # achieve this by adjusting f
    A_norm_ellipse = A_ellipse.make_normal_ellipse()
    B_norm_ellipse = B_ellipse.make_normal_ellipse()
    
    # note that x M x \leq f is the same as x M/c x \leq f/c
    # if f/c = 1, this approach is a regular ellipse and SR applies immediately
    # if not, that's still okay because f/c is our renormalization factor which we can keep as such
    
    state = E.State(A_norm_ellipse.M, B_norm_ellipse.M)
    
    # now compute the grid operator
    G, reshaped_state = E.find_grid_operator(state)
    
    # we can now produce the modified ellipses using these new matrices and f values
    # we need to move the centers:
    a_x0, a_y0 = A_ellipse.center
    a_x0, a_y0 = (G.inverse().eval() / mpmath.sqrt(2)) * matrix([a_x0, a_y0])
    
    A_search = E.Ellipse(reshaped_state.d, (a_x0, a_y0), A_norm_ellipse.f)
    B_search = E.Ellipse(reshaped_state.delta, (0, 0), B_norm_ellipse.f)
    
    return G, A_search, B_search

def sample_from_interval(A_ellipse, B_ellipse, G=GridOperator().G, rescale_factor=1):
    ax, ay = A_ellipse.bounding_box()
    bx, by = B_ellipse.bounding_box()
    
    transformed_candidates = ipe_general.enumerate_2d(
        ax * rescale_factor, 
        ay * rescale_factor, 
        bx * rescale_factor, 
        by * rescale_factor
    )
    
    transformed_candidates = [G @ np.array((ZetaInt((0, 1, 0, -1)) * cand).to_zsq()) for cand in transformed_candidates]
    
    # we have computed G p * sqrt(2)
    # and we need (G / sqrt(2)) p
    
    return transformed_candidates


def produce_candidate(theta, q, eps, k):
    """Produce a series of candidates for the upper left spot such that the angle meets desired criteria. Does not imply exact synthesizability.

    Args:
        theta (float): Target angle
        q (float): Minimum acceptance probability, q <= 1
        eps (float): Angle error
        k_max (int): Maximum number of iterations to search

    Returns:
        List: List of potential candidates, potentiall offset by some factor
    """
    # first, generate the ellipses to use and offset
    
    A_ellipse = E.Ellipse.make_ellipse_rus(theta, q, eps)
    B_ellipse = E.Ellipse(mpmath.eye(2), (0, 0), 1)
    
    G, A_search, B_search = create_upright_intervals(A_ellipse, B_ellipse)
    
    # now, we enumerate the final region to search over
    leader = mpmath.sqrt(2)**k
    sampled_pts = sample_from_interval(A_search, B_search, G.G, leader)
    
    # filter the candidates
    # TODO: make it from lowest k to k_max so you can choose the cheapest implementation
    out = []
    hitrate = 0
    for candidate in sampled_pts:
        # want to first check if this candidate is in the original region
        
        cand_val = candidate[0].eval() + 1j * candidate[1].eval()
        cand_val /= 2 * leader
        
        print(f"Checking candidate: {cand_val}")
        
        candzeta = candidate[0].to_zeta() + candidate[1].to_zeta() * ZetaInt((0, 0, 1, 0)) # 1j
        # and again divide
        # candzeta /= 2
        # if (k % 2) == 1:
        #     candzeta *= ZetaInt((0, 1, 0, -1)) # sqrt(2)
        # candzeta /= 2**((k + 1) // 2)
        
        if E.check_in_region(cand_val, theta, q, eps):
            # check factorization

            xi_val = ZSqrt2Int((2**k * 4, 0)) - (candzeta * ~candzeta).to_zsq()[0]
            
            test_factorization = F.factorize_xi_into_two(xi_val)
            
            if test_factorization:
                return candzeta, test_factorization
            
            hitrate += 1
            print("Found a candidate!")
            print(cand_val)
            print(candzeta.eval())
            # out.append(candzeta)
            # return [candzeta]
    
    print(f"Hitrate: {hitrate} out of {len(sampled_pts)} or {hitrate / len(sampled_pts) * 100}%")
    
    return out


def iterate_from_interval_old(A_ellipse, B_ellipse, G=GridOperator().G, rescale_factor=1):
    ax, ay = A_ellipse.bounding_box()
    bx, by = B_ellipse.bounding_box()
    
    transformed_candidates = ipe_general.iterate_2d(
        ax * rescale_factor, 
        ay * rescale_factor, 
        bx * rescale_factor, 
        by * rescale_factor
    )

    # we have computed G p * sqrt(2)
    # and we need (G / sqrt(2)) p
    for cand in transformed_candidates:
        yield G @ np.array((ZetaInt((0, 1, 0, -1)) * cand).to_zsq())
    
def iterate_from_interval(ax: Interval, ay: Interval, bx: Interval, by: Interval, 
                          G=GridOperator().G, rescale_factor=1):
    transformed_candidates = ipe_general.iterate_2d(
        ax * rescale_factor, 
        ay * rescale_factor, 
        bx * rescale_factor, 
        by * rescale_factor
    )

    # we have computed G p * sqrt(2)
    # and we need (G / sqrt(2)) p
    for cand in transformed_candidates:
        yield G @ np.array((ZetaInt((0, 1, 0, -1)) * cand).to_zsq())


def search_for_candidate(theta, q, eps, verbose=False):
    # iteratively increase the k value until a candidate has been found
    
    # first, generate the ellipses to use and offset
    
    A_ellipse = E.Ellipse.make_ellipse_rus(theta, q, eps)
    B_ellipse = E.Ellipse(mpmath.eye(2), (0, 0), 1)
    
    if verbose:
        print("Finding upright intervals")
    G, A_search, B_search = create_upright_intervals(A_ellipse, B_ellipse)
    
    as_x, as_y = A_search.bounding_box()
    bs_x, bs_y = B_search.bounding_box()
    
    if verbose:
        print("Starting search for candidates")
    k = 0
    targets = 0
    hits = 0
    while True:
        if verbose:
            print(f"Checking k={k}")
        
        # now, we enumerate the final region to search over
        leader = mpmath.sqrt(2)**k
        sampled_pts_iterator = iterate_from_interval(as_x, as_y, bs_x, bs_y, G.G, leader)
        
        # filter the candidates
        for candidate in sampled_pts_iterator:
            targets += 1
            # Prop 5.22 Reminds us that we can filter out candidates without a-c or b-d odd
            
            candzeta = candidate[0].to_zeta() + candidate[1].to_zeta() * ZetaInt((0, 0, 1, 0)) # 1j
            d, c, b, a = candzeta / 2
            if (a - c) % 2 == 0 and (b - d) % 2 == 0:
                continue
            
            cand_val = candidate[0].eval_mpmath() + 1j * candidate[1].eval_mpmath()
            cand_val /= 2 * leader
            
            # want to first check if this candidate is in the original region
            if E.check_in_region(cand_val, theta, q, eps):
                hits += 1
                # check factorization

                xi_val = ZSqrt2Int((2**k * 4, 0)) - (candzeta * ~candzeta).to_zsq()[0]
                
                if xi_val == ZSqrt2Int((0, 0)):
                    return candzeta, ZetaInt((0, 0, 0, 0))
                
                test_factorization = F.factorize_xi_into_two(xi_val)
                
                if test_factorization:
                    return candzeta, test_factorization
        
        # one-line division that's safe by 0
        if verbose:
            if targets == 0:
                print("No valid points found; increasing k")
            else:
                print(f"Hitrate: {hits} out of {targets} or {hits / targets * 100}%")
        
        # no valid points; increase k
        k += 1


def synthesize(theta, d_prec, failure, verbose=False):
    return find_rus(theta, mpf(1 - failure), mpmath.power(10, - d_prec), verbose=verbose)

def find_rus(theta, q, eps, verbose=False):
    if type(theta) is str:
        # may be in symbolic form; convert to float
        # first look for "pi"
        
        theta_str = theta.replace("pi", "mpmath.pi")
        assert len(theta_str) < 40, "Symbolic expression too long"
        theta = eval(theta_str)
    
    
    m1, m2 = search_for_candidate(theta, q, eps, verbose=verbose)
    if verbose:
        print(f"Found a candidate: {m1}, {m2}")
    seq = MA.synthesize(np.array([
        [m1, ~m2 * -1],
        [m2, ~m1]
    ]))
    return "".join(seq)

def find_rus_old(theta, q, eps, k):
    m1, m2 = produce_candidate(theta, q, eps, k)
    seq = MA.synthesize(np.array([
        [m1, ~m2 * -1],
        [m2, ~m1]
    ]))
    return "".join(seq)