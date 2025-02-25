import sys

sys.path.append('..')

import numpy as np
import parameters as P

# import ipe_ellipse as ipe_ellipse
import ipe_ellipse_mpmath as E
import mpmath
from mpmath import mpf

import search as S
from int_field import ZetaInt, ZSqrt2Int

import utils

def _check_equivalent_intervals(G: E.GridOperator, orig_ellipse: E.Ellipse, new_ellipse: E.Ellipse):
    xint, yint = orig_ellipse.bounding_box()
    xint_prime, yint_prime = new_ellipse.bounding_box()
    
    corner_pts = np.array([
        [xint.imin, yint.imin],
        [xint.imax, yint.imin],
        [xint.imin, yint.imax],
        [xint.imax, yint.imax]
    ])
    
    corner_pts_prime = np.array([
        [xint_prime.imin, yint_prime.imin],
        [xint_prime.imax, yint_prime.imin],
        [xint_prime.imin, yint_prime.imax],
        [xint_prime.imax, yint_prime.imax]
    ])
    
    # now, ensure the transformation actually matches!
    gval = G.eval()
    
    transformed_corners = gval @ corner_pts.T / np.sqrt(2)
    diff = transformed_corners.T - corner_pts_prime
    assert np.all(np.isclose(transformed_corners.T, corner_pts_prime)), (transformed_corners.T, corner_pts_prime)



def test_upright_interval_manipulation():
    # perform the interval deformation technique
    # and see how well this deformed interval 
    # samples from our original space
    
    mpmath.mp.dps = 300
    
    for _ in range(P.MANY_TRIALS):
        z, e = np.random.randint(-20, 20), np.random.randint(1, 20)
        e = 2 if e == 0 else e
        z = mpf(z)
        e = mpf(e)
        b = mpmath.sqrt(e**2 - 1) # e**2 - b**2 = 1 so b = sqrt(e**2 - 1)
        
        zeta = mpf(0)
        eta = mpf(1)
        beta = mpf(0)
        
        A_ellipse = E.Ellipse.make_ellipse_from_parameters(z, e, b)
        B_ellipse = E.Ellipse.make_ellipse_from_parameters(zeta, eta, beta)
        
        G, A_prime, B_prime = S.create_upright_intervals(A_ellipse, B_ellipse)
        
        # validate that our G operator actually translates between the two produced matrices
        
        s_orig = E.State(A_ellipse.M, B_ellipse.M)
        s_orig = s_orig.apply(G, constant=mpmath.mpf(1 / 2))
        
        A_translated = s_orig.d
        B_translated = s_orig.delta
        
        if not utils.mpmath_allclose(B_translated, B_prime.M, rel_eps=1e-5):
            print(B_translated - B_prime.M, A_translated, A_prime.M)
        
        assert utils.mpmath_allclose(A_translated, A_prime.M, rel_eps=1e-5), (A_translated - A_prime.M, A_translated, A_prime.M)
        assert utils.mpmath_allclose(B_translated, B_prime.M, rel_eps=1e-5), (B_translated - B_prime.M, B_translated, B_prime.M)
        # assert np.allclose(A_translated, A_prime.M, atol=1e-5), (A_translated, A_prime.M)
        # assert np.allclose(B_translated, B_prime.M), (B_translated, B_prime.M)
        # this means that:
        # A' = (G.T / sqrt(2)) A (G / sqrt(2))
        # so: any point p which works for A' also has G p / sqrt(2) work for A 
        # note that G has an additional sqrt(2) factor that needs to be removed; everything else is fine, actually

def test_upright_interval_sampling():
    # we know from prev test that our intervals created are good
    # so now check sampling
    for _ in range(P.MANY_TRIALS):
        z, e = np.random.randint(-10, 10), np.random.randint(1, 10)
        e = 2 if e == 0 else e
        b = np.sqrt(e**2 - 1) # e**2 - b**2 = 1 so b = sqrt(e**2 - 1)
        
        # z = 0
        # e = 8
        # b = np.sqrt(e**2 - 1)
        
        zeta = 0
        eta = 1
        beta = 0
        
        A_ellipse = E.Ellipse.make_ellipse_from_parameters(z, e, b)
        B_ellipse = E.Ellipse.make_ellipse_from_parameters(zeta, eta, beta)
        
        G, A_prime, B_prime = S.create_upright_intervals(A_ellipse, B_ellipse)
    
        # sample from new intervals, translate, and ensure they are relatively good
        # use sample_from_interval to accomplish this 
        leader = np.sqrt(2)**5
        sampled_points = S.sample_from_interval(A_prime, B_prime, G.G, rescale_factor=leader)
        
        # check how many of the points in sampled_points end up in the original region
        hitrate = 0
        for point in sampled_points:
            val = point[0].eval() + 1j * point[1].eval()
            val /= 2 * leader
            
            if A_ellipse.in_ellipse((val.real, val.imag)):
                hitrate += 1
        
        assert hitrate / len(sampled_points) > 0.2, f"{G}\n{A_ellipse}\n{A_prime}\n Hitrate: {hitrate} out of {len(sampled_points)} or {hitrate / len(sampled_points) * 100}%"


def test_search_easy():
    K = 15
    
    leader = np.sqrt(2)**K
    for _ in range(P.FEW_TRIALS):
        # randomly generate an angle with an easy precision
        
        theta = (np.random.rand() * 2 - 1) * np.pi
        q = 0.9
        eps = 1e-5
        
        cand, comp = S.produce_candidate(theta, q, eps, k=K)
        
        assert np.isclose(np.abs(cand.eval())**2 + np.abs(comp.eval())**2, 2**(K + 2)), "Not a valid factorization"
        
        cand = cand.eval() / (2 * leader)
        actual_angle = np.angle(cand)
        actual_prob = np.abs(cand)
        
        assert np.abs(actual_angle - theta) < eps, f"Actual angle: {actual_angle}, theta: {theta}"
        assert actual_prob >= q, f"Actual prob: {actual_prob}, q: {q}"
        
