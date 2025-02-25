import numpy as np
import parameters as P

# import ipe_ellipse as E
import ipe_ellipse_mpmath as E
from int_field import ZetaInt, ZSqrt2Int

import mpmath

def test_make_ellipse():
    raise NotImplementedError
    pass

def test_ellipse_circle():
    unit_circle = E.Ellipse(np.eye(2), (0, 0), 1)
    # bounding box
    
    bb_w, bb_h = unit_circle.bounding_box()
    assert np.isclose(float(bb_w.width()), 2) 
    assert np.isclose(float(bb_h.width()), 2)
    
    # in_ellipse
    # make a circle and see that it works lol
    for _ in range(P.MANY_TRIALS):
        x = np.random.rand() * 2 - 1
        y = np.random.rand() * 2 - 1
        
        truth_in_circle = x**2 + y**2 <= 1
        assert truth_in_circle == unit_circle.in_ellipse((x, y))

def test_ellipse_bb_general():
    for _ in range(10):
        # set a random a, b
        a = np.random.rand()
        b = np.random.rand()
        theta = (np.random.rand() * 2 - 1) * np.pi
        
        ellipse = E.Ellipse.make_ellipse_from_axis(theta, a, b)
        
        bb_w, bb_h = ellipse.bounding_box()
        
        # we'd like to now check whether the approximated area is correct
        num_pts = 200000
        x_rand = np.random.random(num_pts) * (bb_w.width()) + bb_w.imin
        x_rand *= 1.5
        y_rand = np.random.random(num_pts) * (bb_h.width()) + bb_h.imin
        y_rand *= 1.5
        
        hitrate = 0
        num_box_pts = 0
        for idx in range(num_pts):
            coord = (x_rand[idx], y_rand[idx])
            
            if bb_w.contains(x_rand[idx]) and bb_h.contains(y_rand[idx]):
                # this point is in the bounding box and should contribute to our area estimation
                num_box_pts += 1
                if ellipse.in_ellipse(coord):
                    hitrate += 1
            else:
                # this point is outside the bounding box and should NEVER be in the ellipse
                assert not ellipse.in_ellipse(coord)
        ratio = hitrate / num_box_pts
        area_diff = ratio * bb_w.width() * bb_h.width() - np.pi * a * b
        assert np.abs(area_diff) < 0.03, area_diff
        
        # also check that things outside the bounding box don't work
        
        

def test_make_ellipse_enclosed():
    # build an ellipse for RUS and check that it totally encompasses the region
    
    for _ in range(P.MANY_TRIALS):
        # build an ellipse
        theta = (mpmath.rand() * 2 - 1) * np.pi
        q = 0.9
        eps = 1e-5
        
        _test_parametrized_ellipse(theta, q, eps)
        

def _test_parametrized_ellipse(theta, q, eps, missed_tol=0):
    ellipse = E.Ellipse.make_ellipse_rus(theta, q, eps)
    num_missed = 0
    
    # create a gridmesh over angle and r and then evaluate
    # note that rounding errors are problematic, so slightly constrain the angle range and r
    angle_range = mpmath.linspace(theta - eps * 0.99, theta + eps * 0.99, 10)
    r_range = mpmath.linspace(q + (1 - q) * 0.01, 1 - (1 - q) * 0.01, 10)
    
    for angle in angle_range:
        for r in r_range:
            x = r * mpmath.cos(angle)
            y = r * mpmath.sin(angle)
            
            assert E.check_in_region(x + 1j * y, theta, q, eps), f"angle: {mpmath.arg(x + 1j * y)}, targ: {theta}, x: {x}, y: {y}, abs: {mpmath.fabs(x + 1j * y)}, angle-targ: {mpmath.arg(x + 1j * y) - theta}, eps: {eps}, q: {q}"
            
            if not ellipse.in_ellipse((x, y)):
                num_missed += 1
                # assert False, f"Failed for x={x}, y={y}, angle={angle}, r={r}, theta={theta}, q={q}, eps={eps}, Ellipse_val: {ellipse._point_value((x, y))}, ellipse: {ellipse.M}, f: {ellipse.f}"
                assert ellipse._point_value((x, y)) / ellipse.f < 10, f"Failed for x={x}, y={y}, angle={angle}, r={r}, theta={theta}, q={q}, eps={eps}, Ellipse_val: {ellipse._point_value((x, y))}, ellipse: {ellipse.M}, f: {ellipse.f} || {ellipse._point_value((x, y)) / ellipse.f} "
    assert num_missed / P.MANY_TRIALS <= missed_tol, num_missed / P.MANY_TRIALS


def test_make_ellipse_stress():
    # want to stress test making an ellipse
    for _ in range(P.FEW_TRIALS):
        theta = (mpmath.rand() * 2 - 1) * mpmath.pi
        q = mpmath.mpf(0.9 + np.random.rand() * (1 - 0.9))
        
        for eps_power in mpmath.linspace(-12, -3, 10):
            eps = mpmath.power(10, eps_power)
            _test_parametrized_ellipse(theta, q, eps, missed_tol=0.05)
    
    ## we are probably losing precision implicitly by doing (x, y) (center points) + the offset within the region
    # but it's not clear what to do because searching explicitly within the offset will 
    ## may want to just do this in c++ with higher values
    # for now, let this fail




def test_GridOperator():
    test_val = np.array([
        [ZSqrt2Int((1, 0)), ZSqrt2Int((0, 1))], [ZSqrt2Int((0, 1)), ZSqrt2Int((1, 0))]
    ])
    
    test_G = E.GridOperator(test_val)
    
    test_G_bul = +test_G
    
    true_G_bul = np.array([
        [ZSqrt2Int((1, 0)), ZSqrt2Int((0, -1))], [ZSqrt2Int((0, -1)), ZSqrt2Int((1, 0))]
    ])
    
    for i in range(2):
        for j in range(2):
            assert test_G_bul.G[i, j] == true_G_bul[i, j]

def test_make_inverse_grid_operator():
    test_G = E.GridOperator()
    
    # TODO: this needs to be mpmath
    matrix_diff = mpmath.norm(test_G.eval() * test_G.inverse().eval() - 2 * mpmath.eye(2))
    assert mpmath.almosteq(matrix_diff, 0), (test_G, test_G.inverse())

def test_find_grid_operators():
    for _ in range(P.MANY_TRIALS):
        # randomly generate a state
        
        z, e = np.random.randint(-10, 10), np.random.randint(1, 10)
        e = 2 if e == 0 else e
        b = np.sqrt(e**2 - 1) # e**2 - b**2 = 1 so b = sqrt(e**2 - 1)
        
        zeta = 0
        eta = 1
        beta = 0
        
        s = E.State.from_parameters(z, e, b, zeta, eta, beta)
        
        G, s = E.find_grid_operator(s)
        
        assert s.skew() <= 5