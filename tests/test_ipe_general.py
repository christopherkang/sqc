import numpy as np
import parameters as P

import ipe_general as G
from intervals import Interval
from int_field import ZetaInt, ZSqrt2Int

def test_enumerate_1D():
    for _ in range(P.MANY_TRIALS):
        amin = np.random.randint(-10, 10)
        bmin = np.random.randint(-10, 10)
        
        aint = Interval(amin, amin + 3)
        bint = Interval(bmin, bmin + 3)
        
        sols = G.enumerate_1d(aint, bint)
        
        for sol in sols:
            assert aint.contains(sol.eval())
            assert bint.contains((~sol).eval())

def test_enumerate_2D():
    for _ in range(P.MANY_TRIALS):
        amin_re = np.random.randint(-10, 10)
        amin_im = np.random.randint(-10, 10)
        bmin_re = np.random.randint(-10, 10)
        bmin_im = np.random.randint(-10, 10)
        
        aint_re = Interval(amin_re, amin_re + 3)
        aint_im = Interval(amin_im, amin_im + 3)
        bint_re = Interval(bmin_re, bmin_re + 3)
        bint_im = Interval(bmin_im, bmin_im + 3)
        
        for sol in G.iterate_2d(aint_re, aint_im, bint_re, bint_im):
            assert aint_re.contains(np.real(sol.eval())), sol
            assert aint_im.contains(np.imag(sol.eval())), sol
            assert bint_re.contains(np.real((+sol).eval())), sol
            assert bint_im.contains(np.imag((+sol).eval())), sol
