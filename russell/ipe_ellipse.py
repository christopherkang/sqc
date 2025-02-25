import numpy as np
from .int_field import ZSqrt2Int, ZetaInt
from .intervals import Interval
from typing import Tuple

########################
# Ellipse building     #
########################

class Ellipse:
    def __init__(self, M, center: Tuple[float, float], f: float, normalize=False):
        # want an ellipse defined by some state matrix M such that 
        # (x - center) M (x - center) <= f
        # so that M has determinant 1 (as enforced by the invariant)
        # note that M is 2x2 representing the real, complex dimensions
        # similarly, the center will be a tuple representing real, complex
        self.M = M
        self.center = np.array(center)
        self.f = f
        
        if normalize:
            self._normalize()
        
        self._invariant()
    
    def _normalize(self):
        # convert an ellipse of form x M x \leq f to 
        # one of form x M' x \leq 1
        self.M /= self.f
        self.f = 1
    
    @classmethod
    def get_rotated_ellipse_coefficients(cls, theta, a, b):
        # x0, y0 are the center of the ellipse
        # theta is the angle of rotation
        # a, b are the semi-major and semi-minor axes
        # returns the coefficients of the ellipse equation
        # a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0
        
        # first we need to rotate the ellipse
        A = a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2
        B = 2*(b**2 - a**2) * np.sin(theta) * np.cos(theta)
        C = a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2
        F = -1 * -a**2*b**2
        
        # A /= a * b
        # B /= a * b
        # F /= a * b
        
        return A, B, C, F
    
    @classmethod
    def make_ellipse_from_axis(cls, theta, a, b):
        a, b, c, d, e, f = get_ellipse_coefficients(0, 0, theta, a, b)
        
        M = np.array([
            [a, b / 2],
            [b / 2, c]
        ], dtype=np.float64)
        
        return Ellipse(M, (0, 0), -f)
    
    @classmethod
    def make_ellipse_rus(cls, theta: np.float64, q: np.float64, eps: np.float64):
        # TODO: Verify
        # in fact these semimajor and semiminor axis values are likely inappropriate
        
        # we first want to compute the length from the center to one of the extremal points
        # this will be the distance between two points:
        # ((1 + q)/2, theta) and (1, theta + eps)
        
        # a, b, c, f = Ellipse.get_rotated_ellipse_coefficients(theta, (1 - q) / 2 / np.sqrt(2), eps / np.sqrt(2))
        
        fatten_coeff = 1.0
        if eps < 1e-6:
            fatten_coeff = -5 * np.log10(eps) / 100 + 1
        
        a, b, c, d, e, f = get_ellipse_coefficients(0, 0, theta, (1 - q) / np.sqrt(2), 2 * eps / np.sqrt(2), fatten_coeff=fatten_coeff)
        
        # note that the circle's arc will be longer than the height of the rectangle
        # the arc is 2pi r * theta/2pi = r * theta
        
        center = make_ellipse_center(theta, q)
        
        M = np.array([
            [a, b / 2],
            [b / 2, c]
        ], dtype=np.float64)
        
        return Ellipse(M, center, -f)
    
    @classmethod
    def make_ellipse_from_parameters(cls, z: float, e: float, b: float):
        M = np.array([
            [e * constant_lambda**z, b],
            [b, e * constant_lambda**(-z)]
        ])
        
        f = 1
        ctr = (0, 0)
        
        return Ellipse(M, ctr, f)
    
    def _invariant(self):
        # assert np.isclose(np.linalg.det(self.M), 1), f"M={self.M}, det={np.linalg.det(self.M)}"
        assert self.f > 0, f"f={self.f}"
    
    def bounding_box(self) -> Tuple[Interval, Interval]:
        # return w, h of bounding box 
        # for a bounding box of x M x \leq 1
        # the bounding box would be d/sqrt(det(M)), a/sqrt(det(M))
        
        # TODO: This is wrong;
        # note that if we have M, 1 -> M, c
        # that Mprime = M/c
        # so aprime = a/c
        # and sqrt(det(Mprime)) = 1/c
        # so a/sqrt = a
        # this is totally wrong, because something should have changed when we modified the interval6
        
        a = self.M[0, 0]
        d = self.M[1, 1]
        
        det = np.linalg.det(self.M)
        
        wA = np.sqrt(d * self.f / det)
        wH = np.sqrt(a * self.f / det)
        
        width = Interval(self.center[0] - wA, self.center[0] + wA)
        height = Interval(self.center[1] - wH, self.center[1] + wH)
        
        return width, height
    
    def _point_value(self, point):
        if isinstance(point, tuple):
            assert len(point) == 2
            point = np.array([point[0], point[1]], dtype=np.float64)
        return (point - self.center) @ self.M @ (point - self.center).T
    
    def in_ellipse(self, point):
        # check if a point is in the ellipse by evaluating the point's value
        return self._point_value(point) <= self.f
    
    def make_normal_ellipse(self):
        # returns new matrix and renormalization factor
        # returns the renormalized ellipse 
        norm_factor = np.sqrt(np.linalg.det(self.M))
        
        return Ellipse(self.M / norm_factor, self.center, self.f / norm_factor)
    
    def __repr__(self) -> str:
        return f"M: {self.M}, \nCtr: {self.center} \nF: {self.f}"


def get_ellipse_coefficients(x0: np.float64, y0: np.float64, theta: np.float64, a: np.float64, b: np.float64, fatten_coeff=1.0):
    # x0, y0 are the center of the ellipse
    # theta is the angle of rotation
    # a, b are the semi-major and semi-minor axes
    # returns the coefficients of the ellipse equation
    # a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0
    
    # rescale all variables if $f$ would be too small
    assert fatten_coeff >= 1, f"fatten_coeff={fatten_coeff}"

    # first we need to rotate the ellipse
    A = a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2
    B = 2*(b**2 - a**2) * np.sin(theta) * np.cos(theta)
    C = a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2
    D = -2*A*x0 - B*y0
    E = -B*x0 - 2*C*y0
    F = A*x0**2 + B*x0*y0 + C*y0**2 - a**2*b**2
    
    F *= fatten_coeff
    
    return A, B, C, D, E, F


def make_ellipse_center(theta: float, q: float) -> Tuple[float, float]:
    """Compute the center of the ellipse given theta and q.

    Args:
        theta (float): Target angle
        q (float): Success probability

    Returns:
        Tuple: (x0, y0) center of the ellipse
    """
    assert - np.pi < theta <= np.pi, f"theta={theta}"
    assert 0 <= q <= 1, f"q={q}"
    
    mid_distance = (1 + q)/2
    x0 = mid_distance * np.cos(theta)
    y0 = mid_distance * np.sin(theta)
    
    return x0, y0

def check_in_region(x, theta: float, q: float, eps: float):
    """Checks whether a point is in our region of interest.

    Args:
        x (complex): Point of interest
        theta (float): Target theta
        q (float): Success probability
        eps (float): Accuracy

    Returns:
        Bool: Whether the point is in the region
    """
    assert - np.pi < theta <= np.pi, f"theta={theta}"
    assert 0 <= q <= 1, f"q={q}"
    assert 0 <= eps <= 1, f"eps={eps}"

    if np.abs(x) < q or np.abs(x) > 1:
        return False
    if np.abs(np.angle(x) - theta) >= eps:
        return False
    return True


########################
# Ellipse manipulation #
########################



def compute_ellipse_parameters(M):
    """Computes z, e, b from ellipse matrix M.

    Args:
        M (Matrix): 2x2 matrix representing an ellipse

    Returns:
        int, int, int: z, e, b
    """
    # TODO: change everything relying on z to instead use e^z because logarithms are expensive
    z = np.log(M[1, 1] / M[0, 0]) / (2 * np.log(constant_lambda))
    
    e = np.sqrt(M[0, 0] * M[1, 1])
    b = M[0, 1]
    
    assert np.isclose(M[0, 0], e * constant_lambda**(-z)), (M[0, 0], M[1, 1], e * constant_lambda**(-z), e, z)
    assert np.isclose(M[1, 1], e * constant_lambda**z), (M[0, 0], M[1, 1], e * constant_lambda**z, e, z)
    
    return z, e, b


class GridOperator:
    def __init__(self, val=None) -> None:
        # stores the representation of G * sqrt(2)
        if val is None:
            self.G = np.array([[one, zero], [zero, one]]) * sqrt2
        else:
            self.G = val
        
        assert self.G.shape == (2, 2), f"val={val}"
    
    def __mul__(self, other):
        if isinstance(other, GridOperator):
            return GridOperator(val=np.matmul(self.G, other.G))
        if isinstance(other, ZSqrt2Int):
            return GridOperator(val=self.G * other)
        raise NotImplementedError
    
    def __matmul__(self, other):
        if isinstance(other, GridOperator):
            # multiply and divide by sq2
            val = self.G @ other.G
            val = val * ZSqrt2Int((0, 1)) / 2
            return GridOperator(val=val)
        raise NotImplementedError
    
    def __pos__(self):
        Gbul = np.array([[~x for x in row] for row in self.G])
        return GridOperator(Gbul)
    
    def __invert__(self):
        # TODO
        raise NotImplementedError
        return GridOperator()
    
    def eval(self):
        return np.array([[x.eval() for x in row] for row in self.G])
    
    def __repr__(self) -> str:
        return f"GridOperator({self.G})"
    
    def inverse(self):
        """Given special grid operator, find its inverse

        Args:
            G (matrix of ZSq2): grid operator in G\sqrt{2} representation
        """
        a = self.G[0, 0]
        b = self.G[0, 1]
        c = self.G[1, 0]
        d = self.G[1, 1]
        
        det = a * d - b * c
        assert np.isclose(abs(det.eval()), 2), f"det={det.eval()}"
        
        det = det[0]
        
        # inverse of 2x2 matrix takes the following form:
        # 1/det * [[d, -b], [-c, a]]
        # so inv(G \sqrt{2}) = 1/sqrt(2) * inv(G)
        # so inv(G) = sqrt(2) * inv(G \sqrt{2})
        # we can muliply by det because it is +- 1
        return GridOperator(np.array([
            [d, b * -1],
            [c * -1, a]
        ]) * int(np.sign(det)))

class State:
    def __init__(self, d, delta) -> None:
        self.d = d
        self.delta = delta
        
        # self._invariant()
    
    @classmethod
    def from_parameters(cls, z, e, b, zeta, eta, beta):
        d = np.array([
            [e * constant_lambda**z, b],
            [b, e * constant_lambda**(-z)]
        ])
        
        delta = np.array([
            [eta * constant_lambda**zeta, beta],
            [beta, eta * constant_lambda**(-zeta)]
        ])
        
        return State(d, delta)
    
    def _invariant(self):
        assert np.isclose(np.linalg.det(self.d), 1, rtol=1e-2), f"d={self.d}"
        assert np.isclose(np.linalg.det(self.delta), 1, rtol=1e-2), f"delta={self.delta}"
        assert np.isclose(self.d[0, 1], self.d[1, 0]), f"d={self.d}"
        assert np.isclose(self.delta[0, 1], self.delta[1, 0]), f"delta={self.delta}"
        
        z, e, b = compute_ellipse_parameters(self.d)
        zeta, eta, beta = compute_ellipse_parameters(self.delta)
        
        assert np.isclose(e**2, b**2 + 1), f"e={e}, b={b}"
        assert np.isclose(eta**2, beta**2 + 1), f"eta={eta}, beta={beta}"
    
    def skew(self):
        b = self.d[0, 1]
        beta = self.delta[1, 0]
        
        return b**2 + beta**2
    
    def bias(self):
        z, _, _ = compute_ellipse_parameters(self.d)
        zeta, _, _ = compute_ellipse_parameters(self.delta)
        
        return zeta - z
    
    def apply(self, G: GridOperator, constant=1.0):
        G_eval = G.eval()
        Gbul_eval = (+G).eval()
        
        d_new = G_eval.T @ self.d @ G_eval * constant
        delta_new = Gbul_eval.T @ self.delta @ Gbul_eval * constant
        
        return State(d_new, delta_new)
    
    def apply_different(self, G0: GridOperator, G1: GridOperator, constant=1.0):
        G0_eval = G0.eval()
        G1_eval = G1.eval()
        
        d_new = G0_eval @ self.d @ G0_eval * constant
        delta_new = G1_eval @ self.delta @ G1_eval * constant
        
        return State(d_new, delta_new)
    
    def rescale(self, c0: float, c1: float):
        return State(self.d * c0, self.delta * c1)
    
    def compute_parameters(self):
        z, e, b = compute_ellipse_parameters(self.d)
        zeta, eta, beta = compute_ellipse_parameters(self.delta)
        
        return z, e, b, zeta, eta, beta
    
    def __repr__(self) -> str:
        return f"State(d=\n{self.d}\ndelta={self.delta})"



constant_lambda: float = 1 + np.sqrt(2)
one: ZSqrt2Int = ZSqrt2Int((1, 0))
zero: ZSqrt2Int = ZSqrt2Int((0, 0))
sqrt2: ZSqrt2Int = ZSqrt2Int((0, 1))
lam: ZSqrt2Int = ZSqrt2Int((1, 1))
inv_lam: ZSqrt2Int = ZSqrt2Int((-1, 1))

R = GridOperator(np.array([
    [one, one * -1], 
    [one, one]
]))

K = GridOperator(np.array([
    [~lam, one * -1],
    [lam, one]
]))

K_bul = +K

A = GridOperator(np.array([
    [one, one * -2], 
    [zero, one],
]))

X = GridOperator(np.array([
    [zero, one], 
    [one, zero],
]))

B = GridOperator(np.array([
    [one, sqrt2], 
    [zero, one],
]))

Z = GridOperator(np.array([
        [one, zero],
        [zero, one * -1]
]))

def find_grid_operator(s: State):
    G = GridOperator()
    
    oldskew = s.skew()
    while s.skew() > 5:
        add_G, s = step_find_grid_operator(s)
        G = G @ add_G
        
        newskew = s.skew()
        assert newskew <= 0.9 * oldskew, f"oldskew={oldskew}, newskew={newskew}"
        oldskew = newskew
    
    return G, s

def step_find_grid_operator(s: State):
    # TODO: it would be nice to rewrite again with G, s paired together... :-|
    G = GridOperator()
    
    fix_bias = False
    if np.abs(s.bias()) > 1:
        fix_bias = True
        # Lemma A.11
        
        shift_k = int(np.floor((1 - s.bias()) / 2))
        
        sk, tk, lf = _get_shift_matrices(shift_k)
        sk_inv, tk_inv, lf_inv = _get_shift_matrices(-shift_k)

        s = s.apply_different(sk, tk, constant=lf.eval())
    
    z, e, b, zeta, eta, beta = s.compute_parameters()
    
    if beta < 0:
        # fix beta; Lemma A.20
        G = G * Z
        s = s.apply(Z)
    
    if z + zeta < 0:
        # fix z + zeta TODO: Lemma?
        G = G * X
        s = s.apply(X)
    
    G, s = _step_lemma(G, s)
    
    if fix_bias:
        G = sk * G * sk * lf
        # TODO: need to check this work2
        
        # s = s.apply_different(sk_inv, tk_inv, constant=lf_inv.eval())
        s = s.apply_different(sk, tk, constant=lf.eval())
        
    
    return G, s

def _step_lemma(G: GridOperator, s: State):
    # Apply Step lemma
    z, e, b, zeta, eta, beta = s.compute_parameters()
    
    assert np.abs(s.bias()) <= 1 + 1e-6, f"bias={s.bias()}"
    assert beta >= 0, beta
    assert z + zeta >= 0, (z, zeta)
    assert s.skew() >= 5, f"skew={s.skew()}"
    
    if b >= 0:
        if -0.8 <= z <= 0.8 and -0.8 <= zeta <= 0.8:
            G, s = _step_R(G, s)
        elif z <= 0.3 and 0.8 <= zeta:
            G, s = _step_K(G, s)
        elif 0.3 <= z and 0.3 <= zeta:
            assert b >= 0 and beta >= 0, (b, beta)
            # Lemma A.17
            G, s = _step_A(G, s, z, zeta)
        elif 0.8 <= z and zeta <= 0.3:
            G, s = _step_Kbul(G, s)
        else:
            raise ValueError("This should not happen")
    else:
        if -0.8 <= z <= 0.8 and -0.8 <= zeta <= 0.8:
            G, s = _step_R(G, s)
        elif z >= -0.2 and zeta >= -0.2:
            # Lemma A.19
            assert b <= 0 and beta >= 0, (b, beta)
            assert abs(s.bias()) <= 1, s.bias()
            
            G, s = _step_B(G, s, z, zeta)
        else:
            raise ValueError("This should not happen")
    
    return G, s


def _step_A(G: GridOperator, s: State, z: float, zeta: float) -> Tuple[GridOperator, State]:
    c = min(z, zeta)
    n = int(max(1, np.floor(constant_lambda**c / 2)))

    An = GridOperator(np.array([
        [one, one * -2 * n],
        [zero, one]
    ]))
    
    G = G * An
    s = s.apply(An)
    
    return G, s

def _step_B(G: GridOperator, s: State, z: float, zeta: float)  -> Tuple[GridOperator, State]:
    c = min(z, zeta)
    n = int(max(1, np.floor(constant_lambda**c / np.sqrt(2))))
    
    Bn = GridOperator(np.array([
        [one, sqrt2 * n],
        [zero, one]
    ]))
    
    G = G * Bn
    s = s.apply(Bn)
    return G, s

def _apply_op(op: GridOperator, G: GridOperator, s: State) -> Tuple[GridOperator, State]:
    G = G @ op
    s = s.apply(op, constant=1/2)
    return G, s


def _step_K(G: GridOperator, s: State) -> Tuple[GridOperator, State]:
    return _apply_op(K, G, s)

def _step_R(G: GridOperator, s: State) -> Tuple[GridOperator, State]:
    return _apply_op(R, G, s)

def _step_Kbul(G: GridOperator, s: State) -> Tuple[GridOperator, State]:
    return _apply_op(K_bul, G, s)

def _get_shift_matrices(k: int) -> Tuple[GridOperator, GridOperator, ZSqrt2Int]:
    sk = _make_sigmak(k)
    tk = _make_tauk(k)
    lf = _compute_lf_k(k)
    
    return sk, tk, lf

def _lam_powers(k: int) -> Tuple[ZSqrt2Int, ZSqrt2Int]:
    if k < 0:
        return inv_lam**(-k)
    return lam**k

def _make_sigmak(k: int) -> GridOperator:
    return GridOperator(np.array([
        [_lam_powers(k), zero],
        [zero, one]
    ]))

def _make_tauk(k: int) -> GridOperator:    
    return GridOperator(np.array([
        [one, zero],
        [zero, _lam_powers(k) * (-1)**(k % 2)]
    ]))

def _compute_lf_k(k: int) -> ZSqrt2Int:
    return _lam_powers(-k)
