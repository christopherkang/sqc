import numpy as np
import parameters as P

from int_field import ZetaInt, ZSqrt2Int, convert_to_zeta, gcd


def test_division_zsq2():
    for _ in range(P.MANY_TRIALS):
        a = ZSqrt2Int((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)))
        b = ZSqrt2Int((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)))
        
        q, r = a / b
        assert a == b * q + r, f"a={a}, b={b}, q={q}, r={r}"

def test_gcd_zsq2():
    for _ in range(P.MANY_TRIALS):
        a = ZSqrt2Int((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)))
        b = ZSqrt2Int((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)))
        
        # print(f"a={a}, b={b}, gcd={gcd_val}")
        gcd_val = gcd(a, b)
        assert (a / gcd_val)[1] == ZSqrt2Int((0, 0)), f"a={a}, b={b}, gcd={gcd_val}"
        assert (b / gcd_val)[1] == ZSqrt2Int((0, 0)), f"a={a}, b={b}, gcd={gcd_val}"

def test_division_omega():
    for _ in range(P.MANY_TRIALS):
        v = ZetaInt((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000), np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)))
        u = ZetaInt((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000), np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)))
        
        x, r = u / v
        assert u == v * x + r, f"u={u}, v={v}, x={x}, r={r}"

def test_gcd_omega():
    for _ in range(P.MANY_TRIALS):
        v = ZetaInt((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000), np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)))
        u = ZetaInt((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000), np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)))
        
        print(f"u={u}, v={v}")
        gcd_val = gcd(u, v)
        assert (u / gcd_val)[1] == ZetaInt((0, 0, 0, 0)), f"u={u}, v={v}, gcd={gcd_val}"
        assert (v / gcd_val)[1] == ZetaInt((0, 0, 0, 0)), f"u={u}, v={v}, gcd={gcd_val}"

def test_convert_to_zeta():
    for _ in range(P.MANY_TRIALS):
        x = ZSqrt2Int((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)))
        y = ZSqrt2Int((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)))
        
        zeta = convert_to_zeta(x, y)
        assert np.isclose(x.eval() + 1j * y.eval(), zeta.eval()), f"x={x}, y={y}, zeta={zeta}"
