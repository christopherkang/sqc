from dataclasses import dataclass
from typing import Tuple
import numpy as np

import mpmath

class ZSqrt2Int:
    def __init__(self, coeffs: Tuple[int, int]) -> None:
        # TODO: make this compatible with numpy int :dizzy:
        a, b = coeffs
        
        assert isinstance(a, int) and isinstance(b, int), f"coeffs={coeffs}"
        a = int(a)
        b = int(b)
        
        self.coeffs = (a, b)
    
    def __add__(self, o):
        return ZSqrt2Int(tuple([self.coeffs[i] + o.coeffs[i] for i in range(2)]))
    
    def __sub__(self, o):
        return ZSqrt2Int(tuple([self.coeffs[i] - o.coeffs[i] for i in range(2)]))
    
    def __mul__(self, o):
        if isinstance(o, int):
            return ZSqrt2Int(tuple([self.coeffs[i] * o for i in range(2)]))
        if isinstance(o, ZSqrt2Int):
            a, b = self.coeffs
            c, d = o.coeffs
            return ZSqrt2Int((a * c + 2 * b * d, a * d + b * c))
        raise NotImplementedError(f"o={o}, type={type(o)}")
    
    def __truediv__(self, o):
        if isinstance(o, int):
            assert self.coeffs[0] % o == 0 and self.coeffs[1] % o == 0, f"self={self}, o={o}"
            return ZSqrt2Int(tuple([self.coeffs[i] // o for i in range(2)]))
        if isinstance(o, ZSqrt2Int):
            numerator = self * ~o
            denominator = o.norm()
            
            q_int = rounddiv(numerator[0], denominator)
            q_sq2 = rounddiv(numerator[1], denominator)
            
            q = ZSqrt2Int((q_int, q_sq2))
            r = self - o * q
            
            return q, r
            
            # a, b = self.coeffs
            # c, d = o.coeffs
            # assert self.test_div(o), f"self={self}, o={o}"
            
            # u0 = (c * a - 2 * b * d) / (a**2 - 2 * b**2)
            # u1 = (d * a - b * c) / (a**2 - 2 * b**2)
            
            # return ZSqrt2Int((u0, u1))
        raise NotImplementedError(f"o={o}, type={type(o)}")
    
    def __pow__(self, n):
        assert n >= 0, f"n={n}"
        out = ZSqrt2Int((1, 0))
        for _ in range(n):
            out *= self

        return out
    
    def eval(self):
        return self.coeffs[0] + self.coeffs[1] * np.sqrt(2)
    
    def eval_mpmath(self):
        return self.coeffs[0] + self.coeffs[1] * mpmath.sqrt(2)
    
    def div_2sq2(self):
        # outputs a, b where a + b \sqrt{2} = self / (2 + \sqrt{2})
        # note that (a + b \sqrt{2}) * (2 + \sqrt{2}) = 2a + 2b + (a + 2b) \sqrt{2} = x + y \sqrt{2}
        # so we want to solve for a, b
        # and x = 2a + 2b
        # y = a + 2b
        # so a = x - y
        # b = (y - a) / 2
        x, y = self.coeffs
        a = x - y
        
        b = y - a
        assert b & 1 == 0, f"Need even b; b={b}"
        b >>= 1
        
        return ZSqrt2Int((a, b))

    def __repr__(self) -> str:
        return f"ZSqrt2Int({self.coeffs})"
    
    def __getitem__(self, key) -> int:
        assert key in range(2), f"key={key}"
        return self.coeffs[key]
    
    def __invert__(self):
        a, b = self.coeffs
        return ZSqrt2Int((a, -b))
    
    def __eq__(self, __value: object) -> bool:
        if __value == 0:
            return self.coeffs[0] == 0 and self.coeffs[1] == 0
        return self.coeffs[0] == __value.coeffs[0] and self.coeffs[1] == __value.coeffs[1]
    
    def __ne__(self, __value: object) -> bool:
        return not self.__eq__(__value)
    
    def test_div(self, o):
        # want to see whether self / o is in ZSqrt2Int
        # i.e. (c + d\sqrt{2}) = (u0 + u1 \sqrt{2})(a + b \sqrt{2})
        # c = u0 a + 2 u1 b
        # d = u0 b + u1 a
        
        c, d = self.coeffs
        a, b = o.coeffs
        
        # theoretically u0 = (ac - 2bd) / (a^2 - 2b^2)
        # and u1 = (ad - bc) / (a^2 - 2b^2)
        # so check integer divisibility
        
        remainder_0 = (c * a - 2 * b * d) % (a**2 - 2 * b**2)
        remainder_1 = (d * a - b * c) % (a**2 - 2 * b**2)
        
        return remainder_0 == 0 and remainder_1 == 0
    
    def to_zeta(self):
        a, b = self.coeffs
        return ZetaInt((a, b, 0, -b))
    
    def norm(self):
        a, b = self.coeffs
        return a**2 - 2 * b**2

class ZetaInt:
    def __init__(self, coeffs: Tuple[int, int, int, int]) -> None:
        a, b, c, d = coeffs
        
        assert isinstance(a, int) and isinstance(b, int) and isinstance(c, int) and isinstance(d, int), f"coeffs={coeffs}"
        
        self.coeffs = (a, b, c, d)
    
    def __getitem__(self, key) -> int:
        assert key in range(4), f"key={key}"
        return self.coeffs[key]
        
    def eval(self):
        zeta = np.exp(1j * 2 * np.pi / 8)
        return sum([self.coeffs[i] * zeta**i for i in range(4)])
    
    def eval_mpmath(self):
        zeta = mpmath.exp(1j * 2 * mpmath.pi / 8)
        return mpmath.fsum([self.coeffs[i] * zeta**i for i in range(4)])

    def __add__(self, o):
        assert isinstance(o, ZetaInt), f"o={o}, type={type(o)}"
        return ZetaInt(tuple([self.coeffs[i] + o.coeffs[i] for i in range(4)]))
    
    def __sub__(self, o):
        assert isinstance(o, ZetaInt), f"o={o}, type={type(o)}"
        return ZetaInt(tuple([self.coeffs[i] - o.coeffs[i] for i in range(4)]))

    def __mul__(self, o):
        if isinstance(o, int):
            return ZetaInt(tuple([self.coeffs[i] * o for i in range(4)]))
        if isinstance(o, ZetaInt):
            coeffs = [0, 0, 0, 0]
            for i in range(4):
                for j in range(4):
                    idx = (i + j) & 0b11
                    has_phase = (i + j) >> 2
                    phase = (-1)**has_phase
                    
                    coeffs[idx] += self.coeffs[i] * o.coeffs[j] * phase
            
            return ZetaInt(tuple(coeffs))
        if isinstance(o, ZSqrt2Int):
            a, b, c, d = self.coeffs
            int_component = self * ZetaInt((o[0], 0, 0, 0))
            
            # homeomorphism = np.array([
            #     [0, 1, 0, -1], 
            #     [1, 0, 1, 0], 
            #     [0, 1, 0, 1], 
            #     [-1, 0, 1, 0],
            # ])

            # sqrt2_component = homeomorphism @ np.array(self.coeffs)
            
            b1 = b - d
            b2 = a + c
            b3 = b + d
            b4 = -a + c
            
            return int_component + ZetaInt((b1, b2, b3, b4)) * o[1]
        raise NotImplementedError(f"o={o}, type={type(o)}")
    
    def __truediv__(self, o):
        if isinstance(o, int):
            assert self.coeffs[0] % o == 0 and self.coeffs[1] % o == 0 and self.coeffs[2] % o == 0 and self.coeffs[3] % o == 0, f"self={self}, o={o}"
            return ZetaInt(tuple([self.coeffs[i] // o for i in range(4)]))
        if isinstance(o, ZetaInt):
            # we need to do a similar trick as with the other divide function
            # i.e. u / v = 
            #      u * v^* / v * v^* = x
            # a / b = a * b^\dagger / b * b^\dagger = a * b^\dagger * ~(b *  b^\dagger) / b * b^\dagger * ~(b * b^\dagger)
            # first, get the real norm of the divisor
            
            v_v_dag, im = (o * ~o).to_zsq()
            assert im == ZSqrt2Int((0, 0)), f"im={im}"
            
            v_bul = ~v_v_dag
            v_bul = v_bul.to_zeta()
            
            numerator = self * (~o) * v_bul
            denominator = o.norm()
            
            x0, x1, x2, x3 = numerator.coeffs
            x0 = rounddiv(x0, denominator)
            x1 = rounddiv(x1, denominator)
            x2 = rounddiv(x2, denominator)
            x3 = rounddiv(x3, denominator)
            
            q = ZetaInt((x0, x1, x2, x3))
            return q, self - o * q
    
    def __floordiv__(self, o):
        q, _ = self.__truediv__(o)
        return q
    
    def to_zsq(self):
        # UNSAFE! Converts to ZSqrt2Int
        # returns a, b where a + bi = self and a, b are ZSqrt2Order
        # a0 + a1 zeta + a2 zeta^2 + a3 zeta^3
        # a0 + (1 + i)/\sqrt{2} a1 + i a2 + (-1 + i)/\sqrt{2} a3
        # a0 + a1/\sqrt{2} + a1/\sqrt{2} i + a2 i - a3/\sqrt{2} + a3/\sqrt{2} i
        # a0 + a1\sqrt{2}/2 - a3\sqrt{2}/2 + (a1 + a3)i\sqrt{2}/2 + a2 i
        a0, a1, a2, a3 = self.coeffs

        real_sq2 = a1 - a3
        assert real_sq2 & 1 == 0, f"Need even real_sq2={real_sq2}"
        real_sq2 >>= 1
        
        imag_sq2 = (a1 + a3)
        assert imag_sq2 & 1 == 0, f"Need even imag_sq2={imag_sq2}"
        imag_sq2 >>= 1
        
        real = ZSqrt2Int((a0, real_sq2))
        imag = ZSqrt2Int((a2, imag_sq2))
        
        return real, imag
    
    def abs_sq(self) -> ZSqrt2Int:
        # returns magnitude squared
        # note that z = a0 + a1 zeta + a2 zeta^2 + a3 zeta^3
        # so z = a0 + (1 + i)/sqrt{2} a1 + i a2 + (-1 + i)/sqrt{2} a3
        # = a0 + a1/sqrt{2} - a3/\sqrt{2} + (a1/sqrt{2} + a2 + a3/sqrt{2}) i
        # so re^2 = 1/2 (a1 - a3 + a0\sqrt{2})**2
        #         = 1/2 (a1^2 - 2a1a3 + a3^2 + 2 a0^2 + 2 (a1 - a3) a0 \sqrt{2})
        # +  im^2 = 1/2 (a1 + a3 + a2\sqrt{2})**2
        #         = 1/2 (a1^2 + 2a1a3 + a3^2 + 2 a2^2 + 2 (a1 + a3) a2 \sqrt{2})
        # so that the sum is
        # 1/2 (2 a1^2 + 2 a3^2 + 2 a0^2 + 2a2^2 + 2(a1 - a3) a0 \sqrt{2} + 2 (a1 + a3) a2 \sqrt{2})
        # or
        # a0^2 + a1^2 + a2^2 + a3^2, (a1 - a3) a0 + (a1 + a3) a2
        a0, a1, a2, a3 = self.coeffs

        return ZSqrt2Int((a0**2 + a1**2 + a2**2 + a3**2, (a1 - a3) * a0 + (a1 + a3) * a2))
    
    def __repr__(self) -> str:
        return f"ZetaInt({self.coeffs})"

    def __eq__(self, __value: object) -> bool:
        if __value == 0:
            return self.coeffs[0] == 0 and self.coeffs[1] == 0 and self.coeffs[2] == 0 and self.coeffs[3] == 0
        return self.coeffs[0] == __value.coeffs[0] and self.coeffs[1] == __value.coeffs[1] and self.coeffs[2] == __value.coeffs[2] and self.coeffs[3] == __value.coeffs[3]
    
    def __ne__(self, __value: object) -> bool:
        return not self.__eq__(__value)
    
    def __invert__(self):
        # inversion over i
        return ZetaInt((self.coeffs[0], -self.coeffs[3], -self.coeffs[2], -self.coeffs[1]))
    
    def __pos__(self):
        # inversion over sq2
        return ZetaInt((self.coeffs[0], -self.coeffs[1], self.coeffs[2], -self.coeffs[3]))
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def norm(self):
        # (a^2+b^2+c^2+d^2)^2-2*(a*b+b*c+c*d-d*a)^2
        # a = v3
        # b = v2 
        # c = v1
        # d = v0
        
        v0, v1, v2, v3 = self.coeffs
        
        return (v0**2 + v1**2 + v2**2 + v3**2)**2 - 2 * (v3 * v2 + v2 * v1 + v1 * v0 - v0 * v3)**2
    
    def __pow__(self, n):
        assert n >= 0, f"n={n}"
        out = ZetaInt((1, 0, 0, 0))
        for _ in range(n):
            out *= self

        return out

def gcd(a, b):
    assert type(a) == type(b), f"a={a}, b={b}"
    
    k = 0
    while a != 0 and b != 0:
        _, r = a / b
        a = b
        b = r
        
        k += 1
        if k > 1000:
            raise Exception("Infinite loop")

    return a + b

#######################
### HELPER METHODS ####
#######################

def rounddiv(a: int, b: int):
    return (a + b // 2) // b 

def convert_to_zeta(x: ZSqrt2Int, y: ZSqrt2Int) -> ZetaInt:
    """Take x + yi and convert it to zeta

    Args:
        x (ZSqrt2Int): x
        y (ZSqrt2Int): y

    Returns:
        ZetaInt: x + i y
    """
    # note that:
    # x = d + (c - a)/2 \sqrt{2}
    # y = b + (c + a)/2 \sqrt{2} i
    # for a zeta^3 + b zeta^2 + c zeta + d
    
    # so we can identify a, b, c, d
    d = x[0]
    b = y[0]
    c_min_a_2 = x[1]
    c_plus_a_2 = y[1]
    
    c = c_min_a_2 + c_plus_a_2
    a = c_plus_a_2 - c_min_a_2
    
    return ZetaInt((d, c, b, a))
