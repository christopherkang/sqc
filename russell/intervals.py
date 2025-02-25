from mpmath import mpf


class Interval:
    def __init__(self, imin: float, imax: float):
        assert imin < imax, f"imin={imin}, imax={imax}"
        self.imin = imin
        self.imax = imax
    
    @classmethod
    def from_width(cls, ctr: float, width: float):
        return Interval(ctr - width / 2, ctr + width / 2)
    
    @classmethod
    def from_pts(cls, left: float, right: float):
        if left < right:
            return Interval(left, right)
        return Interval(right, left)

    def __mul__(self, o):
        if isinstance(o, int) or isinstance(o, float) or isinstance(o, mpf):
            return Interval.from_pts(self.imin * o, self.imax * o)
        # raise NotImplementedError(f"o={o}, type={type(o)}")
    
    def __truediv__(self, o):
        if isinstance(o, int) or isinstance(o, float) or isinstance(o, mpf):
            return Interval.from_pts(self.imin / o, self.imax / o)
        raise NotImplementedError(f"o={o}, type={type(o)}")

    def __add__(self, o):
        if isinstance(o, int) or isinstance(o, float) or isinstance(o, mpf):
            return Interval.from_pts(self.imin + o, self.imax + o)
        raise NotImplementedError(f"o={o}, type={type(o)}")
    
    def __sub__(self, o):
        if isinstance(o, int) or isinstance(o, float) or isinstance(o, mpf):
            return Interval.from_pts(self.imin - o, self.imax - o)
        raise NotImplementedError(f"o={o}, type={type(o)}")
    
    def width(self):
        return self.imax - self.imin
    
    def endpoints(self):
        return self.imin, self.imax
    
    def contains(self, point, tol=1e-6):
        return self.imin - tol <= point <= self.imax + tol

    def __repr__(self) -> str:
        return f"Interval[{self.imin}, {self.imax}]"