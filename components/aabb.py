import numpy as np

class AABB:
    __slots__ = (
        "minimum",
        "maximum"
    )
    
    def __init__(self, minimum, maximum):
        self.minimum = np.asarray(minimum, dtype=np.float64)
        self.maximum = np.asarray(maximum, dtype=np.float64)

    def hit(self, ray, tmin, tmax):
        for axis in range(3):
            d = ray.direction[axis]
            if abs(d) < 1e-12:
                if ray.origin[axis] < self.minimum[axis]:
                    return False
                if ray.origin[axis] > self.maximum[axis]:
                    return False
                continue
            invD = 1.0/d
            t0 = (self.minimum[axis] - ray.origin[axis]) * invD
            t1 = (self.maximum[axis] - ray.origin[axis]) * invD

            if invD < 0:
                t0, t1 = t1, t0

            tmin = max(tmin, t0)
            tmax = min(tmax, t1)

            if tmax < tmin:
                return False

        return True

def surrounding_box(box0, box1):
    if box0 is None:
        return box1
    if box1 is None:
        return box0
    small = np.minimum(box0.minimum, box1.minimum)
    big = np.maximum(box0.maximum, box1.maximum)
    return AABB(small, big)