import numpy as np

from .ray import Ray
from .hit import Hit


class Shape:
    pass

class Sphere(Shape):
    def __init__(self, center=[0, 0, 0], radius=1.0, material=None):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.material = material

    def intersect(self, ray: Ray):

        oc = ray.origin - self.center

        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None

        sqrt_d = np.sqrt(discriminant)

        t0 = (-b - sqrt_d) / (2.0 * a)
        t1 = (-b + sqrt_d) / (2.0 * a)

        t = None

        if t0 > 1e-6:
            t = t0
        elif t1 > 1e-6:
            t = t1

        if t is None:
            return None

        pos = ray.at(t)

        normal = (pos - self.center) / self.radius

        hit = Hit(
            t=t,
            pos=pos,
            normal=normal,
            material=self.material
        )

        hit.set_face_normal(ray.direction)

        return hit

class Plane(Shape):
    def __init__(self, normal=[0, 1, 0], material=None):
        self.normal = self.normalize(np.array(normal))
        self.material = material

    def intersect(self, ray: Ray):
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) < 1e-6:
            return None
        t = -np.dot(ray.origin, self.normal) / denom
        if t < 1e-6:
            return None
        pos = ray.at(t)
        hit = Hit(
            t=t,
            pos=pos,
            normal=self.normal,
            material=self.material
        )
        hit.set_face_normal(ray.direction)
        return hit

    def normalize(self, v):
        return v / np.linalg.norm(v)

class Box(Shape):
    def __init__(self,
                min_corner=[-1, -1, -1],
                max_corner=[1, 1, 1],
                material=None):

        self.min_corner = np.array(min_corner, dtype=float)
        self.max_corner = np.array(max_corner, dtype=float)
        self.material = material

    def intersect(self, ray: Ray):
        tmin = -np.inf
        tmax = np.inf
        hit_axis = 0
        for i in range(3):
            if abs(ray.direction[i]) < 1e-8:
                if ray.origin[i] < self.min_corner[i] or ray.origin[i] > self.max_corner[i]:
                    return None
            else:
                t1 = (self.min_corner[i] - ray.origin[i]) / ray.direction[i]
                t2 = (self.max_corner[i] - ray.origin[i]) / ray.direction[i]
                near = min(t1, t2)
                far = max(t1, t2)

                if near > tmin:
                    tmin = near
                    hit_axis = i

                tmax = min(tmax, far)

        if tmin > tmax:
            return None

        t = tmin if tmin > 1e-6 else tmax

        if t < 1e-6:
            return None

        pos = ray.at(t)
        normal = np.zeros(3)
        eps = 1e-5

        if abs(pos[0] - self.min_corner[0]) < eps:
            normal = np.array([-1, 0, 0])
        elif abs(pos[0] - self.max_corner[0]) < eps:
            normal = np.array([1, 0, 0])
        elif abs(pos[1] - self.min_corner[1]) < eps:
            normal = np.array([0, -1, 0])
        elif abs(pos[1] - self.max_corner[1]) < eps:
            normal = np.array([0, 1, 0])
        elif abs(pos[2] - self.min_corner[2]) < eps:
            normal = np.array([0, 0, -1])
        else:
            normal = np.array([0, 0, 1])

        hit = Hit(
            t=t,
            pos=pos,
            normal=normal,
            material=self.material
        )

        hit.set_face_normal(ray.direction)

        return hit