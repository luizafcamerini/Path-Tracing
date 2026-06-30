import numpy as np
from components.bvh import BVHNode

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.ambient_light = np.array([0.0, 0.0, 0.0])
        self.max_depth = 20
        self.min_depth = 4
        self.rr_probability = 0.8
        self.epsilon = 1e-3
        self.bvh = None
        self.bvh_dirty = True

    def add_object(self, obj):
        self.objects.append(obj)
        self.bvh_dirty = True

    def add_light(self, light):
        self.lights.append(light)

    def build_bvh(self):
        if not self.bvh_dirty:
            return
        if len(self.objects) == 0:
            self.bvh = None
            return
        self.bvh = BVHNode(self.objects)
        self.bvh_dirty = False

    def compute_intersection(self, ray):
        if self.bvh_dirty:
            self.build_bvh()
        if self.bvh is not None:
            return self.bvh.intersect(ray)
        closest_hit = None
        min_t = float("inf")
        return self.bvh.intersect(ray)

    def trace_ray(self, ray, depth=0):
        if depth >= self.max_depth:
            return np.zeros(3)
        hit = self.compute_intersection(ray)
        if hit is None:
            return np.zeros(3)
        return hit.material.eval(self, hit, ray.origin, depth)