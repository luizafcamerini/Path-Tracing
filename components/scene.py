import numpy as np

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.ambient_light = np.array([0.0, 0.0, 0.0])
        self.max_depth = 20

    def add_object(self, obj):
        self.objects.append(obj)

    def add_light(self, light):
        self.lights.append(light)

    def compute_intersection(self, ray):
        closest_hit = None
        min_t = float("inf")
        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit and hit.t < min_t:
                min_t = hit.t
                closest_hit = hit
        return closest_hit

    def trace_ray(self, ray, depth=0):
        hit = self.compute_intersection(ray)

        if hit is None:
            return np.array([0.0, 0.0, 0.0])
        
        if depth >= self.max_depth:
            return np.array([0.0, 0.0, 0.0])

        return hit.material.eval(self, hit, ray.origin, depth)