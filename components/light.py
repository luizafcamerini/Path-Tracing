import numpy as np
from .ray import Ray

class Light():
    def __init__(self, power: int):
        self.power = power


class PointLight(Light):
    def __init__(self, pos, power):
        super().__init__(power)
        self.pos = np.array(pos)

    def radiance(self, scene, p):
        l = self.pos - p
        l = l / np.linalg.norm(l)
        shadow_ray = Ray(p + 1e-5 * l, l)
        hit = scene.compute_intersection(shadow_ray)
        if hit is not None:
            return np.array([0, 0, 0]), np.array([0, 0, 0])
        r = np.linalg.norm(self.pos - p)
        Li = self.power / (r**2)
        return np.array([Li, Li, Li]), l
    
    def get_sample(self):
        return self.pos, np.array([0.0, 1.0, 0.0])
    
    def get_area(self):
        return 1.0


class AreaLight(Light):
    def __init__(self, position, normal, width, height, power, samples=16):
        super().__init__(power)
        self.position = np.array(position)
        normal = np.array(normal)
        self.normal = normal / np.linalg.norm(normal)
        self.width = width
        self.height = height
        self.samples = samples
        self.sample_index = 0
        
        self.grid_size = int(np.sqrt(self.samples))
        self.samples = self.grid_size ** 2
        
        if abs(self.normal[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])
        
        self.u = np.cross(self.normal, temp)
        self.u = self.u / np.linalg.norm(self.u)
        self.v = np.cross(self.normal, self.u)
        self.v = self.v / np.linalg.norm(self.v)
    
    def get_sample(self, p=None):
        u = np.random.uniform(-self.width / 2, self.width / 2)
        v = np.random.uniform(-self.height / 2, self.height / 2)
        
        sample_point = self.position + u * self.u + v * self.v
        return sample_point, self.normal
    
    def get_area(self):
        return self.width * self.height
    
    def reset_sample_index(self):
        self.sample_index = 0
    
    def sample_radiance(self, scene, p):
        s, n_s = self.get_sample(p)
        l = s - p
        l_norm = np.linalg.norm(l)
        l = l / l_norm
        shadow_ray = Ray(p + 1e-5 * l, l)
        hit = scene.compute_intersection(shadow_ray)
        if hit is not None:
            return np.array([0, 0, 0]), np.array([0, 0, 0])
        cos_theta = max(0, np.dot(-l, n_s))
        area = self.get_area()
        Li = self.power * cos_theta * area / (self.samples * l_norm**2)
        return np.array([Li, Li, Li]), l