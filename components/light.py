import numpy as np
from .ray import Ray
from .hit import Hit

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
        shadow_ray = Ray(p + 1e-3 * l, l)
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
    @property
    def emission(self):
        return np.array([
            self.power,
            self.power,
            self.power
        ])
    
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
        wi = s - p
        dist = np.linalg.norm(wi)
        if dist < 1e-6:
            return np.zeros(3), np.zeros(3)
        wi /= dist
        shadow_ray = Ray(
            p + scene.epsilon * wi,
            wi
        )
        hit = scene.compute_intersection(shadow_ray)
        if hit is not None and hit.t < dist - scene.epsilon:
            return np.zeros(3), np.zeros(3)
        cos_light = max(0.0, np.dot(-wi, n_s))
        if cos_light <= 0.0:
            return np.zeros(3), np.zeros(3)
        area = self.get_area()
        Li = (
            self.power
            * area
            * cos_light
            / (dist * dist * self.samples)
        )
        return np.array([Li, Li, Li]), wi

    def centroid(self):
        return self.position.copy()

    def bounding_box(self):
        corners = []
        for du in (-0.5, 0.5):
            for dv in (-0.5, 0.5):
                p = (
                    self.position
                    + du*self.width*self.u
                    + dv*self.height*self.v
                )
                corners.append(p)
        corners = np.array(corners)
        eps = 1e-4
        return (
            corners.min(axis=0)-eps,
            corners.max(axis=0)+eps
        )

    def intersect(self, ray):
        denom = np.dot(ray.direction, self.normal)
        if abs(denom) < 1e-6:
            return None
        t = np.dot(
            self.position-ray.origin,
            self.normal
        ) / denom
        if t < 1e-3:
            return None
        p = ray.at(t)
        local = p-self.position
        u = np.dot(local,self.u)
        v = np.dot(local,self.v)
        if abs(u)>self.width*0.5:
            return None
        if abs(v)>self.height*0.5:
            return None
        hit = Hit(
            t=t,
            pos=p,
            normal=self.normal,
            material=self
        )
        hit.set_face_normal(ray.direction)
        return hit