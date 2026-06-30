import numpy as np
from .ray import Ray

class Material:
    def eval(self, scene, hit, eye, depth=0):
        raise NotImplementedError
    
    def sample_direction(self, hit_normal):
        raise NotImplementedError
    
    def get_albedo(self):
        raise NotImplementedError


class DiffuseMaterial(Material):
    def __init__(self, albedo):
        self.albedo = np.array(albedo)
    
    def get_albedo(self):
        return self.albedo
    
    def sample_direction(self, hit_normal):
        u1, u2 = np.random.random(2)
        
        theta = np.arccos(np.sqrt(u1))
        phi = 2 * np.pi * u2
        
        if abs(hit_normal[0]) < 0.9:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = np.array([0.0, 1.0, 0.0])
        
        tangent = tangent - np.dot(tangent, hit_normal) * hit_normal
        inv = 1.0 / np.sqrt(np.dot(tangent, tangent))
        tangent *= inv
        bitangent = np.cross(hit_normal, tangent)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        direction = x * tangent + y * bitangent + z * hit_normal
        direction = direction / np.linalg.norm(direction)
        
        pdf = np.cos(theta) / np.pi
        
        return direction, pdf
    
    def eval(self, scene, hit, eye, depth=0):
        p = hit.pos
        n = hit.normal
        L = np.zeros(3)
        brdf = self.albedo / np.pi

        for light in scene.lights:
            Li, wi = light.sample_radiance(scene, p)
            if np.allclose(Li, 0.0):
                continue
            cos_theta = max(0.0, np.dot(n, wi))
            if cos_theta <= 0.0:
                continue
            L += brdf * Li * cos_theta

        rr_weight = self.russian_roulette(scene, depth)
        if rr_weight == 0.0:
            return L

        direction, pdf = self.sample_direction(n)
        if pdf <= 0.0:
            return L

        bounce_ray = Ray(
            p + scene.epsilon * direction,
            direction
        )

        incoming = scene.trace_ray(
            bounce_ray,
            depth + 1
        )

        cos_theta = max(0.0, np.dot(n, direction))
        L += incoming * brdf * cos_theta * rr_weight / pdf
        return L
    
    def normalize(self, v):
        return v / np.linalg.norm(v)

    def russian_roulette(self, scene, depth):
        if depth < scene.min_depth:
            return 1.0
        p = scene.rr_probability
        if np.random.random() > p:
            return 0.0
        return 1.0 / p


class PhongMaterial(Material):
    def __init__(self, ambient, diffuse, specular, shininess):
        self.ambient = np.array(ambient)
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular)
        self.shininess = shininess

    def eval(self, scene, hit, eye, depth=0):
        color = self.ambient * scene.ambient_light
        p = hit.pos
        n = self.normalize(hit.normal)
        v = self.normalize(eye - p)
        for light in scene.lights:
            if hasattr(light, 'sample_radiance'):
                Li, l = light.sample_radiance(scene, p)
            else:
                Li, l = light.radiance(scene, p)
            
            shadow_ray = Ray(p + 1e-5 * l, l)
            shadow_hit = scene.compute_intersection(shadow_ray)
            if shadow_hit is not None:
                continue
            if np.dot(n, l) > 0:
                color += self.diffuse * Li * np.dot(n, l)
                r = self.reflect(-l, n)
                color += self.specular * Li * max(0, np.dot(r, v))**self.shininess
        return color

    def reflect(self, d, n):
        return d - 2 * np.dot(d, n) * n

    def normalize(self, v):
        return v / np.linalg.norm(v)
    
    def sample_direction(self, hit_normal):
        raise NotImplementedError("PhongMaterial does not support path tracing")
    
    def get_albedo(self):
        return self.diffuse