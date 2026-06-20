import numpy as np
from .ray import Ray

class Camera:
    def __init__(self, eye, center, up, fov, aspect):
        self.eye = np.array(eye)
        self.center = np.array(center)
        self.up = np.array(up)
        self.fov = fov
        self.aspect = aspect
        self.compute_basis()

    def compute_basis(self):
        self.w = self.normalize(self.eye - self.center)
        self.u = self.normalize(np.cross(self.up, self.w))
        self.v = np.cross(self.w, self.u)

    def generate_ray(self, x, y):
        alpha = np.tan(self.fov/2) * (2*x - 1) * self.aspect
        beta = np.tan(self.fov/2) * (1 - 2*y)

        direction = alpha*self.u + beta*self.v - self.w
        return Ray(self.eye, direction)

    def normalize(self, v):
        return v / np.linalg.norm(v)