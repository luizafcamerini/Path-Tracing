import numpy as np

class Hit:
    def __init__(self, t: float, pos: np.array, normal: np.array, material, backfacing: bool = False):
        self.t = t
        self.pos = pos
        self.normal = self.normalize(normal)  
        self.material = material
        self.backfacing = backfacing

    def set_face_normal(self, ray_direction):
        if np.dot(ray_direction, self.normal) > 0:
            self.normal = -self.normal
            self.backfacing = True
        else:
            self.backfacing = False

    def normalize(self, v):
        return v / np.linalg.norm(v)