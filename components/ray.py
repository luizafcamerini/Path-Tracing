import numpy as np

class Ray:
    def __init__(self, origin: np.array, direction: np.array):
        self.origin = origin
        self.direction = self.normalize(direction)

    def at(self, t: float):
        return self.origin + t * self.direction

    def normalize(self, v):
        return v / np.linalg.norm(v)