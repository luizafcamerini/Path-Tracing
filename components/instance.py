import itertools
import numpy as np
from .ray import Ray

class Instance:
    def __init__(self, shape, material=None, transform=None):
        self.shape = shape
        self.material = material if material else shape.material
        self.transform = transform

    def intersect(self, ray):
        if self.transform is None:
            hit = self.shape.intersect(ray)
            if hit:
                hit.material = self.material
            return hit

        local_origin = self.transform.inverse_transform_point(ray.origin)
        local_direction = self.transform.inverse_transform_vector(ray.direction)
        local_ray = Ray(local_origin, local_direction)
        hit = self.shape.intersect(local_ray)

        if hit is None:
            return None

        world_pos = self.transform.transform_point(hit.pos)
        world_normal = self.transform.transform_normal(hit.normal)
        world_normal /= np.linalg.norm(world_normal)

        hit.pos = world_pos
        hit.normal = world_normal
        hit.material = self.material
        hit.t = hit.t = np.dot(world_pos - ray.origin, ray.direction)
        hit.set_face_normal(ray.direction)

        return hit

    def centroid(self):
        c = self.shape.centroid()
        if self.transform is None:
            return c
        return self.transform.transform_point(c)

    def bounding_box(self):
        bmin, bmax = self.shape.bounding_box()
        if self.transform is None:
            return bmin, bmax
        
        corners = np.array(list(itertools.product(
            [bmin[0], bmax[0]],
            [bmin[1], bmax[1]],
            [bmin[2], bmax[2]]
        )))

        transformed = np.array([
            self.transform.transform_point(p)
            for p in corners
        ])

        return (
            transformed.min(axis=0),
            transformed.max(axis=0)
        )