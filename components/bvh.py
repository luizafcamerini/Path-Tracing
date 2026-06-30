import numpy as np

from components.aabb import AABB, surrounding_box

class BVHNode:
    __slots__ = (
        "left",
        "right",
        "box"
    )
    
    def __init__(self, objects):
        if len(objects) == 0:
            raise ValueError("BVHNode recebeu uma lista vazia.")
        mins = []
        maxs = []
        for obj in objects:
            bmin, bmax = obj.bounding_box()
            mins.append(bmin)
            maxs.append(bmax)
        mins = np.asarray(mins)
        maxs = np.asarray(maxs)
        extent = maxs.max(axis=0) - mins.min(axis=0)
        axis = int(np.argmax(extent))
        
        objects.sort(key=lambda obj: obj.centroid()[axis])
        n = len(objects)
        
        if n == 1:
            self.left = objects[0]
            self.right = None

        elif n == 2:
            self.left = objects[0]
            self.right = objects[1]

        else:
            mid = n // 2
            self.left = BVHNode(objects[:mid])
            self.right = BVHNode(objects[mid:])

        left_box = self._box(self.left)
        right_box = self._box(self.right)
        self.box = surrounding_box(left_box, right_box)

    def _box(self, obj):
        if obj is None:
            return None
        if isinstance(obj, BVHNode):
            return obj.box
        min, max = obj.bounding_box()
        return AABB(min, max)

    def intersect(self, ray, tmin=1e-4, tmax=np.inf):
        if self.box is None:
            return None
        if not self.box.hit(ray, tmin, tmax):
            return None
        hit_left = None
        hit_right = None

        if self.left is not None:
            hit_left = self.left.intersect(ray)
            if hit_left is not None:
                tmax = hit_left.t

        if self.right is not None:
            hit_right = self.right.intersect(ray)

        if hit_left is None:
            return hit_right

        if hit_right is None:
            return hit_left

        if hit_left.t < hit_right.t:
            return hit_left

        return hit_right