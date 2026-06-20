import numpy as np

class Transform:
    def __init__(self,
                translation=np.zeros(3),
                rotation=np.zeros(3),
                scale=np.ones(3)):

        self.translation = np.array(translation, dtype=float)
        self.rotation = np.radians(rotation)
        self.scale = np.array(scale, dtype=float)

        self.matrix = self.build_matrix()
        self.inverse = np.linalg.inv(self.matrix)
        self.inverse_transpose = self.inverse[:3, :3].T

    def build_matrix(self):
        sx, sy, sz = self.scale
        tx, ty, tz = self.translation
        rx, ry, rz = self.rotation

        S = np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ])

        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1]
        ])

        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

        return T @ Rz @ Ry @ Rx @ S

    def transform_point(self, p):
        p4 = np.append(p, 1.0)
        return (self.matrix @ p4)[:3]

    def transform_vector(self, v):
        v4 = np.append(v, 0.0)
        return (self.matrix @ v4)[:3]

    def inverse_transform_point(self, p):
        p4 = np.append(p, 1.0)
        return (self.inverse @ p4)[:3]

    def inverse_transform_vector(self, v):
        v4 = np.append(v, 0.0)
        return (self.inverse @ v4)[:3]

    def transform_normal(self, n):
        n = self.inverse_transpose @ n
        return n / np.linalg.norm(n)