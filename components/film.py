from PIL import Image
import random
import numpy as np

class Film:
    def __init__(self, width, height, samples=1):
        self.width = width
        self.height = height
        self.samples = samples
        self.image = np.zeros((height, width, 3))

    def sample_count(self):
        return self.samples

    def get_sample(self, i, j):
        return (
            (i + random.random()) / self.width,
            (j + random.random()) / self.height
        )

    def set_pixel(self, i, j, color):
        self.image[j, i] = np.clip(color, 0, 1)

    def save(self, filename):
        img = (self.image * 255).astype(np.uint8)
        Image.fromarray(img).save(filename)