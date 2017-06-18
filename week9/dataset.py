
import numpy as np


class GANDataset(object):

    def __init__(self, images, noise_len, batch_size):
        self.images = images
        self.batch_size = batch_size
        self.noise_len = noise_len
        self._total_batch_num = len(images) // batch_size
        self.current_batch = 0
        self.last_noise_batch = None
        self.last_image_batch = None

    def has_more_than(self, count):
        return self.current_batch + count < self._total_batch_num

    def next_batch(self):
        pos = self.current_batch * self.batch_size
        self.last_image_batch = self.images[pos:pos + self.batch_size]
        self.current_batch += 1
        return self.last_image_batch

    def next_noise(self):
        self.last_noise_batch = np.random.uniform(low=-1.0, size=(self.batch_size, self.noise_len))
        return self.last_noise_batch

    def reset(self):
        self.current_batch = 0
        self.last_image_batch = None
        self.last_noise_batch = None
        np.random.shuffle(self.images)
