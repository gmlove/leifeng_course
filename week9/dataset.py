
import numpy as np


class GANDataset(object):

    def __init__(self, images, labels, noise_len, batch_size):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.noise_len = noise_len
        self._total_batch_num = len(images) // batch_size
        self.current_batch = 0

        self.last_right_label = None
        self.last_wrong_label = None
        self.last_image_batch = None

        self.last_noise_batch = None
        self.last_label_batch = None

    def has_more_than(self, count):
        return self.current_batch + count < self._total_batch_num

    def next_batch(self):
        pos = self.current_batch * self.batch_size
        self.last_image_batch = self.images[pos:pos + self.batch_size]
        self.last_right_label = self.labels[pos:pos + self.batch_size]
        self.last_wrong_label = np.random.randint(0, 9, size=(self.batch_size))
        self.last_wrong_label = [l if l < self.last_right_label[i] else l + 1 for i, l in enumerate(self.last_wrong_label)]
        self.current_batch += 1
        return self.last_image_batch, self.last_right_label, self.last_wrong_label

    def next_noise(self):
        self.last_noise_batch = np.random.uniform(low=-1.0, size=(self.batch_size, self.noise_len))
        return self.last_noise_batch

    def next_label(self):
        self.last_label_batch = np.random.randint(0, 10, size=(self.batch_size))
        return self.last_label_batch

    def reset(self):
        self.current_batch = 0
        self.last_right_label = None
        self.last_wrong_label = None
        self.last_image_batch = None

        self.last_noise_batch = None
        self.last_label_batch = None

        indices = np.random.permutation(len(self.images))
        self.images = self.images[indices]
        self.labels = self.labels[indices]
