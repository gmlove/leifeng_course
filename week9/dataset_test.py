from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from dataset import GANDataset


class GANDatasetTest(tf.test.TestCase):

    def test_dataset(self):
        dataset = GANDataset(np.random.normal(size=(3, 28, 28, 1)), 100, 1)
        self.assertEqual(dataset.next_batch().shape, (1, 28, 28, 1))
        self.assertTrue(dataset.has_more_than(1))
        self.assertFalse(dataset.has_more_than(2))


if __name__ == '__main__':
    tf.test.main()
