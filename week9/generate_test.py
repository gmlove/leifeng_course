import tensorflow as tf
import numpy as np

from generate import plot_images


class GenerateTest(tf.test.TestCase):

    def test_generate(self):
        images = np.random.normal(size=(20, 28, 28, 1))
        plot_images(images, 'tmp/mnist.png')


if __name__ == '__main__':
    tf.test.main()
