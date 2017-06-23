import shutil
import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

from main import GANModel
from dataset import GANDataset


def plot_images(images, filename='mnist.png'):
    plt.figure(figsize=(25, 5))
    for i in range(images.shape[0]):
        plt.subplot(2, 10, i+1)
        image = images[i, :, :, :]
        image = np.reshape(image, [28, 28])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')


def main(_):
    save_path = 'samples'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    with tf.device('/cpu:0'):
        model = GANModel()
        with tf.Session() as session:
            session.run([tf.global_variables_initializer()])
            model.load(session)
            for i in range(10):
                generated_image = session.run(model.generated_image, feed_dict={
                    model.noise_input: np.random.normal(size=(20, 100)),
                    model.right_condition_input: [i] * 20
                })
                plot_images(generated_image, filename='{}/mnist_{}.png'.format(save_path, i))


if __name__ == '__main__':
    tf.app.run(main=main)
