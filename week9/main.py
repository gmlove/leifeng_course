import numpy as np
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib import layers as contrib_layers
from tensorflow.examples.tutorials import mnist

from dataset import GANDataset


tf.logging.set_verbosity(tf.logging.INFO)


def _build_generator(input_data, name='generator'):
    with tf.variable_scope(name) as var_scope:
        net = layers.dense(input_data, 128)
        net = tf.nn.relu(net)
        net = tf.reshape(net, [-1, 4, 4, 8])
        net = layers.conv2d_transpose(net, 128, [5, 5], activation=tf.nn.relu, strides=[2, 2], padding='same')  # 8x8
        net = layers.batch_normalization(net, momentum=0.9, training=True)
        net = layers.conv2d_transpose(net, 64, [5, 5], activation=tf.nn.relu, strides=[2, 2])  # 19x19
        net = layers.batch_normalization(net, momentum=0.9, training=True)
        net = layers.conv2d_transpose(net, 32, [5, 5], activation=tf.nn.relu)  # 23x23
        net = layers.batch_normalization(net, momentum=0.9, training=True)
        net = layers.conv2d_transpose(net, 16, [5, 5], activation=tf.nn.relu)  # 27x27
        net = layers.batch_normalization(net, momentum=0.9, training=True)
        net = layers.conv2d_transpose(net, 1, [2, 2], activation=tf.nn.relu)  # 28x28
    return net


def _build_discriminator(input_data, reuse_variables=False, name='discriminator'):
    with tf.variable_scope(name, reuse=reuse_variables) as var_scope:
        net = layers.conv2d(input_data, 16, [3, 3], strides=[2, 2], activation=tf.nn.relu, padding='same', name='conv2d_1')
        net = layers.batch_normalization(net, momentum=0.9, training=True)
        net = layers.conv2d(net, 32, [3, 3], strides=[2, 2], activation=tf.nn.relu, padding='same', name='conv2d_2')
        net = layers.batch_normalization(net, momentum=0.9, training=True)
        net = layers.conv2d(net, 64, [3, 3], strides=[2, 2], activation=tf.nn.relu, padding='same', name='conv2d_3')
        net = layers.conv2d(net, 128, [3, 3], strides=[2, 2], activation=tf.nn.relu, padding='same', name='conv2d_4')
        net = contrib_layers.flatten(net)
        net = layers.dense(net, 1)
    return net


class Callback(object):

    def __init__(self, every_step, func):
        self.every_step = every_step
        self.func = func

    def __call__(self, dataset, current_step):
        self.func(dataset, current_step)


class SummaryCallback(Callback):

    def __init__(self, session, model, data_dir='./summary/train_tf-bn', every_step=10):
        summary_writer = tf.summary.FileWriter(data_dir, session.graph)

        def func(dataset, current_step):
            summaries = session.run(model.summaries, feed_dict={
                model.noise_input: dataset.last_noise_batch,
                model.discriminator_input: dataset.last_image_batch
            })
            summary_writer.add_summary(summaries, current_step)

        super().__init__(every_step, func)


class LogCallback(Callback):

    def __init__(self, every_step=100):

        def func(dataset, current_step):
            tf.logging.info('current step: %s', current_step)

        super().__init__(every_step, func)


class GANModel(object):

    def __init__(self, noise_len=100, learning_rate=0.0002):
        self.noise_len = noise_len

        self.noise_input = tf.placeholder(tf.float32, shape=(None, self.noise_len))
        self.generated_image = _build_generator(self.noise_input)

        self.discriminator_input = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.discriminated_real_logits = _build_discriminator(self.discriminator_input)
        self.discriminated_real = tf.nn.sigmoid(self.discriminated_real_logits)

        self.discriminated_fake_logits = _build_discriminator(
            self.generated_image, reuse_variables=True)
        self.discriminated_fake = tf.nn.sigmoid(self.discriminated_fake_logits)

        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.discriminated_fake_logits, labels=tf.ones_like(self.discriminated_fake)))

        self.discriminator_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.discriminated_real_logits, labels=tf.ones_like(self.discriminated_real)))
        self.discriminator_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.discriminated_fake_logits, labels=tf.zeros_like(self.discriminated_fake)))

        self.discriminator_loss = self.discriminator_real_loss + self.discriminator_fake_loss

        all_vars = tf.trainable_variables()
        self.generator_vars = [var for var in all_vars if 'generator' in var.name]
        self.discriminator_vars = [var for var in all_vars if 'discriminator' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([var for var in update_ops if 'discriminator' in var.name]):
            self.d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(
                self.discriminator_loss, var_list=self.discriminator_vars)
        with tf.control_dependencies([var for var in update_ops if 'generator' in var.name]):
            self.g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(
                self.generator_loss, var_list=self.generator_vars)

        tf.summary.scalar('loss/p_generator', tf.reduce_mean(self.discriminated_fake))
        tf.summary.scalar('loss/p_discriminator', tf.reduce_mean(self.discriminated_real))
        tf.summary.scalar('loss/generator_loss', self.generator_loss)
        tf.summary.scalar('loss/discriminator_loss', self.discriminator_loss)
        tf.summary.scalar('loss/discriminator_real_loss', self.discriminator_real_loss)
        tf.summary.scalar('loss/discriminator_fake_loss', self.discriminator_fake_loss)
        tf.summary.image('generated_image', self.generated_image)
        tf.summary.image('real_image', self.discriminator_input)
        self.summaries = tf.summary.merge_all()

    def _fit_callbacks(self, session):
        return [
            SummaryCallback(session, self),
            LogCallback()
        ]

    def fit(self, session, dataset, epochs, k_steps, callbacks=None):
        callbacks = callbacks or []
        callbacks += self._fit_callbacks(session)

        train_step = 0
        for i in range(epochs):
            while dataset.has_more_than(k_steps + 1):
                train_step += 1
                for k in range(k_steps):
                    real_images, noise_input = dataset.next_batch(), dataset.next_noise()
                    session.run(self.d_optimizer, feed_dict={
                        self.discriminator_input: real_images,
                        self.noise_input: noise_input
                    })
                noise_input = dataset.next_noise()
                session.run(self.g_optimizer, feed_dict={
                    self.noise_input: noise_input
                })
                self._run_callbacks(callbacks, train_step, dataset)
            dataset.reset()

    def _run_callbacks(self, callbacks, current_train_step, dataset):
        callbacks = callbacks or []
        for callback in callbacks:
            if current_train_step > 0 and current_train_step % callback.every_step == 0:
                callback(dataset, current_train_step)


def main(_):
    model = GANModel()
    mnist_data = mnist.input_data.read_data_sets('./dataset/mnist', validation_size=0)
    dataset = GANDataset(np.reshape(mnist_data.train.images, (-1, 28, 28, 1)), 100, 32)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer()])
        model.fit(session, dataset, 20, 1)


if __name__ == '__main__':
    tf.app.run(main=main)
