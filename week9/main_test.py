from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from main import GANModel
from dataset import GANDataset


class GANModelTest(tf.test.TestCase):

    def test_generate(self):
        model = GANModel()
        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            noise = np.random.normal(size=(1, 100))
            generated = session.run(model.generated_image, feed_dict={model.noise_input: noise})
            self.assertTupleEqual(generated.shape, (1, 28, 28, 1))

    def test_discriminate_real(self):
        model = GANModel()
        images = np.random.normal(size=(1, 28, 28, 1))
        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            discriminate_logits = session.run(model.discriminated_real_logits, feed_dict={
                model.discriminator_input: images
            })
            self.assertTupleEqual(discriminate_logits.shape, (1, 1))

    def test_discriminate_fake(self):
        model = GANModel()
        noise = np.random.normal(size=(1, 100))
        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            discriminate_logits = session.run(model.discriminated_fake_logits, feed_dict={
                model.noise_input: noise
            })
            self.assertTupleEqual(discriminate_logits.shape, (1, 1))

    def test_fit(self):
        model = GANModel(100)
        dataset = GANDataset(np.random.normal(size=(3, 28, 28, 1)), 100, 1)
        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            model.fit(session, dataset, 1, 2)


if __name__ == '__main__':
    tf.test.main()
