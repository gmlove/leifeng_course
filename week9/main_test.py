from __future__ import absolute_import

import os
import shutil

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
            condition = np.random.randint(0, 10, size=(1, ))
            generated = session.run(model.generated_image, feed_dict={
                model.noise_input: noise,
                model.right_condition_input: condition
            })
            self.assertTupleEqual(generated.shape, (1, 28, 28, 1))

    def test_discriminate_real_with_right_condition(self):
        model = GANModel()
        images = np.random.normal(size=(1, 28, 28, 1))
        condition = np.random.randint(0, 10, size=(1, ))
        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            discriminate_logits = session.run(model.discriminated_real_right_logits, feed_dict={
                model.discriminator_input: images,
                model.right_condition_input: condition
            })
            self.assertTupleEqual(discriminate_logits.shape, (1, 1))

    def test_discriminate_real_with_wrong_condition(self):
        model = GANModel()
        images = np.random.normal(size=(1, 28, 28, 1))
        condition = np.random.randint(0, 10, size=(1, ))
        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            discriminate_logits = session.run(model.discriminated_real_wrong_logits, feed_dict={
                model.discriminator_input: images,
                model.wrong_condition_input: condition
            })
            self.assertTupleEqual(discriminate_logits.shape, (1, 1))

    def test_discriminate_fake(self):
        model = GANModel()
        noise = np.random.normal(size=(1, 100))
        condition = np.random.randint(0, 10, size=(1, ))
        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            discriminate_logits = session.run(model.discriminated_fake_logits, feed_dict={
                model.noise_input: noise,
                model.right_condition_input: condition
            })
            self.assertTupleEqual(discriminate_logits.shape, (1, 1))

    def test_fit(self):
        model = GANModel(100)
        dataset = GANDataset(np.random.normal(size=(3, 28, 28, 1)), np.array([1, 2, 3]), 100, 1)
        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            model.fit(session, dataset, 1, 2)

    def test_load_model(self):
        test_save_path = 'tmp/test_save'
        model = GANModel(100, save_path=test_save_path)
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)
        os.makedirs(test_save_path)
        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            model.saver.save(session, test_save_path + "/1.ckpt")
            model.load(session)


if __name__ == '__main__':
    tf.test.main()
