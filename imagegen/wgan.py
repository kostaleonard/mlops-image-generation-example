"""Demonstrates a WGAN architecture on the pokemon dataset. The Wasserstein GAN
(WGAN) makes modifications to the architecture and training loop to improve
stability and alleviate the problem of no gradient information passing to the
generator when the discriminator perfectly classifies a batch. According to the
Wasserstein GAN paper (https://arxiv.org/abs/1701.07875) and blog guides
(https://machinelearningmastery.com/
how-to-implement-wasserstein-loss-for-generative-adversarial-networks/), the
following changes to GAN architecture are introduced:
1. Use a linear activation function in the output layer of the critic model
(instead of sigmoid).
2. Use Wasserstein loss to train the critic and generator models that promote
larger difference between scores for real and generated images.
3. Constrain critic model weights to a limited range after each mini batch
update (e.g., [-0.01, 0.01]). Note that WGAN with Gradient Penalty (WGAN-GP)
instead introduces a gradient penalty that helps the critic converge. See
https://keras.io/examples/generative/wgan_gp/ for adding Gradient Penalty (very
easy).
4. Update the critic model more times than the generator each iteration
(e.g., 5).
5. Use the RMSProp version of gradient descent with small learning rate
(e.g., 0.00005) and no momentum. Momentum interferes with training stability.
"""
# pylint: disable=no-name-in-module

import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import linear
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, load_model as tf_load_model
from imagegen.gan import GAN
from imagegen.errors import WGANDiscriminatorActivationNotLinearError, \
    WGANOptimizersNotRMSPropError

NUM_DISCRIMINATOR_UPDATES_PER_BATCH = 5
DISCRIMINATOR_WEIGHT_CLIP = 0.01
DEFAULT_RMSPROP_LR = 5e-5


class WGAN(GAN):
    """Represents a WGAN model."""

    def __init__(self, generator: Model, discriminator: Model) -> None:
        """Instantiates the WGAN.

        :param generator: The compiled generator model. Generates new images.
            The input shape must be m x n, where m is the number of examples
            and n is the length of the noise vector that the generator uses as
            input. The output shape must be m x h x w x c, where m is the number
            of examples, h is the image height, w is the image width, and c is
            the image channels. The output must be in the range [0, 1].
        :param discriminator: The compiled discriminator model. Classifies
            images as either real or fake. The input shape must be
            m x h x w x c, the same shape as the output of the generator. The
            output shape must be m x 1, where the output represents the
            probability that each example is real. This output must be in the
            range [0, 1].
        """
        super().__init__(generator, discriminator)
        if discriminator.layers[-1].activation is not linear:
            raise WGANDiscriminatorActivationNotLinearError
        if not isinstance(generator.optimizer, RMSprop) or \
                not isinstance(discriminator.optimizer, RMSprop):
            raise WGANOptimizersNotRMSPropError

    @staticmethod
    def load(generator_filename: str, discriminator_filename: str) -> 'WGAN':
        """Returns the WGAN loaded from the generator and discriminator files.

        :param generator_filename: The path to the saved generator.
        :param discriminator_filename: The path to the saved discriminator.
        :return: The WGAN loaded from the generator and discriminator files.
        """
        generator = tf_load_model(generator_filename)
        discriminator = tf_load_model(discriminator_filename)
        return WGAN(generator, discriminator)

    @staticmethod
    def _generator_loss(fake_output: np.ndarray) -> float:
        """Returns the generator's loss based on the discriminator's predictions
        on generated images.

        :param fake_output: The discriminator's predictions on fake images
            output by the generator. If the discriminator can spot the fakes,
            these predictions will be very small (they do not represent
            probabilities); if the generator can fool the discriminator, these
            predictions will be very large. The predictions are a tensor of
            shape m x 1, where m is the batch size.
        :return: The negated mean score (i.e., prediction) on fake images; the
            generator loss is minimized when the discriminator scores fake
            images high.
        """
        return -tf.reduce_mean(fake_output)

    @staticmethod
    def _discriminator_loss(real_output: np.ndarray,
                            fake_output: np.ndarray) -> float:
        """Returns the discriminator's loss on batches of real and fake images.

        :param real_output: The discriminator's predictions on real images. If
            the discriminator can spot real images, these predictions will be
            very large (they do not represent probabilities). The predictions
            are a tensor of shape m x 1, where m is the batch size.
        :param fake_output: The discriminator's predictions on fake images
            output by the generator. If the discriminator can spot the fakes,
            these predictions will be very small (they do not represent
            probabilities); if the generator can fool the discriminator, these
            predictions will be very large. The predictions are a tensor of
            shape m x 1, where m is the batch size.
        :return: The difference between the mean score (i.e., prediction) on
            fake images and the mean score on real images; the discriminator
            loss is minimized when it correctly scores fake images low and real
            images high.
        """
        # TODO every implementation of this that I can find does the means separately. Can I do this: return (fake_output - real_output).mean()
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    @tf.function
    def _train_step(self, X_batch: np.ndarray) -> tuple[float, float]:
        """Runs one batch of images through the model, computes the loss, and
        applies the gradients to the model; returns the generator and
        discriminator losses on the batch.

        :param X_batch: A batch of input images; a tensor of shape m x h x w x
            c, where m is the batch size, h is the image height, w is the image
            width, and c is the number of channels.
        :return: A 2-tuple of the generator and discriminator losses on the
            batch.
        """
        total_dis_loss = 0
        # Update discriminator weights (several times).
        for _ in range(NUM_DISCRIMINATOR_UPDATES_PER_BATCH):
            noise = tf.random.normal((X_batch.shape[0], self.gen_input_dim))
            with tf.GradientTape() as dis_tape:
                generated_images = self.generator(noise, training=True)
                real_output = self.discriminator(X_batch, training=True)
                fake_output = self.discriminator(generated_images,
                                                 training=True)
                dis_loss = WGAN._discriminator_loss(real_output, fake_output)
            dis_gradients = dis_tape.gradient(
                dis_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(dis_gradients, self.discriminator.trainable_variables))
            # Clip discriminator weights.
            for weight in self.discriminator.trainable_weights:
                weight.assign(tf.clip_by_value(
                    weight,
                    -DISCRIMINATOR_WEIGHT_CLIP,
                    DISCRIMINATOR_WEIGHT_CLIP))
            total_dis_loss += dis_loss
        # Update generator weights.
        noise = tf.random.normal((X_batch.shape[0], self.gen_input_dim))
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = WGAN._generator_loss(fake_output)
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))
        return gen_loss, total_dis_loss / NUM_DISCRIMINATOR_UPDATES_PER_BATCH
