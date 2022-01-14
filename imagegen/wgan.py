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
instead introduces a gradient penalty that helps the critic converge.
4. Update the critic model more times than the generator each iteration
(e.g., 5).
5. Use the RMSProp version of gradient descent with small learning rate
(e.g., 0.00005) and no momentum. Momentum interferes with training stability.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.activations import linear
from tensorflow.keras.optimizers import RMSprop
from imagegen.gan import GAN
from imagegen.errors import WGANDiscriminatorActivationNotLinearError, \
    WGANOptimizersNotRMSPropError


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
        # TODO test errors
        if discriminator.layers[-1].activation != linear:
            raise WGANDiscriminatorActivationNotLinearError
        if not isinstance(generator.optimizer, RMSprop) or \
                not isinstance(discriminator.optimizer, RMSprop):
            raise WGANOptimizersNotRMSPropError
