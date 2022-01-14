"""Tests wgan.py."""
# pylint: disable=no-name-in-module

import os
import pytest
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten
from imagegen.wgan import WGAN
from imagegen.errors import WGANOptimizersNotRMSPropError, \
    WGANDiscriminatorActivationNotLinearError
from tests.test_gan import TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS, \
    _get_test_generator, TEST_GENERATOR_FILENAME, TEST_DISCRIMINATOR_FILENAME


def _get_test_wgan_discriminator() -> Model:
    """Returns a test WGAN discriminator.

    :return: A test WGAN discriminator.
    """
    return Sequential([
        Input((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)),
        Flatten(),
        Dense(1)
    ])


def test_init_discriminator_nonlinear_activation_raises_error() -> None:
    """Tests that WGAN.__init__ raises an error when the provided discriminator
    has a nonlinear activation function."""
    generator = _get_test_generator()
    discriminator = Sequential([
        Input((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)),
        Flatten(),
        # Non-linear activation.
        Dense(1, activation='sigmoid')
    ])
    generator.compile(optimizer='rmsprop')
    discriminator.compile(optimizer='rmsprop')
    with pytest.raises(WGANDiscriminatorActivationNotLinearError):
        _ = WGAN(generator, discriminator)


def test_init_optimizer_not_rmsprop_raises_error() -> None:
    """Tests that WGAN.__init__ raises an error when either the generator or the
    discriminator uses an optimizer other than RMSprop."""
    generator = _get_test_generator()
    discriminator = _get_test_wgan_discriminator()
    generator.compile(optimizer='adam')
    discriminator.compile(optimizer='rmsprop')
    with pytest.raises(WGANOptimizersNotRMSPropError):
        _ = WGAN(generator, discriminator)
    generator.compile(optimizer='rmsprop')
    discriminator.compile(optimizer='adam')
    with pytest.raises(WGANOptimizersNotRMSPropError):
        _ = WGAN(generator, discriminator)


def test_load_returns_wgan() -> None:
    """Tests that load returns a WGAN object."""
    for filename in TEST_GENERATOR_FILENAME, TEST_DISCRIMINATOR_FILENAME:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
    generator = _get_test_generator()
    discriminator = _get_test_wgan_discriminator()
    generator.compile(optimizer='rmsprop')
    discriminator.compile(optimizer='rmsprop')
    wgan = WGAN(generator, discriminator)
    wgan.save(TEST_GENERATOR_FILENAME, TEST_DISCRIMINATOR_FILENAME)
    loaded = WGAN.load(TEST_GENERATOR_FILENAME, TEST_DISCRIMINATOR_FILENAME)
    assert isinstance(loaded, WGAN)


def test_generator_loss() -> None:
    """Tests _generator_loss."""
    # TODO
    assert False


def test_discriminator_loss() -> None:
    """Tests _discriminator_loss."""
    # TODO
    assert False


@pytest.mark.slowtest
def test_train_step_updates_weights() -> None:
    """Tests that _train_step updates the generator and discriminator
    weights."""
    # TODO
    assert False


@pytest.mark.slowtest
def test_train_step_clips_discriminator_weights() -> None:
    """Tests that, after a training step has finished, a WGAN's discriminator's
    weights have been clipped."""
    # TODO
    assert False
