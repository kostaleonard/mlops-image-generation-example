"""Tests wgan.py."""
# pylint: disable=no-name-in-module

import os
import pytest
import numpy as np
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
    factor = 2
    good_pred = np.ones((10, 1))
    better_pred = factor * good_pred
    worse_pred = -factor * good_pred
    good_loss = WGAN._generator_loss(good_pred)
    better_loss = WGAN._generator_loss(better_pred)
    worse_loss = WGAN._generator_loss(worse_pred)
    assert better_loss < good_loss < worse_loss
    # Concrete scores.
    assert np.isclose(good_loss, -1)
    assert np.isclose(better_loss, -factor)
    assert np.isclose(worse_loss, factor)


def test_discriminator_loss() -> None:
    """Tests _discriminator_loss."""
    good_real = np.ones((10, 1))
    good_fake = -np.ones_like(good_real)
    okay_real = 0.5 * np.ones_like(good_real)
    okay_fake = -0.5 * np.ones_like(good_real)
    bad_real = good_fake
    bad_fake = good_real
    # Higher scores gets lower loss.
    assert WGAN._discriminator_loss(good_real, good_fake) < \
           WGAN._discriminator_loss(okay_real, good_fake)
    assert WGAN._discriminator_loss(good_real, good_fake) < \
           WGAN._discriminator_loss(good_real, okay_fake)
    # Completely incorrect predictions are the worst.
    assert WGAN._discriminator_loss(good_real, good_fake) < \
           WGAN._discriminator_loss(okay_real, okay_fake) < \
           WGAN._discriminator_loss(bad_real, bad_fake)
    # Concrete scores.
    assert np.isclose(WGAN._discriminator_loss(good_real, good_fake), -2)
    assert np.isclose(WGAN._discriminator_loss(good_real, okay_fake), -1.5)
    assert np.isclose(WGAN._discriminator_loss(okay_real, okay_fake), -1)
    assert np.isclose(WGAN._discriminator_loss(bad_real, bad_fake), 2)


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
