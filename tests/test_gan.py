"""Tests gan.py."""

import pytest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Reshape, Dense, Flatten
from imagegen.gan import GAN, GANShapeError, GANHasNoOptimizerError

TEST_NOISE = 128
TEST_HEIGHT = 1
TEST_WIDTH = 2
TEST_CHANNELS = 3
EXPECTED_GAN_ATTRIBUTES = {'generator', 'discriminator', 'model_hyperparams'}

# TODO mark slowtests


def test_init_raises_error_on_bad_generator_input_shape() -> None:
    """Tests that init raises an error on an invalid generator input shape."""
    generator = Sequential([
        # Extra first dimension.
        Input((1, TEST_NOISE)),
        Flatten(),
        Dense(TEST_HEIGHT * TEST_WIDTH * TEST_CHANNELS),
        Reshape((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS))
    ])
    discriminator = Sequential([
        Input((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)),
        Flatten(),
        Dense(1)
    ])
    generator.compile()
    discriminator.compile()
    with pytest.raises(GANShapeError):
        _ = GAN(generator, discriminator)


def test_init_raises_error_on_bad_generator_output_shape() -> None:
    """Tests that init raises an error on an invalid generator output shape."""
    generator = Sequential([
        Input((TEST_NOISE,)),
        Flatten(),
        # Flattened output.
        Dense(TEST_HEIGHT * TEST_WIDTH * TEST_CHANNELS)
    ])
    discriminator = Sequential([
        Input((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)),
        Flatten(),
        Dense(1)
    ])
    generator.compile()
    discriminator.compile()
    with pytest.raises(GANShapeError):
        _ = GAN(generator, discriminator)


def test_init_raises_error_on_generator_discriminator_shape_mismatch() -> None:
    """Tests that init raises an error when the generator output shape is not
    the same as the generator input shape."""
    generator = Sequential([
        Input((TEST_NOISE,)),
        Flatten(),
        # One fewer channel than discriminator.
        Dense(TEST_HEIGHT * TEST_WIDTH * (TEST_CHANNELS - 1)),
        Reshape((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS - 1))
    ])
    discriminator = Sequential([
        Input((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)),
        Flatten(),
        Dense(1)
    ])
    generator.compile()
    discriminator.compile()
    with pytest.raises(GANShapeError):
        _ = GAN(generator, discriminator)


def test_init_raises_error_on_bad_discriminator_output_shape() -> None:
    """Tests that init raises an error on an invalid discriminator output
    shape."""
    generator = Sequential([
        Input((TEST_NOISE,)),
        Flatten(),
        Dense(TEST_HEIGHT * TEST_WIDTH * TEST_CHANNELS),
        Reshape((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS))
    ])
    discriminator = Sequential([
        Input((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)),
        Flatten(),
        # Extra classifier output.
        Dense(2)
    ])
    generator.compile()
    discriminator.compile()
    with pytest.raises(GANShapeError):
        _ = GAN(generator, discriminator)


def test_init_raises_error_on_missing_optimizer() -> None:
    """Tests that init raises an error when the generator or discriminator is
    missing an optimizer."""
    generator = Sequential([
        Input((TEST_NOISE,)),
        Flatten(),
        Dense(TEST_HEIGHT * TEST_WIDTH * TEST_CHANNELS),
        Reshape((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS))
    ])
    discriminator = Sequential([
        Input((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)),
        Flatten(),
        Dense(1)
    ])
    # Neither generator nor discriminator have optimizers.
    with pytest.raises(GANHasNoOptimizerError):
        _ = GAN(generator, discriminator)
    generator.compile()
    # Discriminator still does not have an optimizer.
    with pytest.raises(GANHasNoOptimizerError):
        _ = GAN(generator, discriminator)
    discriminator.compile()
    _ = GAN(generator, discriminator)


def test_init_creates_expected_attributes() -> None:
    """Tests that init creates the expected object attributes."""
    generator = Sequential([
        Input((TEST_NOISE,)),
        Flatten(),
        Dense(TEST_HEIGHT * TEST_WIDTH * TEST_CHANNELS),
        Reshape((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS))
    ])
    discriminator = Sequential([
        Input((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)),
        Flatten(),
        Dense(1)
    ])
    generator.compile()
    discriminator.compile()
    gan = GAN(generator, discriminator)
    for attr_name in EXPECTED_GAN_ATTRIBUTES:
        assert hasattr(gan, attr_name)


def test_save_creates_expected_files() -> None:
    """Tests that save creates the expected model save files."""
    # TODO
    assert False


def test_load_models_match() -> None:
    """Tests that load creates a GAN whose models match those of the GAN that
    was saved."""
    # TODO
    assert False


def test_generator_loss() -> None:
    """Tests _generator_loss."""
    perfect_pred = np.ones((10, 1))
    assert GAN._generator_loss(perfect_pred) == 0
    bad_pred = 0.5 * np.ones_like(perfect_pred)
    worse_pred = 0.25 * np.ones_like(perfect_pred)
    worst_pred = np.zeros_like(perfect_pred)
    assert GAN._generator_loss(bad_pred) > 0
    assert GAN._generator_loss(worse_pred) > \
           GAN._generator_loss(bad_pred)
    assert GAN._generator_loss(worst_pred) > \
           GAN._generator_loss(worse_pred)


def test_discriminator_loss() -> None:
    """Tests _discriminator_loss."""
    perfect_real = np.ones((10, 1))
    perfect_fake = np.zeros_like(perfect_real)
    assert GAN._discriminator_loss(perfect_real, perfect_fake) == 0
    bad_real = 0.5 * np.ones_like(perfect_real)
    bad_fake = 0.5 * np.ones_like(perfect_real)
    worst_real = perfect_fake
    worst_fake = perfect_real
    # Unsure about fakes and reals.
    assert GAN._discriminator_loss(bad_real, bad_fake) > 0
    # Correctly identify all reals; unsure about fakes.
    assert GAN._discriminator_loss(bad_real, bad_fake) > \
           GAN._discriminator_loss(perfect_real, bad_fake)
    # Correctly identify all fakes; unsure about reals.
    assert GAN._discriminator_loss(perfect_real, bad_fake) == \
           GAN._discriminator_loss(bad_real, perfect_fake)
    # Completely incorrect predictions.
    assert GAN._discriminator_loss(worst_real, worst_fake) > \
           GAN._discriminator_loss(bad_real, bad_fake)


def test_train_step_updates_weights() -> None:
    """Tests that _train_step updates the generator and discriminator
    weights."""
    # TODO
    assert False


def test_train_step_returns_losses() -> None:
    """Tests that _train_step returns a tuple of generator and discriminator
    losses."""
    # TODO
    assert False


def test_train_returns_expected_training_config() -> None:
    """Tests that train returns the TrainingConfig object with the expected
    values."""
    # TODO probably slow
    assert False


def test_train_creates_model_checkpoints() -> None:
    """Tests that train creates model checkpoints when specified."""
    # TODO probably slow
    assert False


@pytest.mark.wandbtest
@pytest.mark.slowtest
def test_train_syncs_with_wandb() -> None:
    """Tests that train syncs with wandb when specified."""
    # TODO sleep for a few seconds to ensure that the run is created
    assert False


def test_generate_returns_valid_images() -> None:
    """Tests that generate returns valid images (correct shape, all values in
    [0, 1])."""
    # TODO
    assert False


def test_concatenate_images_correct_shape() -> None:
    """Tests that concatenate_images returns a tensor of the expected shape."""
    # TODO
    assert False
