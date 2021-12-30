"""Tests gan.py."""

import pytest
import numpy as np
from imagegen.gan import GAN, GANShapeError, GANHasNoOptimizerError

# TODO mark slowtests


def test_init_raises_error_on_bad_generator_input_shape() -> None:
    """Tests that init raises an error on an invalid generator input shape."""
    # TODO
    assert False


def test_init_raises_error_on_bad_generator_output_shape() -> None:
    """Tests that init raises an error on an invalid generator output shape."""
    # TODO
    assert False


def test_init_raises_error_on_generator_discriminator_shape_mismatch() -> None:
    """Tests that init raises an error when the generator output shape is not
    the same as the generator input shape."""
    # TODO
    assert False


def test_init_raises_error_on_bad_discriminator_output_shape() -> None:
    """Tests that init raises an error on an invalid discriminator output
    shape."""
    # TODO
    assert False


def test_init_raises_error_on_missing_optimizer() -> None:
    """Tests that init raises an error when the generator or discriminator is
    missing an optimizer."""
    # TODO test generator and discriminator
    assert False


def test_init_creates_expected_attributes() -> None:
    """Tests that init creates the expected object attributes."""
    # TODO
    assert False


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
