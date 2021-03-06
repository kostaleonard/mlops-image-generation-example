"""Tests gan.py."""
# pylint: disable=no-name-in-module, protected-access

import os
from time import sleep
import pytest
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Dense, Flatten
from tensorflow.keras.callbacks import History
from mlops.dataset.versioned_dataset import VersionedDataset
from imagegen.publish_dataset import DATASET_VERSION
from imagegen.train_model import get_baseline_gan
from imagegen.gan import GAN, GANShapeError, GANHasNoOptimizerError, \
    WANDB_PROJECT_TITLE
from tests.test_train_model import TEST_DATASET_PUBLICATION_PATH_LOCAL, \
    _create_dataset
from tests.wandb_api import is_wandb_connected, get_last_wandb_project_run

# For the purposes of several tests, we have the following models.
# Generator: Takes length 4 noise vectors as input and outputs 1 x 2 x 3 images.
# Discriminator: Takes 1 x 2 x 3 images as input and outputs length 1 classes.
TEST_NOISE = 4
TEST_HEIGHT = 1
TEST_WIDTH = 2
TEST_CHANNELS = 3
EXPECTED_GAN_ATTRIBUTES = {'generator', 'discriminator', 'model_hyperparams'}
TEST_GENERATOR_FILENAME = '/tmp/test_gan/generator.h5'
TEST_DISCRIMINATOR_FILENAME = '/tmp/test_gan/discriminator.h5'
TEST_CKPT_PREFIX = '/tmp/test_gan/ckpt'
TEST_CKPT_GENERATOR = f'{TEST_CKPT_PREFIX}_generator.h5'
TEST_CKPT_DISCRIMINATOR = f'{TEST_CKPT_PREFIX}_discriminator.h5'
WANDB_SYNC_TIME_SECONDS = 10


def _get_test_generator() -> Model:
    """Returns a test GAN generator.

    :return: A test GAN generator.
    """
    return Sequential([
        Input((TEST_NOISE,)),
        Flatten(),
        Dense(TEST_HEIGHT * TEST_WIDTH * TEST_CHANNELS, activation='sigmoid'),
        Reshape((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS))
    ])


def _get_test_discriminator() -> Model:
    """Returns a test GAN discriminator.

    :return: A test GAN discriminator.
    """
    return Sequential([
        Input((TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])


def _gan_architecture_is_equal(gan1: GAN, gan2: GAN) -> bool:
    """Returns True if the two GANs have the same architecture, False otherwise.

    :param gan1: The first GAN.
    :param gan2: The second GAN.
    :return: True if the two GANS have the same architecture, False otherwise.
    """
    gen1_summary = []
    gan1.generator.summary(print_fn=gen1_summary.append)
    gen2_summary = []
    gan2.generator.summary(print_fn=gen2_summary.append)
    dis1_summary = []
    gan1.discriminator.summary(print_fn=dis1_summary.append)
    dis2_summary = []
    gan2.discriminator.summary(print_fn=dis2_summary.append)
    return gen1_summary == gen2_summary and dis1_summary == dis2_summary


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
    generator = _get_test_generator()
    discriminator = _get_test_discriminator()
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
    generator = _get_test_generator()
    discriminator = _get_test_discriminator()
    generator.compile()
    discriminator.compile()
    gan = GAN(generator, discriminator)
    for attr_name in EXPECTED_GAN_ATTRIBUTES:
        assert hasattr(gan, attr_name)


def test_save_creates_expected_files() -> None:
    """Tests that save creates the expected model save files."""
    for filename in TEST_GENERATOR_FILENAME, TEST_DISCRIMINATOR_FILENAME:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
    generator = _get_test_generator()
    discriminator = _get_test_discriminator()
    generator.compile()
    discriminator.compile()
    gan = GAN(generator, discriminator)
    gan.save(TEST_GENERATOR_FILENAME, TEST_DISCRIMINATOR_FILENAME)
    assert os.path.exists(TEST_GENERATOR_FILENAME)
    assert os.path.exists(TEST_DISCRIMINATOR_FILENAME)


def test_load_models_match() -> None:
    """Tests that load creates a GAN whose models match those of the GAN that
    was saved."""
    for filename in TEST_GENERATOR_FILENAME, TEST_DISCRIMINATOR_FILENAME:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
    generator = _get_test_generator()
    discriminator = _get_test_discriminator()
    generator.compile()
    discriminator.compile()
    gan = GAN(generator, discriminator)
    gan.save(TEST_GENERATOR_FILENAME, TEST_DISCRIMINATOR_FILENAME)
    loaded_gan = GAN.load(TEST_GENERATOR_FILENAME, TEST_DISCRIMINATOR_FILENAME)
    assert _gan_architecture_is_equal(gan, loaded_gan)


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


@pytest.mark.slowtest
def test_train_step_updates_weights() -> None:
    """Tests that _train_step updates the generator and discriminator
    weights."""
    # pylint: disable=no-member
    _create_dataset()
    dataset = VersionedDataset(os.path.join(
        TEST_DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION))
    gan = get_baseline_gan(dataset)
    img_batch = dataset.X_train[:2]
    gen_weights_before = gan.generator.trainable_variables[-1].numpy()
    dis_weights_before = gan.discriminator.trainable_variables[-1].numpy()
    _ = gan._train_step(img_batch)
    gen_weights_after = gan.generator.trainable_variables[-1].numpy()
    dis_weights_after = gan.discriminator.trainable_variables[-1].numpy()
    assert not (gen_weights_after == gen_weights_before).all()
    assert not (dis_weights_after == dis_weights_before).all()


@pytest.mark.slowtest
def test_train_step_returns_losses() -> None:
    """Tests that _train_step returns a tuple of generator and discriminator
    losses."""
    # pylint: disable=no-member
    _create_dataset()
    dataset = VersionedDataset(os.path.join(
        TEST_DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION))
    gan = get_baseline_gan(dataset)
    img_batch = dataset.X_train[:2]
    gen_loss, dis_loss = gan._train_step(img_batch)
    assert gen_loss > 0
    assert dis_loss > 0


@pytest.mark.slowtest
def test_train_returns_expected_training_config() -> None:
    """Tests that train returns the TrainingConfig object with the expected
    values."""
    _create_dataset()
    dataset = VersionedDataset(os.path.join(
        TEST_DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION))
    gan = get_baseline_gan(dataset)
    train_kwargs = {'epochs': 3, 'batch_size': 8}
    train_config = gan.train(dataset, use_wandb=False, **train_kwargs)
    assert isinstance(train_config.history, History)
    for name, value in train_kwargs.items():
        assert train_config.train_args[name] == value


@pytest.mark.slowtest
def test_train_creates_model_checkpoints() -> None:
    """Tests that train creates model checkpoints when specified."""
    for filename in TEST_CKPT_GENERATOR, TEST_CKPT_DISCRIMINATOR:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
    _create_dataset()
    dataset = VersionedDataset(os.path.join(
        TEST_DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION))
    gan = get_baseline_gan(dataset)
    _ = gan.train(dataset,
                  epochs=1,
                  use_wandb=False,
                  model_checkpoint_prefix=TEST_CKPT_PREFIX)
    assert os.path.exists(TEST_CKPT_GENERATOR)
    assert os.path.exists(TEST_CKPT_DISCRIMINATOR)


@pytest.mark.wandbtest
@pytest.mark.slowtest
def test_train_syncs_with_wandb() -> None:
    """Tests that train syncs with wandb when specified."""
    assert is_wandb_connected()
    _create_dataset()
    dataset = VersionedDataset(os.path.join(
        TEST_DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION))
    gan = get_baseline_gan(dataset)
    train_config = gan.train(dataset, epochs=3, use_wandb=True)
    sleep(WANDB_SYNC_TIME_SECONDS)
    last_run = get_last_wandb_project_run(WANDB_PROJECT_TITLE)
    wandb_history = last_run.history(pandas=False)
    assert train_config.history.history['loss'][0] == wandb_history[0]['loss']


def test_generate_returns_valid_images() -> None:
    """Tests that generate returns valid images (correct shape, all values in
    [0, 1])."""
    generator = _get_test_generator()
    discriminator = _get_test_discriminator()
    generator.compile()
    discriminator.compile()
    gan = GAN(generator, discriminator)
    num_samples = 4
    images = gan.generate(num_samples)
    assert images.shape == (num_samples, TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)
    assert images.min() >= 0
    assert images.max() <= 1


def test_concatenate_images_correct_shape() -> None:
    """Tests that concatenate_images returns a tensor of the expected shape."""
    generator = _get_test_generator()
    discriminator = _get_test_discriminator()
    generator.compile()
    discriminator.compile()
    gan = GAN(generator, discriminator)
    rows = 2
    cols = 4
    num_samples = rows * cols
    images = gan.generate(num_samples)
    concatenated = GAN.concatenate_images(images, rows, cols)
    assert concatenated.shape == (TEST_HEIGHT * rows,
                                  TEST_WIDTH * cols,
                                  TEST_CHANNELS)
