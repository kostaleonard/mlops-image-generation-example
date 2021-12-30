"""Tests train_model.py."""

import os
import shutil
import pytest
from mlops.dataset.versioned_dataset import VersionedDataset
from imagegen.publish_dataset import publish_dataset, DATASET_VERSION
from imagegen.gan import GAN
from imagegen import train_model

TEST_DATASET_PUBLICATION_PATH_LOCAL = '/tmp/test_train_model/datasets'
TEST_CHECKPOINT_PATH = '/tmp/test_train_model/models/checkpoints'
TEST_MODEL_PUBLICATION_PATH_LOCAL = '/tmp/test_train_model/models/versioned'


def _create_dataset() -> None:
    """Creates the dataset files."""
    try:
        shutil.rmtree(TEST_DATASET_PUBLICATION_PATH_LOCAL)
    except FileNotFoundError:
        pass
    publish_dataset(TEST_DATASET_PUBLICATION_PATH_LOCAL)


def test_get_baseline_gan_returns_gan() -> None:
    """Tests that get_baseline_gan returns a properly instantiated GAN."""
    _create_dataset()
    dataset = VersionedDataset(os.path.join(
        TEST_DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION))
    gan = train_model.get_baseline_gan(dataset)
    assert isinstance(gan, GAN)


@pytest.mark.slowtest
def test_publish_gan_creates_files() -> None:
    """Tests that publish_gan creates the published model files."""
    _create_dataset()
    try:
        shutil.rmtree(TEST_MODEL_PUBLICATION_PATH_LOCAL)
    except FileNotFoundError:
        pass
    dataset = VersionedDataset(os.path.join(
        TEST_DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION))
    gan = train_model.get_baseline_gan(dataset)
    training_config = gan.train(dataset, epochs=1, use_wandb=False)
    base_path = train_model.publish_gan(
        gan,
        dataset,
        training_config,
        TEST_MODEL_PUBLICATION_PATH_LOCAL)
    assert os.path.exists(TEST_MODEL_PUBLICATION_PATH_LOCAL)
    assert len(os.listdir(TEST_MODEL_PUBLICATION_PATH_LOCAL)) == 1
    assert os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL,
        os.listdir(TEST_MODEL_PUBLICATION_PATH_LOCAL)[0]) == base_path
    assert set(os.listdir(base_path)) == {'generator', 'discriminator'}
    generator_base_path = os.path.join(base_path, 'generator')
    discriminator_base_path = os.path.join(base_path, 'discriminator')
    assert len(os.listdir(generator_base_path)) == 1
    assert len(os.listdir(discriminator_base_path)) == 1
    generator_path = os.path.join(generator_base_path,
                                  os.listdir(generator_base_path)[0])
    discriminator_path = os.path.join(discriminator_base_path,
                                      os.listdir(discriminator_base_path)[0])
    assert set(os.listdir(generator_path)) == {'meta.json', 'model.h5'}
    assert set(os.listdir(discriminator_path)) == {'meta.json', 'model.h5'}


@pytest.mark.slowtest
def test_main_publishes_model() -> None:
    """Tests that main publishes a VersionedModel."""
    try:
        num_models_before = len(os.listdir(
            train_model.MODEL_PUBLICATION_PATH_LOCAL))
    except FileNotFoundError:
        num_models_before = 0
    train_model.main()
    num_models_after = len(os.listdir(train_model.MODEL_PUBLICATION_PATH_LOCAL))
    assert num_models_after == num_models_before + 1
