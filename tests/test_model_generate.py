"""Tests model_generate.py."""

import os
import shutil
import pytest
import matplotlib
import matplotlib.pyplot as plt
from mlops.dataset.versioned_dataset import VersionedDataset
from imagegen.publish_dataset import DATASET_VERSION
from imagegen.train_model import get_baseline_gan, publish_gan, \
    main as train_model_main
from imagegen.gan import GAN
from imagegen import model_generate
from tests.test_train_model import TEST_DATASET_PUBLICATION_PATH_LOCAL, \
    TEST_MODEL_PUBLICATION_PATH_LOCAL, _create_dataset
matplotlib.use('Agg')


@pytest.mark.slowtest
def test_get_gan_returns_gan() -> None:
    """Tests that get_gan returns a GAN from the versioned model path."""
    _create_dataset()
    try:
        shutil.rmtree(TEST_MODEL_PUBLICATION_PATH_LOCAL)
    except FileNotFoundError:
        pass
    dataset = VersionedDataset(os.path.join(
        TEST_DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION))
    gan = get_baseline_gan(dataset)
    training_config = gan.train(dataset, epochs=1, use_wandb=False)
    base_path = publish_gan(
        gan,
        dataset,
        training_config,
        TEST_MODEL_PUBLICATION_PATH_LOCAL)
    loaded_gan = model_generate.get_gan(base_path)
    assert isinstance(loaded_gan, GAN)
    gen_summary = []
    gan.generator.summary(print_fn=gen_summary.append)
    loaded_gen_summary = []
    loaded_gan.generator.summary(print_fn=loaded_gen_summary.append)
    assert gen_summary == loaded_gen_summary
    dis_summary = []
    gan.discriminator.summary(print_fn=dis_summary.append)
    loaded_dis_summary = []
    loaded_gan.discriminator.summary(print_fn=loaded_dis_summary.append)
    assert dis_summary == loaded_dis_summary


def test_get_gan_invalid_path_raises_error() -> None:
    """Tests that get_gan raises an error when attempting to load a GAN from
    an invalid model path."""
    with pytest.raises(FileNotFoundError):
        _ = model_generate.get_gan('dne')


@pytest.mark.slowtest
def test_main_shows_image() -> None:
    """Tests that main shows an image to the screen."""
    try:
        model_generate.main()
    except FileNotFoundError:
        train_model_main()
        model_generate.main()
    fig = plt.gcf()
    assert fig is not None
