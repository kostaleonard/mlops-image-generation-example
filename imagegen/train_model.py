"""Trains a new model on the Pokemon generation task."""
# pylint: disable=no-name-in-module

import os
from typing import Optional
from datetime import datetime
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, \
    Conv2DTranspose, BatchNormalization, LeakyReLU
from mlops.errors import PublicationPathAlreadyExistsError
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model_builder import VersionedModelBuilder
from mlops.model.training_config import TrainingConfig
from imagegen.publish_dataset import publish_dataset, \
    DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION
from imagegen.gan import GAN

MODEL_PUBLICATION_PATH_LOCAL = os.path.join('models', 'versioned')
MODEL_PUBLICATION_PATH_S3 = \
    's3://kosta-mlops/mlops-image-generation-example/models'
TAGS = ['baseline']
DEFAULT_GEN_INPUT_DIM = 128


def _get_baseline_gan_generator() -> Model:
    """Returns a new instance of the baseline GAN's generator network.

    :return: A new instance of the baseline GAN's generator network.
    """
    generator = Sequential()
    # Shape: (None, generator_input_dim).
    generator.add(Dense(15 * 15 * 16,
                        input_shape=(DEFAULT_GEN_INPUT_DIM,)))
    # Shape: (None, 3600).
    generator.add(Reshape(target_shape=(15, 15, 16)))
    # Shape: (None, 15, 15, 16)
    generator.add(Conv2DTranspose(16,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same'))
    # Shape: (None, 30, 30, 16).
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.1))
    generator.add(Conv2DTranspose(8,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same'))
    # Shape: (None, 60, 60, 8).
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.1))
    generator.add(Conv2DTranspose(4,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same'))
    # Shape: (None, 120, 120, 4).
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.1))
    generator.add(Conv2DTranspose(3,
                                  kernel_size=3,
                                  activation='sigmoid',
                                  strides=1,
                                  padding='same'))
    # Shape: (None, 120, 120, 3).
    generator.compile()
    return generator


def _get_baseline_gan_discriminator(dataset: VersionedDataset) -> Model:
    """Returns a new instance of the baseline GAN's discriminator network.

    :param dataset: The input dataset. Used to determine model input and output
        shapes.
    :return: A new instance of the baseline GAN's discriminator network.
    """
    discriminator = Sequential()
    # Shape: (None, 120, 120, 3).
    discriminator.add(
        Conv2D(4, (3, 3), activation='relu', padding='same', strides=2,
               input_shape=dataset.X_train.shape[1:]))
    # Shape: (None, 60, 60, 4).
    discriminator.add(
        Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    # Shape: (None, 30, 30, 8).
    discriminator.add(
        Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    # Shape: (None, 15, 15, 16).
    discriminator.add(
        Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    # Shape: (None, 8, 8, 32).
    discriminator.add(
        Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    # Shape: (None, 4, 4, 64).
    discriminator.add(Flatten())
    # Shape: (None, 1024).
    discriminator.add(Dense(1, activation='sigmoid'))
    # Shape: (None, 1).
    discriminator.compile()
    return discriminator


def get_baseline_gan(dataset: VersionedDataset) -> GAN:
    """Returns a new GAN for use on the dataset. This model is only a baseline;
    developers should also experiment with custom models in notebook
    environments.

    :param dataset: The input dataset. Used to determine model input and output
        shapes.
    :return: A new GAN for use on the dataset.
    """
    generator = _get_baseline_gan_generator()
    discriminator = _get_baseline_gan_discriminator(dataset)
    return GAN(generator, discriminator)


def publish_gan(gan: GAN,
                dataset: VersionedDataset,
                training_config: TrainingConfig,
                publication_path: str,
                tags: Optional[list[str]] = None) -> str:
    """Publishes the GAN to the path on the local or remote filesystem.

    :param gan: The GAN to be published, with the exact weights desired for
        publication (the user needs to set the weights to the best found during
        training if that is what they desire). The generator and discriminator
        will be published in two separate but related VersionedModels.
    :param dataset: The input dataset.
    :param training_config: The training configuration.
    :param publication_path: The path to which the model will be published.
    :param tags: Optional tags for the published model.
    :return: The base path to the versioned generator and discriminator models.
    """
    tags = tags or []
    timestamp = datetime.now().isoformat()
    base_path = os.path.join(publication_path, timestamp)
    gen_publication_path = os.path.join(base_path, 'generator')
    dis_publication_path = os.path.join(base_path, 'discriminator')
    gen_builder = VersionedModelBuilder(dataset, gan.generator, training_config)
    dis_builder = VersionedModelBuilder(dataset, gan.discriminator,
                                        training_config)
    gen_builder.publish(gen_publication_path, tags=tags + ['generator'])
    dis_builder.publish(dis_publication_path, tags=tags + ['discriminator'])
    return base_path


def main() -> None:
    """Runs the program."""
    try:
        dataset_path = publish_dataset(DATASET_PUBLICATION_PATH_LOCAL)
    except PublicationPathAlreadyExistsError:
        dataset_path = os.path.join(DATASET_PUBLICATION_PATH_LOCAL,
                                    DATASET_VERSION)
    dataset = VersionedDataset(dataset_path)
    gan = get_baseline_gan(dataset)
    training_config = gan.train(
        dataset,
        use_wandb=False,
        batch_size=32,
        epochs=5
    )
    publish_gan(
        gan,
        dataset,
        training_config,
        MODEL_PUBLICATION_PATH_LOCAL,
        tags=TAGS)


if __name__ == '__main__':
    main()
