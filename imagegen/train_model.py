"""Trains a new model on the Pokemon generation task."""
# pylint: disable=no-name-in-module

import os
from typing import Optional
from datetime import datetime
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, \
    Conv2DTranspose, BatchNormalization, LeakyReLU, Cropping2D, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.activations import linear
from mlops.errors import PublicationPathAlreadyExistsError
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model_builder import VersionedModelBuilder
from mlops.model.training_config import TrainingConfig
from imagegen.publish_dataset import publish_dataset, \
    DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION
from imagegen.gan import GAN
from imagegen.wgan import WGAN, DEFAULT_RMSPROP_LR

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
    generator.add(Dense(4 * 4 * 512,
                        use_bias=False,
                        input_shape=(DEFAULT_GEN_INPUT_DIM,)))
    # Shape: (None, 8192).
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.1))
    generator.add(Reshape(target_shape=(4, 4, 512)))
    # Shape: (None, 4, 4, 512)
    generator.add(Conv2DTranspose(256,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same'))
    # Shape: (None, 8, 8, 256).
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.1))
    generator.add(Conv2DTranspose(128,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same'))
    # Shape: (None, 16, 16, 128).
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.1))
    generator.add(Conv2DTranspose(64,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same'))
    # Shape: (None, 32, 32, 64).
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.1))
    generator.add(Conv2DTranspose(32,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same'))
    # Shape: (None, 64, 64, 32).
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.1))
    generator.add(Conv2DTranspose(16,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same'))
    # Shape: (None, 128, 128, 16).
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.1))
    generator.add(Conv2DTranspose(3,
                                  kernel_size=3,
                                  activation='sigmoid',
                                  strides=1,
                                  padding='same'))
    # Shape: (None, 128, 128, 3).
    generator.add(Cropping2D((4, 4)))
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
        Conv2D(128, (3, 3), activation='relu', padding='same', strides=2,
               input_shape=dataset.X_train.shape[1:]))
    # Shape: (None, 60, 60, 128).
    discriminator.add(
        Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    # Shape: (None, 30, 30, 256).
    discriminator.add(
        Conv2D(512, (3, 3), activation='relu', padding='same', strides=2))
    discriminator.add(Dropout(0.3))
    # Shape: (None, 15, 15, 512).
    discriminator.add(
        Conv2D(512, (3, 3), activation='relu', padding='same', strides=2))
    discriminator.add(Dropout(0.3))
    # Shape: (None, 8, 8, 512).
    discriminator.add(
        Conv2D(1024, (3, 3), activation='relu', padding='same', strides=2))
    # Shape: (None, 4, 4, 1024).
    discriminator.add(Flatten())
    # Shape: (None, 16384).
    discriminator.add(Dropout(0.2))
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


def get_baseline_wgan(dataset: VersionedDataset) -> WGAN:
    """Returns a new WGAN for use on the dataset. This model is only a baseline;
    developers should also experiment with custom models in notebook
    environments.

    :param dataset: The input dataset. Used to determine model input and output
        shapes.
    :return: A new WGAN for use on the dataset.
    """
    generator = _get_baseline_gan_generator()
    gen_optimizer = RMSprop(learning_rate=DEFAULT_RMSPROP_LR, momentum=0)
    generator.compile(optimizer=gen_optimizer)
    discriminator = _get_baseline_gan_discriminator(dataset)
    discriminator.layers[-1].activation = linear
    dis_optimizer = RMSprop(learning_rate=DEFAULT_RMSPROP_LR, momentum=0)
    discriminator.compile(optimizer=dis_optimizer)
    return WGAN(generator, discriminator)


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
    gan = get_baseline_wgan(dataset)
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
