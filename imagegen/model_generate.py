"""Loads a GAN from VersionedModels and uses it to generate new images."""

import os
import matplotlib.pyplot as plt
from mlops.model.versioned_model import VersionedModel
from imagegen.gan import GAN
from imagegen.train_model import MODEL_PUBLICATION_PATH_LOCAL

NUM_SAMPLE_ROWS = 8
NUM_SAMPLE_COLS = 8


def get_gan(versioned_model_base_path: str) -> GAN:
    """Returns the GAN at the given base path.

    :param versioned_model_base_path: The base path to the versioned models.
        The base path is the path to the directory containing two
        VersionedModels, generator and discriminator; the base path should have
        two subdirectories, "generator" and "discriminator", each of which
        contain a published VersionedModel.
    :return: The GAN at the given base path.
    """
    versioned_generator_base_path = os.path.join(versioned_model_base_path,
                                                 'generator')
    versioned_generator_path = os.path.join(
        versioned_generator_base_path,
        os.listdir(versioned_generator_base_path)[0])
    versioned_generator = VersionedModel(versioned_generator_path)
    versioned_discriminator_base_path = os.path.join(versioned_model_base_path,
                                                     'discriminator')
    versioned_discriminator_path = os.path.join(
        versioned_discriminator_base_path,
        os.listdir(versioned_discriminator_base_path)[0])
    versioned_discriminator = VersionedModel(versioned_discriminator_path)
    return GAN(versioned_generator.model, versioned_discriminator.model)


def main() -> None:
    """Runs the program."""
    model_filenames = sorted(os.listdir(MODEL_PUBLICATION_PATH_LOCAL))
    model_paths = [os.path.join(MODEL_PUBLICATION_PATH_LOCAL, filename)
                   for filename in model_filenames]
    most_recent = model_paths[-1]
    gan = get_gan(most_recent)
    images = gan.generate(NUM_SAMPLE_ROWS * NUM_SAMPLE_COLS)
    image_grid = GAN.concatenate_images(images, NUM_SAMPLE_ROWS,
                                        NUM_SAMPLE_COLS)
    plt.imshow(image_grid)
    plt.show()


if __name__ == '__main__':
    main()
