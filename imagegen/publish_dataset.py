"""Publishes a new dataset to the local or remote filesystem. This script should
be run any time the data processor changes."""

from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder
from imagegen.pokemon_generation_data_processor import \
    PokemonGenerationDataProcessor, DEFAULT_DATASET_PATH

DATASET_VERSION = 'v2'
DATASET_PUBLICATION_PATH_LOCAL = 'datasets'
DATASET_PUBLICATION_PATH_S3 = \
    's3://kosta-mlops/mlops-image-generation-example/datasets'
TAGS = ['image', 'generation']


def publish_dataset(publication_path: str) -> str:
    """Builds and publishes the dataset.

    :param publication_path: The path on the local or remote filesystem to which
        to publish the dataset.
    :return: The versioned dataset's publication path.
    """
    processor = PokemonGenerationDataProcessor()
    builder = VersionedDatasetBuilder(DEFAULT_DATASET_PATH, processor)
    return builder.publish(publication_path, DATASET_VERSION, tags=TAGS)


def main() -> None:
    """Runs the program."""
    publish_dataset(DATASET_PUBLICATION_PATH_LOCAL)


if __name__ == '__main__':
    main()
