"""Tests pokemon_generation_data_processor.py."""

import pytest
import numpy as np
from imagegen.pokemon_generation_data_processor import \
    PokemonGenerationDataProcessor, DEFAULT_DATASET_PATH, HEIGHT, WIDTH, \
    CHANNELS
from imagegen.errors import AttemptToUseLabelsError

EXPECTED_FEATURE_SHAPE = HEIGHT, WIDTH, CHANNELS
MAX_BACKGROUND_GRAYSCALE = 10 / 255


def test_get_raw_features_and_labels_returns_expected_keys() -> None:
    """Tests that get_raw_features_and_labels returns the expected keys for the
    train/val/test dataset."""
    processor = PokemonGenerationDataProcessor()
    features, _ = processor.get_raw_features_and_labels(DEFAULT_DATASET_PATH)
    assert set(features.keys()) == {'X_train', 'X_val', 'X_test'}


def test_get_raw_features_and_labels_trainvaltest_correct_split() -> None:
    """Tests that the train/val/test datasets are split into non-empty
    subsets."""
    processor = PokemonGenerationDataProcessor()
    features, _ = processor.get_raw_features_and_labels(DEFAULT_DATASET_PATH)
    assert len(features['X_train']) > 0
    assert len(features['X_val']) > 0
    assert len(features['X_test']) > 0


def test_get_raw_features_match() -> None:
    """Tests that the features produced by get_raw_features_and_labels and
    get_raw_features are the same features."""
    processor = PokemonGenerationDataProcessor()
    features, _ = processor.get_raw_features_and_labels(DEFAULT_DATASET_PATH)
    features_only = processor.get_raw_features(DEFAULT_DATASET_PATH)
    assert set(features.keys()) == set(features_only.keys())
    for name, feature_tensor in features.items():
        assert np.array_equal(feature_tensor, features_only[name])


def test_get_raw_features_correct_shape() -> None:
    """Tests that get_raw_features returns tensors with the expected shapes."""
    processor = PokemonGenerationDataProcessor()
    features = processor.get_raw_features(DEFAULT_DATASET_PATH)
    for feature_tensor in features.values():
        assert feature_tensor.shape[1:] == EXPECTED_FEATURE_SHAPE


def test_get_raw_features_correct_dtype() -> None:
    """Tests that get_raw_features returns tensors with dtype float32."""
    processor = PokemonGenerationDataProcessor()
    features = processor.get_raw_features(DEFAULT_DATASET_PATH)
    for feature_tensor in features.values():
        assert feature_tensor.dtype == np.float32


def test_get_raw_features_correct_value_range() -> None:
    """Tests that get_raw_features returns tensors in the range [0, 1]."""
    processor = PokemonGenerationDataProcessor()
    features = processor.get_raw_features(DEFAULT_DATASET_PATH)
    for feature_tensor in features.values():
        assert feature_tensor.min() >= 0
        assert feature_tensor.max() <= 1


def test_get_raw_features_no_na() -> None:
    """Tests that get_raw_features returns tensors with no missing values."""
    processor = PokemonGenerationDataProcessor()
    features = processor.get_raw_features(DEFAULT_DATASET_PATH)
    for feature_tensor in features.values():
        assert not np.isnan(feature_tensor).any()


def test_get_raw_features_have_multiple_pixel_values() -> None:
    """Tests that the images were loaded correctly by ensuring that more than
    one pixel value exists in the tensors."""
    processor = PokemonGenerationDataProcessor()
    features = processor.get_raw_features(DEFAULT_DATASET_PATH)
    for feature_tensor in features.values():
        assert len(np.unique(feature_tensor)) > 1


def test_get_raw_labels_empty() -> None:
    """Tests that the returned labels are the empty dictionary."""
    processor = PokemonGenerationDataProcessor()
    _, labels = processor.get_raw_features_and_labels(DEFAULT_DATASET_PATH)
    assert isinstance(labels, dict)
    assert not labels


def test_preprocessed_features_same_as_raw() -> None:
    """Tests that the preprocessed features are the same as the raw features."""
    processor = PokemonGenerationDataProcessor()
    features_raw = processor.get_raw_features(DEFAULT_DATASET_PATH)
    features_preprocessed = processor.get_preprocessed_features(
        DEFAULT_DATASET_PATH)
    assert set(features_raw.keys()) == set(features_preprocessed.keys())
    for name, tensor_raw in features_raw.items():
        assert np.array_equal(tensor_raw, features_preprocessed[name])


def test_preprocess_labels_raises_error() -> None:
    """Tests that preprocess_labels raises an error, since there are no labels
    for this task."""
    processor = PokemonGenerationDataProcessor()
    fake_labels = np.ones((5, 4))
    with pytest.raises(AttemptToUseLabelsError):
        _ = processor.preprocess_labels(fake_labels)


def test_unpreprocess_features_inverts_transformation() -> None:
    """Tests that unpreprocessing the preprocessed features results in the raw
    features."""
    processor = PokemonGenerationDataProcessor()
    features_raw = processor.get_raw_features(DEFAULT_DATASET_PATH)
    for tensor_raw in features_raw.values():
        tensor_preprocessed = processor.preprocess_features(tensor_raw)
        tensor_unpreprocessed = processor.unpreprocess_features(
            tensor_preprocessed)
        assert np.array_equal(tensor_raw, tensor_unpreprocessed)


def test_unpreprocess_labels_raises_error() -> None:
    """Tests that unpreprocess_labels raises an error, since there are no labels
    for this task."""
    processor = PokemonGenerationDataProcessor()
    fake_labels = np.ones((5, 4))
    with pytest.raises(AttemptToUseLabelsError):
        _ = processor.unpreprocess_labels(fake_labels)


def test_images_have_black_background() -> None:
    """Tests that input images have a black (not white) background."""
    processor = PokemonGenerationDataProcessor()
    features = processor.get_preprocessed_features(DEFAULT_DATASET_PATH)
    images_upper_left_corner = features['X_train'][:, :5, :5, :]
    rgb_mean = images_upper_left_corner.mean(axis=-1)
    assert rgb_mean.max() <= MAX_BACKGROUND_GRAYSCALE
